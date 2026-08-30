#!/usr/bin/env python3
"""
Calibrate v2e or DVS-Voltmeter against measured event data.

Why this external calibration layer exists
------------------------------------------
The simulators map frames and parameters to synthetic events; they cannot
determine whether those events match a physical camera. CAROECT-D therefore
tunes published model inputs without forking or changing either simulator's
physics. Use a real sparse events.h5 recording and a same-scene, same-time
linear-Y TIFF sequence from preprocess.py.

The baseline is coordinate search: change one parameter at a time, simulate a
short prefix, slice real data to the same duration, and minimize
abs(log(rate_sim/rate_real)) + 0.5*abs(on_fraction_sim-on_fraction_real).
The command prints every candidate and writes only when --apply is explicit.

Timestamp evidence and controlled metrics
-----------------------------------------
Current Metavision RAW recordings contain precise per-event sensor timestamps.
Retired Arena .cevt recordings may contain device-buffer, host-arrival, or
synthesized timestamps and can be quantized to accumulation windows. Total
rate and ON fraction need only counts; edge latency and inter-event intervals
require precise, unquantized timestamps plus a known controlled stimulus.
check_timestamp_precision() reads provenance attributes rather than trusting a
filename or operator claim. Missing provenance is a refusal, not approval.

The metric API records any requested subset of rate_total, rate_static,
rate_motion, on_fraction, edge_latency_us, and IEI summary/distribution. Road
footage never supplies timing evidence implicitly.

Optuna/TPE
----------
The lazy --optuna path remains an experimental, proposed single-parameter
extension. It uses the same baseline score and is not required for a
paper-aligned run. Install Optuna only when explicitly testing that extension.

Usage:
  python calibrate_simulator.py --real events_real.h5 --sim-input data/processed/site01 \
      --simulator dvsvolt --param k1 --search 3.0 4.0 5.3 6.5 8.0 --limit 120

  python calibrate_simulator.py --real events_real.h5 --sim-input data/processed/site01 \
      --simulator v2e --param pos_thres --search 0.15 0.20 0.25 0.30 --limit 120

  # Experimental single-parameter Optuna/TPE
  python calibrate_simulator.py --real events_real.h5 --sim-input data/processed/site01 \
      --simulator v2e --param pos_thres --optuna --low 0.10 --high 0.40 --n-trials 25
"""

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile
import yaml

try:
    import h5py
except ImportError:
    h5py = None

from measure_event_rate import load_events_h5, analyze  # reuse, don't duplicate


# ══════════════════════════════════════════════════════════════════════════
#  Valid simulator parameters and their config.yaml locations
# ══════════════════════════════════════════════════════════════════════════

DVSVOLT_K_INDEX = {"k1": 0, "k2": 1, "k3": 2, "k4": 3, "k5": 4, "k6": 5}
DVSVOLT_RATE_SAFE = {"k1", "k2", "k4", "k5"}   # Event-count effects; timing is not needed.
DVSVOLT_TIMING_ONLY = {"k3", "k6"}              # Noise/jitter effects need precise timing.
# k2 xuất hiện ở CẢ μ (Eq.27: μ = k1/(L+k2)·k_dL + k4 + k5·L, rate) LẪN σ
# (sigma = k3/(L+k2)*sqrt(L) + k6, jitter), so rate-only calibration may tune
# its drift role but cannot validate its jitter role. main() emits that caveat.
DVSVOLT_DUAL_ROLE = {"k2"}

V2E_RATE_SAFE = {"pos_thres", "neg_thres", "refractory_period"}
V2E_TIMING_ONLY = {"sigma_thres", "cutoff_hz", "leak_rate_hz", "shot_noise_rate_hz"}

RUNNERS = {
    # simulator -> (module/script name, conda-env key in config.simulator.envs)
    "dvsvolt": ("run_dvsvolt.py", "dvsvolt"),
    "v2e":     ("run_v2e.py", "v2e"),
}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _h5_attr_to_str(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


# Vocabulary of attrs["timestamp_precision_status"], written by raw_to_events.py
# (current recorder path) or legacy/cevt_to_events.py (retired Arena path).
#   precise       : real per-event microsecond sensor timestamp, no accumulation
#                   window at all -- written by raw_to_events.py (Metavision/.raw,
#                   the only path currently entitled to this value).
#   device_buffer : t comes from the camera's own buffer clock — measured, but on
#                   the legacy Arena/.cevt path every event in one accumulated
#                   frame shares this one buffer timestamp (t_quantization_us > 0).
#   host_arrival  : t comes from host buffer-arrival time — measured, but carries
#                   network + scheduling jitter, so not trustworthy at µs scale.
#   synthesized   : t was computed from frame_id x window length — a guess.
#   unknown       : the attribute is absent.
#
# "precise" and "device_buffer" are both measured; only "precise" (real per-event
# t, t_quantization_us == 0) is tight enough for TIMING_ONLY / jitter calibration.
# "device_buffer" with t_quantization_us > 0 is fine for RATE_SAFE params only.
_TS_STATUS_MEASURED_TIGHT = {"device_buffer", "precise"}
_TS_STATUS_MEASURED_LOOSE = {"host_arrival"}


def check_timestamp_precision(real_h5: Path, require_precise: bool, purpose: str = "timing"):
    """
    Guardrail against calibrating physics on invented timestamps.

    Two things were wrong with the previous version:

    1. It accepted "unknown" (i.e. the attribute missing) as good enough. The old
       cevt_to_events.py never wrote the attribute at all, so every real
       recording read as "unknown" and sailed straight through — the guard was
       inert precisely when it mattered. Missing metadata is now a refusal, not
       a pass; you cannot certify data you know nothing about.

    2. It checked only the status string, ignoring t_quantization_us. On this
       camera t is quantised to the accumulation window even when the clock is a
       genuine device clock, because a dense accumulated frame gives every event
       in it the same time. So a "device_buffer" file is fine for RATE and for
       window alignment, and still useless for per-event jitter (k3/k6) work.
       purpose="jitter" now rejects any quantised file regardless of clock.
    """
    with h5py.File(real_h5, "r") as hf:
        status = _h5_attr_to_str(hf.attrs.get("timestamp_precision_status", "unknown"))
        zero_dt = float(hf.attrs.get("timestamp_zero_dt_fraction", 0.0))
        quant_us = float(hf.attrs.get("t_quantization_us", 0.0))

    if not require_precise:
        return status, zero_dt

    hint = (f"\n  Record with evs_recorder.cpp/Metavision -> .raw and convert with\n"
            f"  raw_to_events.py for status='precise' and unquantized sensor time.\n"
            f"  For a retired Arena .cevt file, diagnose timestamp provenance with:\n"
            f"    python legacy/cevt_to_events.py <file>.cevt --debug-time-continuity\n"
            f"  Reconvert files produced by old converters so provenance attributes exist.")

    if status == "unknown":
        raise RuntimeError(
            f"{real_h5} has no timestamp_precision_status attribute. Timestamp "
            f"evidence is unknown, so it cannot be used for {purpose} calibration.{hint}")

    if status not in (_TS_STATUS_MEASURED_TIGHT | _TS_STATUS_MEASURED_LOOSE):
        raise RuntimeError(
            f"{real_h5} has timestamp_precision_status={status!r} "
            f"(zero_dt_fraction={zero_dt:.4f}); timestamps are not measured and "
            f"cannot support {purpose} calibration.{hint}")

    if status in _TS_STATUS_MEASURED_LOOSE:
        raise RuntimeError(
            f"{real_h5} has timestamp_precision_status={status!r}: these are host "
            f"buffer-arrival times with transport/scheduling jitter. They support "
            f"event rate, not microsecond {purpose} calibration. Use device time or "
            f"explicitly accept degraded timing with --allow-degraded-t.{hint}")

    if purpose == "jitter" and quant_us > 0:
        raise RuntimeError(
            f"{real_h5} has t_quantization_us={quant_us:.1f}; every event in an "
            f"accumulation window shares one timestamp (zero_dt_fraction={zero_dt:.4f}). "
            "The inter-event-time distribution was not observed, so noise/jitter "
            "parameters cannot be calibrated from this hardware-limited recording.")

    return status, zero_dt


# ══════════════════════════════════════════════════════════════════════════
#  N1 · Prepare a temporary config with exactly one changed parameter
# ══════════════════════════════════════════════════════════════════════════

def make_candidate_config(base_cfg: dict, simulator: str, param: str, value: float) -> dict:
    cfg = copy.deepcopy(base_cfg)
    if simulator == "dvsvolt":
        idx = DVSVOLT_K_INDEX[param]
        cfg["dvs_voltmeter"]["k"][idx] = float(value)
    elif simulator == "v2e":
        cfg["v2e"][param] = float(value)
    else:
        raise ValueError(f"Unknown simulator: {simulator}")
    return cfg


# ══════════════════════════════════════════════════════════════════════════
#  N2 · Run one candidate in its simulator environment
# ══════════════════════════════════════════════════════════════════════════

def run_candidate(simulator: str, sim_input: Path, work_dir: Path, cfg: dict,
                   limit: int, tag: str) -> Path:
    script, env_key = RUNNERS[simulator]
    if not Path(script).exists():
        raise FileNotFoundError(
            f"{script} does not exist in the current directory; simulator "
            f"{simulator!r} cannot be calibrated without its runner.")

    cand_dir = work_dir / tag
    cand_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cand_dir / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    out_dir = cand_dir / "out"
    env_name = cfg.get("simulator", {}).get("envs", {}).get(env_key, env_key)
    conda = shutil.which("conda") or "conda"
    cmd = [conda, "run", "-n", env_name, "python", script,
           "--input", str(sim_input), "--output", str(out_dir),
           "--config", str(cfg_path), "--limit", str(limit)]
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"{script} failed for candidate {tag}; see the log above.")

    h5path = out_dir / "events.h5"
    if not h5path.exists():
        raise FileNotFoundError(f"{script} completed without producing {h5path}")
    return h5path


# ══════════════════════════════════════════════════════════════════════════
#  N3 · Compare simulated and real events over the same duration
# ══════════════════════════════════════════════════════════════════════════

def slice_real_to_window(real_h5: Path, duration_s: float, sensor_w: int, sensor_h: int):
    """Slice real events to one duration and reuse the shared analyzer."""
    ev = load_events_h5(real_h5)
    if len(ev["t"]) == 0:
        raise ValueError(f"{real_h5} is empty and cannot be compared.")
    t0 = ev["t"].min()
    mask = ev["t"] <= (t0 + duration_s * 1e6)
    sliced = {k: v[mask] for k, v in ev.items()}

    tmp_path = real_h5.parent / f"_tmp_slice_{int(duration_s*1000)}ms.h5"
    with h5py.File(tmp_path, "w") as hf:
        for k in ("x", "y", "t", "p"):
            hf.create_dataset(k, data=sliced[k])
    stats = analyze(str(tmp_path), "real(sliced)", sensor_w, sensor_h, window_s=max(0.1, duration_s / 5))
    tmp_path.unlink(missing_ok=True)
    if stats is None:
        raise ValueError(
            "The sliced real window is empty; check duration and measured clock alignment.")
    return stats


# ══════════════════════════════════════════════════════════════════════════
#  N3b · Eq.30 physical contrast-threshold calibration
# ══════════════════════════════════════════════════════════════════════════

def sorted_tiff_paths(folder: Path):
    exts = ("*.tif", "*.tiff", "*.png")
    paths = []
    for ext in exts:
        paths.extend(folder.glob(ext))
    return sorted(paths)


def read_linear_luma(path: Path, cfg: dict) -> np.ndarray:
    img = tifffile.imread(str(path)).astype(np.float32)
    if img.ndim == 2:
        return img
    coeffs = np.asarray(cfg.get("camera", {}).get("luma_coeffs", [0.2126, 0.7152, 0.0722]),
                        dtype=np.float32)
    return np.tensordot(img[..., :3], coeffs, axes=([-1], [0])).astype(np.float32)


def count_interval_events(events: dict, t0: float, t1: float, width: int, height: int):
    lo = int(np.searchsorted(events["t"], t0, side="left"))
    hi = int(np.searchsorted(events["t"], t1, side="left"))
    x = events["x"][lo:hi].astype(np.int64)
    y = events["y"][lo:hi].astype(np.int64)
    p = events["p"][lo:hi] > 0
    valid = (0 <= x) & (x < width) & (0 <= y) & (y < height)
    x, y, p = x[valid], y[valid], p[valid]

    on = np.zeros((height, width), dtype=np.uint16)
    off = np.zeros((height, width), dtype=np.uint16)
    if len(x):
        np.add.at(on, (y[p], x[p]), 1)
        np.add.at(off, (y[~p], x[~p]), 1)
    return on, off


def estimate_eq30_thresholds(real_h5: Path, frame_dir: Path, cfg: dict, limit: int,
                             min_dlog: float, min_events: int, time_offset_us: float = 0.0):
    """Estimate C_real = ΔlogL / Nbar(ΔlogL) from a calibrated gray-gradient/ramp clip.

    For each adjacent frame pair, compute per-pixel ΔlogL, count real ON/OFF
    events in the matching timestamp interval, then use the paper's physical
    relation: expected event count ≈ |ΔlogL| / C. Summing over active pixels
    gives robust ON/OFF threshold estimates.
    """
    events = load_events_h5(real_h5)
    if len(events["t"]) == 0:
        raise ValueError(f"{real_h5} is empty; Eq.30 cannot be estimated.")

    paths = sorted_tiff_paths(frame_dir)
    if len(paths) < 2:
        raise FileNotFoundError(f"Eq. 30 requires at least two TIFF/PNG frames in {frame_dir}.")
    if limit:
        paths = paths[:max(2, min(limit, len(paths)))]

    fps = float(cfg["camera"]["fps_original"])
    width, height = int(cfg["camera"]["width"]), int(cfg["camera"]["height"])
    eps = 1.0
    t_origin = float(events["t"].min()) + float(time_offset_us)

    total_pos_dlog = 0.0
    total_neg_dlog = 0.0
    total_on = 0
    total_off = 0
    rows = []

    prev = read_linear_luma(paths[0], cfg)
    if prev.shape != (height, width):
        raise ValueError(f"{paths[0]} shape={prev.shape}, config camera={height}x{width}")

    for i in range(len(paths) - 1):
        nxt = read_linear_luma(paths[i + 1], cfg)
        if nxt.shape != prev.shape:
            raise ValueError(f"Frame shape mismatch: {paths[i]} {prev.shape} vs {paths[i+1]} {nxt.shape}")

        dlog = np.log(np.clip(nxt, eps, None)) - np.log(np.clip(prev, eps, None))
        pos_pixels = dlog > min_dlog
        neg_pixels = dlog < -min_dlog
        t0 = t_origin + i * 1e6 / fps
        t1 = t_origin + (i + 1) * 1e6 / fps
        on, off = count_interval_events(events, t0, t1, width, height)

        pos_dlog = float(dlog[pos_pixels].sum())
        neg_dlog = float((-dlog[neg_pixels]).sum())
        n_on = int(on[pos_pixels].sum())
        n_off = int(off[neg_pixels].sum())

        if n_on >= min_events and pos_dlog > 0:
            c_pos = pos_dlog / n_on
            total_pos_dlog += pos_dlog
            total_on += n_on
        else:
            c_pos = None
        if n_off >= min_events and neg_dlog > 0:
            c_neg = neg_dlog / n_off
            total_neg_dlog += neg_dlog
            total_off += n_off
        else:
            c_neg = None
        rows.append(dict(
            frame_i=i,
            t_start_us=t0,
            t_end_us=t1,
            pos_delta_log=pos_dlog,
            neg_delta_log=neg_dlog,
            n_on=n_on,
            n_off=n_off,
            c_pos=c_pos,
            c_neg=c_neg,
        ))
        prev = nxt

    c_pos = total_pos_dlog / total_on if total_on >= min_events and total_pos_dlog > 0 else None
    c_neg = total_neg_dlog / total_off if total_off >= min_events and total_neg_dlog > 0 else None
    return dict(
        method="Eq30_C_real_delta_logL_over_event_count",
        frame_dir=str(frame_dir),
        real_h5=str(real_h5),
        fps=fps,
        min_dlog=min_dlog,
        min_events=min_events,
        total_pos_delta_log=total_pos_dlog,
        total_neg_delta_log=total_neg_dlog,
        total_on_events=total_on,
        total_off_events=total_off,
        c_pos=c_pos,
        c_neg=c_neg,
        rows=rows,
    )


def apply_eq30_to_config(cfg: dict, simulator: str, estimate: dict):
    out = copy.deepcopy(cfg)
    if simulator != "v2e":
        raise ValueError(
            "Direct Eq.30 application maps only to v2e pos_thres/neg_thres. "
            "DVS-Voltmeter uses this as a physical target while k1..k6 remain "
            "closed-loop search parameters.")
    if estimate["c_pos"] is not None:
        out["v2e"]["pos_thres"] = float(estimate["c_pos"])
    if estimate["c_neg"] is not None:
        out["v2e"]["neg_thres"] = float(estimate["c_neg"])
    return out


def run_optuna_search(simulator: str, param: str, low: float, high: float, n_trials: int,
                       base_cfg: dict, sim_input: Path, work_dir: Path, limit: int,
                       W: int, H: int, duration_s: float, real_stats: dict):
    """Experimental TPE search using the same baseline candidate score."""
    try:
        import optuna  # Lazy import: the baseline has no Optuna dependency.
    except ImportError as e:
        raise ImportError(
            "Optuna is not installed. Install it in the host calibration environment "
            "only when testing the experimental --optuna path.") from e

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rows = []

    def objective(trial: "optuna.Trial") -> float:
        value = trial.suggest_float(param, low, high)
        tag = f"optuna_trial{trial.number}"
        cfg = make_candidate_config(base_cfg, simulator, param, value)
        try:
            h5path = run_candidate(simulator, sim_input, work_dir, cfg, limit, tag)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  [trial {trial.number}] [skip] {e}")
            raise optuna.TrialPruned()
        sim_stats = analyze(str(h5path), f"sim(trial{trial.number})", W, H,
                            window_s=max(0.1, duration_s / 5))
        if sim_stats is None:
            raise optuna.TrialPruned()
        err = score_candidate(sim_stats, real_stats)
        rows.append((value, sim_stats["rate_hz"], sim_stats["n_on"] / max(sim_stats["n"], 1), err))
        print(f"  [trial {trial.number}] {param}={value:.5f}  rate={sim_stats['rate_hz']:,.0f} ev/s  "
              f"error={err:.4f}")
        return err

    # TPESampler proposes later trials from earlier results. Every trial still
    # launches the real simulator runner and uses the named baseline score.
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    print(f"\n[optuna] best {param} = {study.best_params[param]:.5f}  "
          f"(error={study.best_value:.4f}, {len(study.trials)} trial(s))")
    return rows


def score_candidate(sim_stats: dict, real_stats: dict) -> float:
    """Named baseline: log-rate mismatch plus half-weight ON-fraction mismatch."""
    rate_term = abs(np.log(max(sim_stats["rate_hz"], 1e-6) / max(real_stats["rate_hz"], 1e-6)))
    sim_on_frac = sim_stats["n_on"] / max(sim_stats["n"], 1)
    real_on_frac = real_stats["n_on"] / max(real_stats["n"], 1)
    onoff_term = abs(sim_on_frac - real_on_frac)
    return rate_term + 0.5 * onoff_term


def _events_in_intervals(events: dict, intervals) -> np.ndarray:
    selected = np.zeros(len(events["t"]), dtype=bool)
    for start_us, end_us in intervals:
        selected |= (events["t"] >= float(start_us)) & (events["t"] < float(end_us))
    return selected


def build_metric_vector(h5_path: Path, stats: dict, requested: list[str],
                        reference: dict | None, source_key: str) -> dict:
    """Build the traceable metric API without inventing timing evidence."""
    vector = {
        "rate_total": float(stats["rate_hz"]),
        "on_fraction": float(stats["n_on"] / max(stats["n"], 1)),
    }
    if requested == ["rate_total", "on_fraction"] or set(requested) <= set(vector):
        return {key: vector[key] for key in requested}
    events = load_events_h5(h5_path)
    if reference is None:
        raise RuntimeError(
            "Requested controlled-data metrics require --stimulus-reference JSON.")
    for field in requested:
        if field in vector:
            continue
        if field in {"rate_static", "rate_motion"}:
            interval_key = f"{field.removeprefix('rate_')}_intervals_us"
            intervals = reference.get(source_key, {}).get(interval_key)
            if not intervals:
                raise RuntimeError(f"Stimulus reference is missing {source_key}.{interval_key}")
            mask = _events_in_intervals(events, intervals)
            duration_s = sum(float(end) - float(start) for start, end in intervals) / 1e6
            vector[field] = float(mask.sum() / max(duration_s, 1e-12))
        elif field == "edge_latency_us":
            transitions = reference.get(source_key, {}).get("transition_times_us")
            if not transitions:
                raise RuntimeError(
                    f"Stimulus reference is missing {source_key}.transition_times_us")
            latencies = []
            for transition in transitions:
                index = int(np.searchsorted(events["t"], float(transition), side="left"))
                if index < len(events["t"]):
                    latencies.append(float(events["t"][index]) - float(transition))
            if not latencies:
                raise RuntimeError("No event follows any controlled transition")
            vector[field] = float(np.median(latencies))
        elif field in {"iei_summary", "iei_distribution"}:
            dt = np.diff(events["t"].astype(np.float64))
            dt = dt[dt > 0]
            if not len(dt):
                raise RuntimeError("No positive inter-event intervals are available")
            vector[field] = {
                "median_us": float(np.median(dt)),
                "p90_us": float(np.percentile(dt, 90)),
                "mean_us": float(np.mean(dt)),
            }
        else:
            raise ValueError(f"Unknown calibration metric {field!r}")
    return {key: vector[key] for key in requested}


def score_metric_vectors(sim_vector: dict, real_vector: dict, weights: dict) -> float:
    """Weighted metric score; the baseline exactly matches score_candidate()."""
    total = 0.0
    for field, real_value in real_vector.items():
        weight = float(weights.get(field, 0.0))
        if not weight:
            continue
        sim_value = sim_vector[field]
        if field in {"rate_total", "rate_static", "rate_motion"}:
            term = abs(np.log(max(float(sim_value), 1e-6) / max(float(real_value), 1e-6)))
        elif field == "on_fraction":
            term = abs(float(sim_value) - float(real_value))
        elif field == "edge_latency_us":
            term = abs(float(sim_value) - float(real_value)) / max(abs(float(real_value)), 1.0)
        else:
            keys = ("median_us", "p90_us", "mean_us")
            term = float(np.mean([
                abs(np.log(max(float(sim_value[key]), 1e-6)
                           / max(float(real_value[key]), 1e-6)))
                for key in keys
            ]))
        total += weight * term
    return float(total)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True,
                    help="Measured events.h5 with timestamp provenance")
    ap.add_argument("--sim-input", required=True,
                    help="Same-scene, same-time linear-Y TIFF directory")
    ap.add_argument("--simulator", required=True, choices=["dvsvolt", "v2e"])
    ap.add_argument("--param", default=None,
                    help="dvsvolt: k1..k6 | v2e: pos_thres,neg_thres,sigma_thres,"
                         "cutoff_hz,leak_rate_hz,shot_noise_rate_hz")
    ap.add_argument("--eq30", action="store_true",
                    help="Estimate physical contrast threshold C_real = ΔlogL/Nbar(ΔlogL) "
                         "from a gray-gradient/ramp clip in --sim-input.")
    ap.add_argument("--eq30-output", default=None,
                    help="JSON report path for --eq30 (default: <work-dir>/eq30_estimate.json).")
    ap.add_argument("--eq30-min-dlog", type=float, default=1e-3,
                    help="Ignore per-pixel |ΔlogL| below this value.")
    ap.add_argument("--eq30-min-events", type=int, default=10,
                    help="Minimum ON/OFF events needed before reporting/applying a threshold.")
    ap.add_argument("--time-offset-us", type=float, default=0.0,
                    help="Offset added to real event t_min when aligning frame i to event time.")
    ap.add_argument("--search", type=float, nargs="+",
                    help="Explicit candidate values for the baseline grid search")
    ap.add_argument("--optuna", action="store_true",
                    help="Experimental single-parameter TPE search")
    ap.add_argument("--low", type=float, help="Experimental Optuna lower bound")
    ap.add_argument("--high", type=float, help="Experimental Optuna upper bound")
    ap.add_argument("--n-trials", type=int, default=20, help="Số trial optuna (default 20)")
    ap.add_argument("--limit", type=int, default=120,
                    help="Frame prefix used for each candidate")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--work-dir", default="_calib_work")
    ap.add_argument(
        "--metrics", nargs="+",
        choices=["rate_total", "rate_static", "rate_motion", "on_fraction",
                 "edge_latency_us", "iei_summary", "iei_distribution"],
        help="Metric vector fields; defaults to simulator.calibration_metrics.fields")
    ap.add_argument("--metric-weights-json",
                    help="Optional JSON object overriding configured metric weights")
    ap.add_argument("--stimulus-reference",
                    help="Controlled-stimulus JSON for static/motion/timing metrics")
    ap.add_argument("--report", help="Calibration report path (default: work-dir/calibration_report.json)")
    ap.add_argument("--require-precise-t", action="store_true", default=True,
                    help="Refuse Eq.30/timing calibration if H5 timestamp metadata says degraded.")
    ap.add_argument("--allow-degraded-t", action="store_false", dest="require_precise_t",
                    help="Override timestamp precision guardrail.")
    ap.add_argument("--apply", action="store_true",
                    help="Write the best value to config.yaml after creating a backup")
    args = ap.parse_args()

    if h5py is None:
        raise ImportError("pip install h5py")

    base_cfg = load_config(args.config)
    real_path = Path(args.real)
    metric_cfg = base_cfg.get("simulator", {}).get("calibration_metrics", {})
    requested_metrics = args.metrics or list(
        metric_cfg.get("fields", ["rate_total", "on_fraction"]))
    weights = dict(metric_cfg.get("weights", {"rate_total": 1.0, "on_fraction": 0.5}))
    if args.metric_weights_json:
        weights.update(json.loads(args.metric_weights_json))
    stimulus_reference = (
        json.loads(Path(args.stimulus_reference).read_text())
        if args.stimulus_reference else None)
    controlled_fields = {
        "rate_static", "rate_motion", "edge_latency_us",
        "iei_summary", "iei_distribution"}
    if set(requested_metrics) & controlled_fields:
        if not stimulus_reference or not stimulus_reference.get("controlled_stimulus", False):
            raise RuntimeError(
                "Static/motion/timing metrics require a controlled --stimulus-reference "
                "with controlled_stimulus=true.")
    timing_fields = {"edge_latency_us", "iei_summary", "iei_distribution"}
    if set(requested_metrics) & timing_fields:
        check_timestamp_precision(real_path, True, purpose="jitter")
    ts_status, zero_dt = check_timestamp_precision(
        real_path, args.require_precise_t and args.eq30, purpose="Eq.30")
    print(f"events_real.h5 timestamp_precision_status = {ts_status} "
          f"(zero_dt_fraction={zero_dt:.4f})")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.eq30:
        estimate = estimate_eq30_thresholds(
            real_path, Path(args.sim_input), base_cfg, args.limit,
            args.eq30_min_dlog, args.eq30_min_events, args.time_offset_us)
        out_path = Path(args.eq30_output) if args.eq30_output else work_dir / "eq30_estimate.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(estimate, indent=2))
        print(f"\n{'='*72}\nEq.30 physical calibration\n{'='*72}")
        print(f"  C_pos = {estimate['c_pos']}  from {estimate['total_on_events']:,} ON events")
        print(f"  C_neg = {estimate['c_neg']}  from {estimate['total_off_events']:,} OFF events")
        print(f"  report -> {out_path}")
        if args.apply:
            cfg_path = Path(args.config)
            backup = cfg_path.with_suffix(cfg_path.suffix + ".bak")
            shutil.copy(cfg_path, backup)
            final_cfg = apply_eq30_to_config(base_cfg, args.simulator, estimate)
            with open(cfg_path, "w") as f:
                yaml.safe_dump(final_cfg, f, sort_keys=False, allow_unicode=True)
            print(f"Wrote Eq.30 thresholds to {cfg_path} (backup: {backup})")
        return

    if args.param is None:
        raise ValueError("Closed-loop search requires --param; otherwise use --eq30.")

    if args.optuna:
        if args.low is None or args.high is None:
            raise ValueError("--optuna requires both --low and --high")
    elif not args.search:
        raise ValueError("Provide --search values or experimental --optuna bounds")

    rate_safe = DVSVOLT_RATE_SAFE if args.simulator == "dvsvolt" else V2E_RATE_SAFE
    timing_only = DVSVOLT_TIMING_ONLY if args.simulator == "dvsvolt" else V2E_TIMING_ONLY
    dual_role = DVSVOLT_DUAL_ROLE if args.simulator == "dvsvolt" else set()
    valid_params = rate_safe | timing_only
    if args.param not in valid_params:
        raise ValueError(
            f"--param {args.param!r} is invalid for {args.simulator}; "
            f"valid values: {sorted(valid_params)}")
    if args.param in timing_only:
        check_timestamp_precision(real_path, args.require_precise_t, purpose="jitter")
        print(
            f"[warning] {args.param!r} affects temporal noise/jitter. "
            "Trust it only with precise timestamp provenance.\n")
    if args.param in dual_role:
        # k2 affects both drift/rate and jitter. This RATE_SAFE path validates
        # only its rate role; a rate-matched value may still have wrong jitter.
        ts_status_for_warn, _ = check_timestamp_precision(real_path, require_precise=False)
        print(
            f"[warning] {args.param!r} affects both drift/rate and jitter in Eq.27. "
            "This search scores rate and ON fraction only; it does not validate "
            f"jitter (timestamp status: {ts_status_for_warn!r}).\n")

    fps = float(base_cfg["camera"]["fps_original"])
    W, H = int(base_cfg["camera"]["width"]), int(base_cfg["camera"]["height"])
    duration_s = args.limit / fps

    with h5py.File(real_path, "r") as hf:
        decode_counts = hf.attrs.get("decode_method_counts", "unknown")
    print(f"events_real.h5 decode_method_counts = {decode_counts}\n")

    real_stats = slice_real_to_window(real_path, duration_s, W, H)
    real_vector = build_metric_vector(
        real_path, real_stats, requested_metrics, stimulus_reference, "real")
    candidate_metric_reports = []

    mode_desc = (f"optuna TPE, {args.n_trials} trial(s) in [{args.low}, {args.high}]"
                 if args.optuna else f"{len(args.search)} candidate(s) (grid)")
    print(f"\n{'='*72}\nCalibrating {args.simulator}.{args.param}  |  {mode_desc}  |  "
          f"window={duration_s:.2f}s ({args.limit} frames @ {fps}fps)\n{'='*72}")

    if args.optuna:
        if requested_metrics != ["rate_total", "on_fraction"] or weights != {
                "rate_total": 1.0, "on_fraction": 0.5}:
            raise RuntimeError(
                "--optuna is an experimental single-parameter extension and currently "
                "supports only the named baseline metric vector.")
        # Experimental TPE selection; reporting and application stay shared.
        rows = run_optuna_search(args.simulator, args.param, args.low, args.high,
                                  args.n_trials, base_cfg, Path(args.sim_input), work_dir,
                                  args.limit, W, H, duration_s, real_stats)
    else:
        rows = []
        for value in args.search:
            tag = f"{args.param}_{value}".replace(".", "p")
            print(f"\n-- candidate {args.param}={value} --")
            cfg = make_candidate_config(base_cfg, args.simulator, args.param, value)
            try:
                h5path = run_candidate(args.simulator, Path(args.sim_input), work_dir, cfg,
                                        args.limit, tag)
            except (FileNotFoundError, RuntimeError) as e:
                print(f"  [skip] {e}")
                continue
            sim_stats = analyze(str(h5path), f"sim({value})", W, H, window_s=max(0.1, duration_s / 5))
            if sim_stats is None:
                print(f"  [skip] candidate sinh 0 event")
                continue
            sim_vector = build_metric_vector(
                h5path, sim_stats, requested_metrics, stimulus_reference, "sim")
            err = score_metric_vectors(sim_vector, real_vector, weights)
            candidate_metric_reports.append({
                "value": value, "metrics": sim_vector, "error": err,
                "events": str(h5path.resolve()),
            })
            rows.append((value, sim_stats["rate_hz"], sim_stats["n_on"] / max(sim_stats["n"], 1), err))
            print(f"  rate={sim_stats['rate_hz']:,.0f} ev/s  on_frac={sim_stats['n_on']/max(sim_stats['n'],1):.3f}  "
                  f"error={err:.4f}")

    if not rows:
        print("\nNo candidate completed successfully; inspect the log above.")
        sys.exit(1)
    if args.optuna:
        candidate_metric_reports = [{
            "value": value,
            "metrics": {"rate_total": rate, "on_fraction": on_fraction},
            "error": error,
        } for value, rate, on_fraction, error in rows]

    print(f"\n{'='*72}\nKẾT QUẢ  (real: rate={real_stats['rate_hz']:,.0f} ev/s  "
          f"on_frac={real_stats['n_on']/max(real_stats['n'],1):.3f})\n{'='*72}")
    print(f"{'value':>10}  {'rate_hz':>12}  {'on_frac':>8}  {'error':>8}")
    for value, rate, on_frac, err in sorted(rows, key=lambda r: r[3]):
        print(f"{value:>10}  {rate:>12,.0f}  {on_frac:>8.3f}  {err:>8.4f}")

    best_value, best_rate, best_on_frac, best_err = min(rows, key=lambda r: r[3])
    print(f"\n→ Tốt nhất: {args.param} = {best_value}  (error={best_err:.4f})")

    if args.apply:
        cfg_path = Path(args.config)
        backup = cfg_path.with_suffix(cfg_path.suffix + ".bak")
        shutil.copy(cfg_path, backup)
        final_cfg = make_candidate_config(base_cfg, args.simulator, args.param, best_value)
        with open(cfg_path, "w") as f:
            yaml.safe_dump(final_cfg, f, sort_keys=False, allow_unicode=True)
        print(f"Wrote {args.param}={best_value} to {cfg_path} (backup: {backup})")
    else:
        print(
            "config.yaml was not modified. Re-run with --apply to write "
            f"{args.param}={best_value}.")

    report_path = Path(args.report) if args.report else work_dir / "calibration_report.json"
    report = {
        "schema_version": 1,
        "simulator": args.simulator,
        "parameter": args.param,
        "search_method": "experimental_optuna_tpe" if args.optuna else "grid_coordinate_search",
        "metrics_requested": requested_metrics,
        "metric_weights": weights,
        "real_metric_vector": real_vector,
        "candidate_metric_vectors": candidate_metric_reports,
        "best": {"value": best_value, "error": best_err},
        "timestamp_precision_status": ts_status,
        "stimulus_reference": str(Path(args.stimulus_reference).resolve())
            if args.stimulus_reference else None,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Calibration report: {report_path}")


if __name__ == "__main__":
    main()
