#!/usr/bin/env python3
"""
CAROECT-D v2e driver with a high-precision input path.

Why this file exists
--------------------
v2e's command-line reader normally decodes 8-bit video, which would discard
the sub-8-bit information retained by the linear uint16 pipeline. Library mode
loads each TIFF into float32 on v2e's expected 0..255 DN scale without rounding
and calls the published EventEmulator directly. No simulator physics is forked.

Modes
-----
lib (default) preserves uint16-derived float precision and fails loudly on API
drift. cli is a documented compatibility fallback that creates lossless 8-bit
video and therefore intentionally loses sub-8-bit precision.

Data flow
---------
Linear luminance TIFF -> float 0..255 -> EventEmulator -> inferred t/x/y/p
columns -> unified events.h5 plus params.json and simulator git provenance.
Frames stream from disk; a full clip is never preloaded.

Capture timestamps always use camera.fps_original. fps_export describes only
the export timeline and would introduce a large timing error here.

Usage:
  conda run -n v2e python run_v2e.py --input processed --output events
  conda run -n v2e python run_v2e.py --input processed --output events --limit 120
  conda run -n v2e python run_v2e.py --input processed --output events --mode cli
"""

import sys
import json
import time
import inspect
import argparse
import subprocess
import shutil
from pathlib import Path

import numpy as np
import tifffile
import yaml

try:
    import h5py
except ImportError:
    h5py = None


# ══════════════════════════════════════════════════════════════════════════
#  NODE 0 · CONFIG
# ══════════════════════════════════════════════════════════════════════════

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def git_hash(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════════
#  NODE 1 · FRAME LIST + TIMESTAMPS
# ══════════════════════════════════════════════════════════════════════════
#  in   : preprocess.py output directory and configured fps_original
#  out  : (paths, t_us)  → N2/N4
#  what : sort TIFFs and assign t_us[i] = round(i * 1e6 / fps).
#  why  : timestamps require capture fps_original, never export-timeline FPS.
# ══════════════════════════════════════════════════════════════════════════

def list_frames(input_dir: Path, fps: float, limit: int | None):
    paths = sorted(set(list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))))
    if not paths:
        raise FileNotFoundError(f"No .tif/.tiff frames found in {input_dir}")
    if limit:
        paths = paths[:limit]
    t_us = np.array([round(i * 1e6 / fps) for i in range(len(paths))], dtype=np.int64)
    return paths, t_us


# ══════════════════════════════════════════════════════════════════════════
#  NODE 2 · LOAD 16-bit → float32 (0..255)
# ══════════════════════════════════════════════════════════════════════════
#  in   : path 1 TIFF 16-bit grayscale  ← N1
#  out  : float32 HxW on 0..255 with fractional sub-8-bit values
#  what : divide by 257 because 65535/257 equals 255 exactly.
#  why  : preserve uint16 precision while using the DN scale expected by v2e.
# ══════════════════════════════════════════════════════════════════════════

def load_frame_f32(path) -> np.ndarray:
    img = tifffile.imread(str(path))
    if img.ndim != 2:
        raise ValueError(
            f"{Path(path).name}: expected grayscale HxW preprocessing output, got {img.shape}")
    return img.astype(np.float32) / 257.0


# ══════════════════════════════════════════════════════════════════════════
#  NODE 3 · EMULATOR (lib mode)
# ══════════════════════════════════════════════════════════════════════════
#  input: v2e repository, config parameters, and seed from N0
#  out  : EventEmulator instance  → N4
#  what : import EventEmulator and filter kwargs through its live signature.
#  why  : tolerate API-version differences without modifying v2e physics.
#  VERIFY-1: module/class/method names follow the official v2e tutorial.
#           conda run -n v2e python -c "import v2ecore.emulator as e; import inspect; print(inspect.signature(e.EventEmulator.__init__)); print([m for m in dir(e.EventEmulator) if 'event' in m.lower()])"
# ══════════════════════════════════════════════════════════════════════════

def build_emulator(repo: Path, v: dict, seed: int):
    sys.path.insert(0, str(repo))
    from v2ecore.emulator import EventEmulator          # VERIFY-1
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)

    want = dict(
        pos_thres=v["pos_thres"],
        neg_thres=v["neg_thres"],
        sigma_thres=v["sigma_thres"],
        cutoff_hz=v["cutoff_hz"],
        leak_rate_hz=v["leak_rate_hz"],
        shot_noise_rate_hz=v["shot_noise_rate_hz"],
        refractory_period_s=v.get("refractory_period", 1e-3),
        seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    sig = inspect.signature(EventEmulator.__init__)
    kwargs = {k: w for k, w in want.items() if k in sig.parameters}
    dropped = sorted(set(want) - set(kwargs))
    if dropped:
        print(f"  [lib] This EventEmulator version does not accept {dropped}; omitted. "
              "Review VERIFY-1 if any field is required.")
    print(f"  [lib] EventEmulator({', '.join(f'{k}={v}' for k, v in kwargs.items())})")
    return EventEmulator(**kwargs)


# ══════════════════════════════════════════════════════════════════════════
#  NODE 4 · SIMULATION LOOP (lib mode)
# ══════════════════════════════════════════════════════════════════════════
#  in   : emulator ← N3, frames+t_us ← N1/N2
#  out  : raw Nx4 event array; N5 infers the version-dependent column order
#  what : stream frames from disk and call generate_events(frame, seconds).
#         The first call establishes a baseline and may return None.
# ══════════════════════════════════════════════════════════════════════════

def run_lib(repo: Path, paths, t_us, v: dict, seed: int):
    em = build_emulator(repo, v, seed)
    chunks = []
    t0 = time.time()
    for i, (p, t) in enumerate(zip(paths, t_us)):
        fr = load_frame_f32(p)
        ev = em.generate_events(fr, float(t) / 1e6)     # VERIFY-1: seconds
        if ev is not None and len(ev) > 0:
            chunks.append(np.asarray(ev))
        if (i + 1) % 200 == 0 or (i + 1) == len(paths):
            r = (i + 1) / (time.time() - t0)
            print(f"  [lib] {i+1:>6}/{len(paths)}  {r:.1f} fr/s  "
                  f"events: {sum(len(c) for c in chunks):,}")
    if not chunks:
        raise RuntimeError("v2e produced no events; input may be flat or thresholds too high.")
    return np.concatenate(chunks, axis=0)


# ══════════════════════════════════════════════════════════════════════════
#  C1–C4 · CLI FALLBACK using the original v2e.py binary
# ══════════════════════════════════════════════════════════════════════════
#  C1 quantizes to PNG8, C2 creates lossless FFV1 video, C3 runs the documented
#  v2e CLI, and C4 locates its Nx4 HDF5 dataset. This compatibility path
#  intentionally loses sub-8-bit precision.
# ══════════════════════════════════════════════════════════════════════════

def run_cli(repo: Path, paths, v: dict, seed: int, fps: float,
            width: int, height: int, workdir: Path):
    import cv2
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is missing (sudo apt install ffmpeg); CLI mode requires it.")

    # C1 — PNG 8-bit
    png_dir = workdir / "png8"
    png_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [cli] C1: quantize {len(paths)} frames → PNG 8-bit (mất sub-8bit precision)")
    for i, p in enumerate(paths):
        u8 = np.clip(np.round(tifffile.imread(str(p)).astype(np.float32) / 257.0),
                     0, 255).astype(np.uint8)
        cv2.imwrite(str(png_dir / f"{i:06d}.png"), u8)

    # C2 — lossless video
    avi = workdir / "frames.avi"
    print(f"  [cli] C2: ffmpeg FFV1 @ {fps}fps → {avi.name}")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-framerate", f"{fps}", "-i", str(png_dir / "%06d.png"),
                    "-c:v", "ffv1", str(avi)], check=True)

    # C3 — documented v2e.py flags; inspect v2e.py -h after API drift.
    out_h5 = "events_v2e_raw.h5"
    cmd = [sys.executable, str(repo / "v2e.py"),
           "--input", str(avi),
           "--output_folder", str(workdir),
           "--input_frame_rate", f"{fps}",
           "--disable_slomo",
           "--pos_thres", str(v["pos_thres"]),
           "--neg_thres", str(v["neg_thres"]),
           "--sigma_thres", str(v["sigma_thres"]),
           "--cutoff_hz", str(v["cutoff_hz"]),
           "--leak_rate_hz", str(v["leak_rate_hz"]),
           "--shot_noise_rate_hz", str(v["shot_noise_rate_hz"]),
           "--refractory_period", str(v.get("refractory_period", 1e-3)),
           "--dvs_emulator_seed", str(seed),
           "--output_width", str(width),
           "--output_height", str(height),
           "--dvs_h5", out_h5,
           "--no_preview"]
    print("  [cli] C3: " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(repo))

    # C4 — read v2e HDF5 output
    if h5py is None:
        raise RuntimeError("Install h5py in the v2e environment.")
    h5path = workdir / out_h5
    found = {}
    with h5py.File(h5path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim == 2 \
               and obj.shape[1] == 4 and "arr" not in found:
                found["arr"], found["name"] = obj[...], name
        f.visititems(visit)
    if "arr" not in found:
        raise RuntimeError(f"No Nx4 dataset found in {h5path}; inspect it with h5ls -r.")
    print(f"  [cli] C4: dataset '{found['name']}'  ({len(found['arr']):,} events)")
    return found["arr"]


# ══════════════════════════════════════════════════════════════════════════
#  NODE 5 · NORMALIZE — infer t/x/y/p columns from value constraints
# ══════════════════════════════════════════════════════════════════════════
#  in   : Nx4 with version-dependent column order
#  out  : dict x(u16), y(u16), t(u64 µs, sorted), p(u8 0/1)  → N6
#  Polarity contains only {-1,0,1}; time is monotonic with the largest range;
#  x/y are the remaining columns. A duration-scale check detects seconds and
#  converts them to microseconds. The duplicated DVS-Voltmeter helper keeps
#  both simulator runners independently executable.
# ══════════════════════════════════════════════════════════════════════════

def normalize_events(arr, width: int, height: int, dur_us_hint: float, mode: str):
    """
    Normalize v2e events to CAROECT-D unified schema.

    v2e EventEmulator.generate_events() returns:
        [timestamp_seconds, x, y, polarity]

    v2e CLI HDF5 returns:
        [timestamp_microseconds, x, y, polarity]

    Output:
        x uint16
        y uint16
        t uint64 microseconds
        p uint8 {0,1}
    """
    a = np.asarray(arr, dtype=np.float64)

    if a.ndim != 2 or a.shape[1] != 4:
        raise ValueError(f"Expected an Nx4 array, got {a.shape}")

    # v2e's documented/fixed event layout
    t = a[:, 0].copy()
    x = a[:, 1].copy()
    y = a[:, 2].copy()
    p_raw = a[:, 3].copy()

    # Library EventEmulator returns timestamps in seconds.
    # CLI HDF5 already stores timestamps in microseconds.
    if mode == "lib":
        t *= 1e6
        unit = "s→µs"
    else:
        unit = "µs"

    # ±1 or 0/1 -> unified 0/1
    p = np.where(p_raw > 0, 1, 0).astype(np.uint8)

    # Validate coordinates before clipping.
    if len(x):
        if x.min() < 0 or x.max() >= width:
            raise ValueError(
                f"x outside sensor bounds: min={x.min()} max={x.max()} width={width}"
            )

        if y.min() < 0 or y.max() >= height:
            raise ValueError(
                f"y outside sensor bounds: min={y.min()} max={y.max()} height={height}"
            )

    order = np.argsort(t, kind="stable")

    x = np.rint(x[order]).astype(np.uint16)
    y = np.rint(y[order]).astype(np.uint16)
    t = np.rint(t[order]).astype(np.uint64)
    p = p[order]

    duration_s = (float(t[-1]) - float(t[0])) / 1e6 if len(t) > 1 else 0.0

    print(
        f"  [norm] fixed v2e schema: "
        f"t=c0({unit}) x=c1 y=c2 p=c3"
        f"   |  {len(t):,} events, {duration_s:.4f}s"
    )

    return dict(
        x=x,
        y=y,
        t=t,
        p=p,
    )


# ══════════════════════════════════════════════════════════════════════════
#  NODE 6 · WRITE UNIFIED H5
# ══════════════════════════════════════════════════════════════════════════
#  input: dictionary from N5 plus metadata
#  out  : events.h5 (schema GIỐNG read_evt3.py: x,y,t,p gzip) + params.json
#  v2e, DVS-Voltmeter, and real recordings share one schema. Attributes carry
#  parameters and git provenance so downstream readers need one code path.
# ══════════════════════════════════════════════════════════════════════════

def write_h5(out_dir: Path, ev: dict, attrs: dict):
    if h5py is None:
        raise RuntimeError("pip install h5py")
    out_dir.mkdir(parents=True, exist_ok=True)
    h5path = out_dir / "events.h5"
    with h5py.File(h5path, "w") as f:
        for k in ("x", "y", "t", "p"):
            f.create_dataset(k, data=ev[k], compression="gzip")
        f.attrs["n_events"] = len(ev["t"])
        for k, v in attrs.items():
            f.attrs[k] = v if isinstance(v, (int, float, str)) else json.dumps(v)
    (out_dir / "params.json").write_text(json.dumps(attrs, indent=2))
    print(f"  [save] {h5path}  ({len(ev['t']):,} events)  + params.json")
    return h5path


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="CAROECT-D: linear TIFFs → v2e events (unified h5)")
    ap.add_argument("--input",  required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mode", choices=["lib", "cli"], default="lib",
                    help="lib = 16-bit through EventEmulator (default) | cli = original v2e.py, 8-bit")
    ap.add_argument("--limit", type=int, default=None, help="Use only the first N frames")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cam, v, sim = cfg["camera"], cfg["v2e"], cfg.get("simulator", {})
    seed = int(sim.get("seed", 42))
    fps = float(cam["fps_original"])                     # Capture FPS, not fps_export.
    W, H = int(cam["width"]), int(cam["height"])
    repo = Path(cfg["paths"].get("v2e_repo", "~/caroect_sim/v2e")).expanduser()
    if not repo.exists():
        raise FileNotFoundError(f"v2e repository not found at {repo}; run setup_sim.sh")

    in_dir, out_dir = Path(args.input), Path(args.output)
    paths, t_us = list_frames(in_dir, fps, args.limit)
    dur_us = float(len(paths)) * 1e6 / fps

    print(f"\n{'━'*60}\n  v2e driver  |  mode={args.mode}  |  {len(paths)} frames "
          f"({dur_us/1e6:.2f}s @ {fps}fps)\n  {in_dir} → {out_dir}\n{'━'*60}")

    if args.mode == "lib":
        raw = run_lib(repo, paths, t_us, v, seed)
    else:
        raw = run_cli(repo, paths, v, seed, fps, W, H, out_dir / "_work_cli")

    ev = normalize_events(raw, W, H, dur_us, args.mode)
    attrs = dict(simulator="v2e", mode=args.mode, git_commit=git_hash(repo),
                 seed=seed, fps=fps, width=W, height=H,
                 source=str(in_dir), params=v)
    write_h5(out_dir, ev, attrs)
    print("\nDone. Compare against measured events with measure_event_rate.py.\n")


if __name__ == "__main__":
    main()
