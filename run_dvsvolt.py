#!/usr/bin/env python3
"""
CAROECT-D DVS-Voltmeter driver with a high-precision input path.

Why this file exists
--------------------
The upstream command-line program decodes grayscale images as uint8. Library
mode instead loads linear uint16 TIFFs, divides by 257 into the model's expected
0..255 DN scale without rounding, and calls the published Brownian-motion-with-
drift simulator. Its physics and k1..k6 implementation remain unmodified.

Modes
-----
lib (default) preserves fractional precision, seeds NumPy/Torch, and discovers
the research code's constructor/method signatures with fail-loud diagnostics.
cli is a compatibility fallback through the original main.py and 8-bit PNG;
it intentionally loses sub-8-bit precision and cannot guarantee bitwise
reproducibility when the upstream CLI exposes no seed.

Performance
-----------
The upstream simulator is CPU-oriented. Runtime grows roughly with pixel count,
so independent clips should be processed in parallel rather than changing the
published model.

Data flow
---------
Linear luminance TIFF -> float 0..255 -> simulator -> inferred t/x/y/p columns
-> unified events.h5 plus params.json and simulator git provenance.

Usage:
  conda run -n dvsvolt python run_dvsvolt.py --input processed --output events
  conda run -n dvsvolt python run_dvsvolt.py --input processed --output events --limit 120
  conda run -n dvsvolt python run_dvsvolt.py --input processed --output events --mode cli
"""
from __future__ import annotations
# Required by the Python 3.9 DVS-Voltmeter environment: postponed annotation
# evaluation keeps PEP 604 hints compatible with the older runtime.

import sys
import json
import time
import inspect
import argparse
import importlib
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
#  NODE 0 · CONFIG and intentionally duplicated standalone helpers
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
#  NODE 1 · FRAME LIST + TIMESTAMPS (µs)
# ══════════════════════════════════════════════════════════════════════════
#  DVS-Voltmeter uses microseconds. Derive timestamps from fps_original,
#  never from the export timeline.
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
#  Unit contract: k1..k6 were calibrated on 0..255 DN. Absolute L appears in
#  both drift and noise equations, so passing 0..65535 would invalidate the
#  model scale. Division by 257 preserves fractional uint16 information.
# ══════════════════════════════════════════════════════════════════════════

def load_frame_f32(path) -> np.ndarray:
    img = tifffile.imread(str(path))
    if img.ndim != 2:
        raise ValueError(f"{Path(path).name}: expected grayscale HxW output from preprocess.py, "
                         f"gặp shape {img.shape}")
    return img.astype(np.float32) / 257.0


# ══════════════════════════════════════════════════════════════════════════
#  NODE 3 · IMPORT SIMULATOR and discover the research-code API
# ══════════════════════════════════════════════════════════════════════════
#  Import src.simulator, prefer EventSim, construct its EasyDict-compatible
#  config, and try known constructor signatures. Discovery plus fail-loud
#  diagnostics is safer than silently assuming one unstable API.
#    conda run -n dvsvolt python - <<'PY'
#    import sys; sys.path.insert(0, "<repo>")
#    import src.simulator as s, inspect
#    print([n for n,o in vars(s).items() if inspect.isclass(o)])
#    print(inspect.signature([o for n,o in vars(s).items() if inspect.isclass(o)][0].__init__))
#    PY
# ══════════════════════════════════════════════════════════════════════════

def import_sim_class(repo: Path):
    sys.path.insert(0, str(repo))
    mod = None
    for name in ("src.simulator", "simulator", "src.event_sim"):
        try:
            mod = importlib.import_module(name)
            break
        except ImportError:
            continue
    if mod is None:
        raise RuntimeError(f"Could not import the simulator from {repo}. "
                           "(./setup_sim.sh). If it is installed, inspect `ls " + str(repo) + "/src`")
    for cn in ("EventSim", "Simulator", "EventSimulator", "DVSVoltmeter"):
        if hasattr(mod, cn):
            return mod, getattr(mod, cn)
    own = [o for n, o in vars(mod).items()
           if inspect.isclass(o) and o.__module__ == mod.__name__]
    if len(own) == 1:
        return mod, own[0]
    raise RuntimeError(f"Multiple candidate classes in {mod.__name__}: {[c.__name__ for c in own]} "
                       "— review VERIFY-2 in this file")


def _describe_signature(cls):
    """inspect.signature() can raise on some callables; a diagnostic helper must
    never be the thing that breaks a working path."""
    try:
        return f"{cls.__name__}{inspect.signature(cls.__init__)}"
    except (ValueError, TypeError):
        return f"{cls.__name__}(<signature not introspectable>)"


def construct_sim(cls, K, camera_type, width, height):
    """
    Constructs DVS-Voltmeter's EventSim.

    ROOT CAUSE OF THE `AttributeError: 'list' object has no attribute 'SENSOR'`
    CRASH (fixed here, kept written down so it is not reintroduced a third time):

    EventSim.__init__(cfg, output_folder=None, video_name=None) does an
    os.path.join(output_folder, ...) internally. Called as cls(cfg) — i.e.
    output_folder left at its None default — that join raises
        TypeError: join() argument must be str, not 'NoneType'
    The old code wrapped the call in `except TypeError` intending to mean "this
    signature does not match", so that internal TypeError was misread as a
    signature mismatch. It silently discarded the CORRECT call and fell through
    to the next trial, which passed the bare list [k1..k6] as `cfg`. A list binds
    to __init__(self, cfg) perfectly well, so the failure then happened deep
    inside the third-party repo at `cfg.SENSOR.K[0]` — an AttributeError, which
    the loop did not catch at all, producing a traceback that pointed at someone
    else's code and named the wrong cause.

    The fix is not a smarter probe, it is passing a REAL directory. There is one
    correct call and it is made directly, so any failure now surfaces where it
    actually happened instead of being reinterpreted.
    """
    from easydict import EasyDict
    import tempfile

    cfg = EasyDict()
    cfg.SENSOR = EasyDict()
    cfg.DIR = EasyDict()

    cfg.SENSOR.CAMERA_TYPE = camera_type
    cfg.SENSOR.K = list(map(float, K))

    cfg.DIR.IN_PATH = ""
    cfg.DIR.OUT_PATH = ""

    cfg.Width = width
    cfg.Height = height

    # A real, existing directory — this is the whole fix. EventSim joins paths
    # against it during __init__; None or "" makes that join throw.
    scratch_dir = tempfile.mkdtemp(prefix="dvsvolt_scratch_")

    try:
        sim = cls(cfg, output_folder=scratch_dir, video_name="events")
    except TypeError as e:
        # Reaching here means the repo's constructor genuinely does not take this
        # shape (a different fork/version). Report it with the real signature
        # rather than silently trying something else and corrupting the run.
        raise RuntimeError(
            f"{cls.__name__} rejected (cfg, output_folder=, video_name=): {e}\n"
            f"  Actual signature: {_describe_signature(cls)}\n"
            "  Inspect with: python run_dvsvolt.py --print-sim-api --config <cfg>")
    except AttributeError as e:
        # cfg is missing a key this fork reads. Name it — this is the error that
        # used to arrive disguised as a constructor-signature problem.
        raise RuntimeError(
            f"{cls.__name__} requested a config key not provided here: {e}\n"
            f"  Available: SENSOR.K, SENSOR.CAMERA_TYPE, DIR.IN_PATH, DIR.OUT_PATH, "
            f"Width, Height\n"
            "  Add the documented missing key to construct_sim().")

    print(f"  [lib] {cls.__name__}(cfg)  [scratch: {scratch_dir}]")
    return sim


# ══════════════════════════════════════════════════════════════════════════
#  NODE 4 · Discover one simulation method signature, then stream
# ══════════════════════════════════════════════════════════════════════════
#  The model naturally integrates between frame pairs, but some versions expose
#  a stateful (frame, time) method. Probe both on the first pair, lock the
#  compatible method, and stream. Seed NumPy/Torch before construction.
# ══════════════════════════════════════════════════════════════════════════

def _resolve_caller(sim, f0, f1, t0, t1):
    """
    Probes which event-generating shape this build of the repo exposes:
    pair-style .m(f_prev, f_cur, t_prev, t_cur) or stream-style .m(frame, t).

    Only argument BINDING is probed with a TypeError guard — never the body. The
    distinction matters and is the same one that caused the construct_sim crash:
    a TypeError raised *inside* a correctly-bound method is a real bug, not a
    hint to try the next shape. Binding is checked up front with
    signature.bind(); everything after that is reported, not swallowed.

    The stream probe is destructive (it feeds f0 to prime internal state before
    f1), which is why it only runs when the pair shape did not bind at all.
    """
    failures = []
    for m in ("generate_events", "simulate", "__call__", "run"):
        fn = getattr(sim, m, None)
        if fn is None or not callable(fn):
            continue

        def binds(*a):
            try:
                inspect.signature(fn).bind(*a)
                return True
            except TypeError:
                return False
            except (ValueError, AttributeError):
                return True  # not introspectable — let the actual call decide

        if binds(f0, f1, int(t0), int(t1)):
            ev = fn(f0, f1, int(t0), int(t1))
            print(f"  [lib] method: .{m}(f_prev, f_cur, t_prev, t_cur)")
            return ("pair", fn, ev)

        if binds(f0, int(t0)):
            fn(f0, int(t0))                      # Load the baseline frame.
            ev = fn(f1, int(t1))
            print(f"  [lib] method: .{m}(frame, t)  [stream/stateful]")
            return ("stream", fn, ev)

        failures.append(f"    - .{m}{_describe_method_sig(fn)}: matched neither call form")

    detail = "\n".join(failures) if failures else "    (no candidate method found)"
    raise RuntimeError(
        "No event-generation method matched. Tried:\n" + detail +
        "\n  Inspect the actual API: python run_dvsvolt.py --print-sim-api --config <cfg>")


def _describe_method_sig(fn):
    try:
        return str(inspect.signature(fn))
    except (ValueError, TypeError):
        return "(<not introspectable>)"


def run_lib(repo: Path, paths, t_us, K, camera_type, width, height, seed: int):
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

    _, cls = import_sim_class(repo)
    sim = construct_sim(cls, K, camera_type, width, height)

    chunks, style, fn = [], None, None
    prev = load_frame_f32(paths[0])
    t0w = time.time()
    for i in range(1, len(paths)):
        cur = load_frame_f32(paths[i])
        if fn is None:
            style, fn, ev = _resolve_caller(sim, prev, cur, t_us[i-1], t_us[i])
        elif style == "pair":
            ev = fn(prev, cur, int(t_us[i-1]), int(t_us[i]))
        else:
            ev = fn(cur, int(t_us[i]))
        if ev is not None and len(ev) > 0:
            chunks.append(np.asarray(ev))
        prev = cur
        if i % 200 == 0 or i == len(paths) - 1:
            r = i / (time.time() - t0w)
            eta = (len(paths) - 1 - i) / r if r > 0 else 0
            print(f"  [lib] {i:>6}/{len(paths)-1} pairs  {r:.1f} pairs/s  ETA {eta/60:.1f} min  "
                  f"events: {sum(len(c) for c in chunks):,}")
    if not chunks:
        raise RuntimeError("DVS-Voltmeter produced no events; check input and k parameters.")
    return np.concatenate(chunks, axis=0)


# ══════════════════════════════════════════════════════════════════════════
#  C1–C3 · CLI FALLBACK through the unmodified upstream main.py
# ══════════════════════════════════════════════════════════════════════════
#  C1: export 8-bit PNG plus info.txt ("<abs_path> <t_us>") in the repository layout
#      input_dir/<session>/ expected by main.py.
#  C2: invoke main.py with source-verified flags:
#      --input_dir --output_dir --camera_type --model_para k1..k6
#  C3: concatenate all emitted text events. The upstream CLI exposes no seed,
#      so CLI mode cannot promise bitwise reproducibility.
# ══════════════════════════════════════════════════════════════════════════

def run_cli(repo: Path, paths, t_us, K, camera_type, workdir: Path, env_name: str):
    import cv2
    session = "session"
    in_dir = workdir / "in" / session
    out_dir = workdir / "out"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [cli] C1: {len(paths)} frames → PNG 8-bit + info.txt (mất sub-8bit precision)")
    lines = []
    for i, (p, t) in enumerate(zip(paths, t_us)):
        u8 = np.clip(np.round(tifffile.imread(str(p)).astype(np.float32) / 257.0),
                     0, 255).astype(np.uint8)
        fp = in_dir / f"{i:06d}.png"
        cv2.imwrite(str(fp), u8)
        lines.append(f"{fp.resolve()} {int(t)}")
    (in_dir / "info.txt").write_text("\n".join(lines) + "\n")

    conda = shutil.which("conda") or "conda"
    cmd = [conda, "run", "-n", env_name, "python", "main.py",
           "--input_dir", str((workdir / "in").resolve()),
           "--output_dir", str(out_dir.resolve()),
           "--camera_type", camera_type,
           "--model_para"] + [str(float(k)) for k in K]
    print("  [cli] C2: " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(repo))

    txts = sorted(out_dir.rglob("*.txt"))
    if not txts:
        raise RuntimeError(
            "main.py emitted no text files. Inspect its expected input layout; "
            f"the session directory is {workdir/'in'/session}.")
    arrs = [np.loadtxt(str(t)) for t in txts]
    arrs = [a.reshape(-1, 4) for a in arrs if a.size]
    print(f"  [cli] C3: gom {len(txts)} file .txt → {sum(len(a) for a in arrs):,} events")
    return np.concatenate(arrs, axis=0)


# ══════════════════════════════════════════════════════════════════════════
#  NODE 5 · NORMALIZE — intentionally duplicated from the standalone v2e runner
# ══════════════════════════════════════════════════════════════════════════

def normalize_events(arr, width: int, height: int, dur_us_hint: float):
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 4:
        raise ValueError(f"Expected an Nx4 array, got {a.shape}")
    cols = set(range(4))

    p_col = next((c for c in cols
                  if set(np.unique(a[:, c]).astype(np.int64).tolist()) <= {-1, 0, 1}), None)
    if p_col is None:
        raise ValueError("Could not identify a {-1,0,1} polarity column")
    cols.discard(p_col)

    t_col, best = None, -1.0
    for c in cols:
        d = np.diff(a[:, c])
        if d.size == 0 or d.min() >= 0:
            r = a[:, c].max() - a[:, c].min()
            if r > best:
                best, t_col = r, c
    if t_col is None:
        t_col = max(cols, key=lambda c: a[:, c].max())
    cols.discard(t_col)

    c1, c2 = sorted(cols)
    if a[:, c1].max() >= height > a[:, c2].max():
        x_col, y_col = c1, c2
    elif a[:, c2].max() >= height > a[:, c1].max():
        x_col, y_col = c2, c1
    else:
        x_col, y_col = c1, c2

    t = a[:, t_col].copy()
    unit = "µs"
    if dur_us_hint and (t.max() - t.min()) < dur_us_hint / 1000.0:
        t *= 1e6
        unit = "s→µs"

    p = np.where(a[:, p_col] > 0, 1, 0).astype(np.uint8)
    order = np.argsort(t, kind="stable")

    print(f"  [norm] inferred columns: t=c{t_col}({unit})  x=c{x_col}  y=c{y_col}  p=c{p_col}"
          f"   |  {len(t):,} events, {(t.max()-t.min())/1e6:.2f}s")
    x = np.clip(np.rint(a[order, x_col]), 0, width - 1).astype(np.uint16)
    y = np.clip(np.rint(a[order, y_col]), 0, height - 1).astype(np.uint16)
    return dict(x=x,
                y=y,
                t=t[order].astype(np.uint64),
                p=p[order])


# ══════════════════════════════════════════════════════════════════════════
#  NODE 6 · WRITE UNIFIED H5 (schema = read_evt3.py: x,y,t,p + attrs)
# ══════════════════════════════════════════════════════════════════════════

def write_h5(out_dir: Path, ev: dict, attrs: dict):
    if h5py is None:
        raise RuntimeError("Install h5py in the dvsvolt environment.")
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
    ap = argparse.ArgumentParser(description="CAROECT-D: linear TIFFs → DVS-Voltmeter events (unified h5)")
    ap.add_argument("--input",  required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mode", choices=["lib", "cli"], default="lib",
                    help="lib = 16-bit + seed (default) | cli = original main.py, 8-bit, no seed")
    ap.add_argument("--limit", type=int, default=None, help="Use only the first N frames")
    ap.add_argument("--print-sim-api", action="store_true",
                    help="Print discovered simulator classes/signatures and exit")
    args = ap.parse_args()

    if args.print_sim_api:
        # Diagnostics do not require input/output data.
        cfg0 = load_config(args.config)
        repo0 = Path(cfg0["paths"].get("dvsvolt_repo",
                                       "~/caroect_sim/DVS-Voltmeter")).expanduser()
        if not repo0.exists():
            raise FileNotFoundError(f"DVS-Voltmeter repository not found at {repo0}")
        mod, cls = import_sim_class(repo0)
        print(f"module      : {mod.__name__}  ({getattr(mod, '__file__', '?')})")
        print(f"class       : {cls.__name__}")
        print(f"__init__    : {_describe_signature(cls)}")
        print("candidate event-generation methods:")
        for m in ("generate_events", "simulate", "__call__", "run"):
            fn = getattr(cls, m, None)
            if callable(fn):
                try:
                    print(f"  .{m}{inspect.signature(fn)}")
                except (ValueError, TypeError):
                    print(f"  .{m}(<not introspectable>)")
        print("\nother classes in the module:")
        print("  " + ", ".join(n for n, o in vars(mod).items()
                               if inspect.isclass(o) and o.__module__ == mod.__name__))
        return

    cfg = load_config(args.config)
    cam, dv, sim = cfg["camera"], cfg["dvs_voltmeter"], cfg.get("simulator", {})
    seed = int(sim.get("seed", 42))
    fps = float(cam["fps_original"])
    W, H = int(cam["width"]), int(cam["height"])
    K = dv["k"]
    camera_type = dv.get("camera_type", "DVS346")
    if len(K) != 6:
        raise ValueError(f"dvs_voltmeter.k requires six values; got {len(K)}")
    repo = Path(cfg["paths"].get("dvsvolt_repo", "~/caroect_sim/DVS-Voltmeter")).expanduser()
    if not repo.exists():
        raise FileNotFoundError(
            f"DVS-Voltmeter repository not found at {repo}; run setup_sim.sh")

    in_dir, out_dir = Path(args.input), Path(args.output)
    paths, t_us = list_frames(in_dir, fps, args.limit)
    dur_us = float(len(paths)) * 1e6 / fps

    print(f"\n{'━'*60}\n  DVS-Voltmeter driver  |  mode={args.mode}  |  {camera_type} "
          f"k={K}\n  {len(paths)} frames ({dur_us/1e6:.2f}s @ {fps}fps)   "
          f"{in_dir} → {out_dir}\n{'━'*60}")

    if args.mode == "lib":
        raw = run_lib(repo, paths, t_us, K, camera_type, W, H, seed)
    else:
        env_name = sim.get("envs", {}).get("dvsvolt", "dvsvolt")
        raw = run_cli(repo, paths, t_us, K, camera_type, out_dir / "_work_cli", env_name)

    ev = normalize_events(raw, W, H, dur_us)
    attrs = dict(simulator="dvs_voltmeter", mode=args.mode, git_commit=git_hash(repo),
                 seed=(seed if args.mode == "lib" else "N/A (upstream CLI has no seed)"),
                 fps=fps, width=W, height=H, camera_type=camera_type,
                 source=str(in_dir), params=dict(k=list(map(float, K))))
    write_h5(out_dir, ev, attrs)
    print("\nDone. Process independent clips in parallel to offset CPU runtime.\n")


if __name__ == "__main__":
    main()
