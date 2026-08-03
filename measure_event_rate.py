#!/usr/bin/env python3
"""
measure_event_rate.py — Compare event-rate statistics across recordings.

Built for the no-motion vs with-motion baseline measurement: convert both
.cevt files first (cevt_to_events.py), then run this on the resulting .h5
files to get a clean side-by-side comparison instead of eyeballing CD Frames.

Usage:
    python cevt_to_events.py no_motion.cevt   --output-h5 no_motion.h5
    python cevt_to_events.py with_motion.cevt --output-h5 with_motion.h5

    python measure_event_rate.py no_motion.h5:idle with_motion.h5:motion
    python measure_event_rate.py no_motion.h5 with_motion.h5   # auto-labeled
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def load_events_h5(path):
    """Same schema/reader used across the project (cevt_to_events.py,
    run_v2e.py, run_dvsvolt.py, read_evt3.py)."""
    with h5py.File(str(path), "r") as hf:
        return {k: hf[k][:] for k in ("x", "y", "t", "p")}


def analyze(path: str, label: str, sensor_w: int, sensor_h: int, window_s: float):
    ev = load_events_h5(path)
    x, y, t, p = ev["x"], ev["y"], ev["t"], ev["p"]
    n = len(t)

    if n == 0:
        print(f"\n[{label}] {path}: 0 events — nothing to analyze.")
        return None

    duration_s = (t[-1] - t[0]) / 1e6 if n > 1 else 0.0
    if duration_s <= 0:
        print(f"\n[{label}] {path}: duration is 0 — check --fps used in "
              "cevt_to_events.py matched the real capture rate.")
        return None

    rate_hz = n / duration_s
    n_on = int(np.sum(p == 1))
    n_off = n - n_on

    # Per-pixel background rate: mean events per pixel over the whole clip.
    # A handful of pixels firing constantly (hot pixels / bias too sensitive)
    # will show up as a heavy tail here, not visible from total rate alone.
    n_pixels = sensor_w * sensor_h
    counts = np.zeros(n_pixels, dtype=np.int64)
    idx = y.astype(np.int64) * sensor_w + x.astype(np.int64)
    valid = (idx >= 0) & (idx < n_pixels)
    np.add.at(counts, idx[valid], 1)
    active_px = int(np.sum(counts > 0))
    top1pct = int(n_pixels * 0.01)
    hot_share = counts[np.argsort(counts)[::-1][:top1pct]].sum() / n if n else 0.0

    # Rate stability over time: split into fixed windows, report min/max/std
    # of per-window rate. Bursty/unstable ERC-limited data shows up here.
    n_windows = max(1, int(duration_s // window_s))
    edges = np.linspace(t[0], t[-1], n_windows + 1)
    win_counts, _ = np.histogram(t, bins=edges)
    win_rate_hz = win_counts / window_s

    print(f"\n{'=' * 60}")
    print(f"[{label}] {path}")
    print(f"{'=' * 60}")
    print(f"  total events        : {n:,}")
    print(f"  duration             : {duration_s:.2f} s")
    print(f"  mean rate            : {rate_hz:,.0f} ev/s  ({rate_hz/1e6:.3f} Mev/s)")
    print(f"  ON / OFF split       : {n_on:,} / {n_off:,}  "
          f"({100*n_on/n:.1f}% / {100*n_off/n:.1f}%)")
    print(f"  active pixels        : {active_px:,} / {n_pixels:,} "
          f"({100*active_px/n_pixels:.1f}%)")
    print(f"  hottest 1% pixels    : {100*hot_share:.1f}% of all events "
          f"(high = possible hot pixels / bias too sensitive)")
    print(f"  rate over {window_s:.1f}s windows : "
          f"min={win_rate_hz.min():,.0f}  max={win_rate_hz.max():,.0f}  "
          f"std={win_rate_hz.std():,.0f} ev/s")

    return {
        "label": label, "path": path, "n": n, "duration_s": duration_s,
        "rate_hz": rate_hz, "n_on": n_on, "n_off": n_off,
        "active_px": active_px, "n_pixels": n_pixels, "hot_share": hot_share,
    }


def main():
    ap = argparse.ArgumentParser(description="Compare event rate across recordings")
    ap.add_argument("files", nargs="+",
                    help="events.h5 paths, optionally 'path:label' "
                         "(e.g. no_motion.h5:idle with_motion.h5:motion)")
    ap.add_argument("--sensor-width", type=int, default=1280)
    ap.add_argument("--sensor-height", type=int, default=720)
    ap.add_argument("--window", type=float, default=1.0,
                    help="Window size in seconds for rate-stability check (default 1.0)")
    args = ap.parse_args()

    results = []
    for spec in args.files:
        if ":" in spec:
            path, label = spec.rsplit(":", 1)
        else:
            path, label = spec, Path(spec).stem
        if not Path(path).exists():
            print(f"File not found: {path}", file=sys.stderr)
            sys.exit(1)
        r = analyze(path, label, args.sensor_width, args.sensor_height, args.window)
        if r:
            results.append(r)

    if len(results) < 2:
        return

    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print(f"{'=' * 60}")
    base = results[0]
    for r in results[1:]:
        ratio = r["rate_hz"] / base["rate_hz"] if base["rate_hz"] > 0 else float("inf")
        print(f"  {r['label']} vs {base['label']}: "
              f"{r['rate_hz']:,.0f} vs {base['rate_hz']:,.0f} ev/s "
              f"-> {ratio:.1f}x")

    print(f"\nWhat to look for:")
    print(f"  - 'motion' clip should be CLEARLY higher rate than 'idle' clip.")
    print(f"    If they're close, bias may be too insensitive to catch real")
    print(f"    motion, OR too sensitive so idle is already saturated with noise.")
    print(f"  - 'idle' clip's hottest-1%-pixels share should be LOW (a few %).")
    print(f"    High share on the idle clip = hot pixels or bias too sensitive")
    print(f"    even with nothing moving.")
    print(f"  - rate std/mean ratio on either clip: very high (>0.5-1x mean)")
    print(f"    suggests ERC may be capping bursts (check ErcRateLimit vs")
    print(f"    the peak windows above).")


if __name__ == "__main__":
    main()
