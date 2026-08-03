#!/usr/bin/env python3
"""
hot_pixel_heatmap.py — Where are the idle clip's events actually coming from?

Disambiguates two very different explanations for a high "hottest-1%-pixels"
share on an idle (no-motion) clip:
  A) Clustered along shapes (power lines, sign edges, tree branches) -> real
     static high-contrast edges picking up camera/wind vibration. Expected,
     matches the physical reasoning already established for this project
     (background jitter should be present in training data, not suppressed).
  B) Scattered randomly with no shape -> true sensor hot/defective pixels,
     or bias set too sensitive globally. Worth addressing (raise threshold,
     or flag pixels for masking).

Usage:
    python hot_pixel_heatmap.py no_motion.h5 --output heatmap.png
"""

import argparse
import h5py
import numpy as np
import cv2


def load_events_h5(path):
    with h5py.File(str(path), "r") as hf:
        return {k: hf[k][:] for k in ("x", "y", "t", "p")}


def main():
    ap = argparse.ArgumentParser(description="Spatial heatmap of event counts per pixel")
    ap.add_argument("input", help="events.h5 path")
    ap.add_argument("--output", default="heatmap.png")
    ap.add_argument("--sensor-width", type=int, default=1280)
    ap.add_argument("--sensor-height", type=int, default=720)
    ap.add_argument("--top-percent", type=float, default=1.0,
                    help="Highlight the hottest N%% of pixels separately (default 1.0)")
    args = ap.parse_args()

    ev = load_events_h5(args.input)
    x, y = ev["x"], ev["y"]
    W, H = args.sensor_width, args.sensor_height

    counts = np.zeros((H, W), dtype=np.int64)
    np.add.at(counts, (y.astype(np.int64), x.astype(np.int64)), 1)

    n_events = len(x)
    n_pixels = W * H
    flat = counts.ravel()
    top_n = max(1, int(n_pixels * args.top_percent / 100.0))
    threshold = np.partition(flat, -top_n)[-top_n]

    print(f"Total events: {n_events:,}")
    print(f"Top {args.top_percent}% pixel threshold: >={threshold} events/pixel")
    print(f"Max single-pixel count: {flat.max():,}  "
          f"({100*flat.max()/n_events:.2f}% of all events in ONE pixel)")

    # Heatmap: log-scale so both the diffuse background and the hot spots
    # are visible in the same image (linear scale would wash out one or
    # the other given how skewed pixel-count distributions usually are).
    log_counts = np.log1p(counts).astype(np.float32)
    norm = cv2.normalize(log_counts, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)

    # Outline the top-N% hottest pixels in cyan so they're unambiguous,
    # even where the colormap alone might be ambiguous.
    hot_mask = (counts >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(hot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(heat, contours, -1, (255, 255, 0), 1)

    cv2.imwrite(args.output, heat)
    print(f"Saved: {args.output}")
    print(f"\nHow to read it:")
    print(f"  - Bright spots following LINES/EDGES (wires, sign borders, branches)")
    print(f"    = real static contrast + vibration. Expected, not a problem.")
    print(f"  - Bright spots as ISOLATED DOTS with no shape around them")
    print(f"    = likely true hot/defective pixels or oversensitive bias.")
    print(f"  - A diffuse, roughly-even glow everywhere = normal shot noise,")
    print(f"    fine as-is.")


if __name__ == "__main__":
    main()
