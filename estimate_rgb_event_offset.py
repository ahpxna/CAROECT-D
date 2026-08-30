#!/usr/bin/env python3
"""Estimate a measured RGB-to-event clock offset from a blinking target.

The RGB signal is ROI mean intensity at frame timestamps. The event signal is
the ROI event count in matching sensor-time bins. A derivative-envelope
cross-correlation provides a coarse offset; matched transition peaks refine it.

Sign convention:
    event_time_us = rgb_time_us + offset_us
"""

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


CONVENTION = "event_time_us = rgb_time_us + offset_us"


def _transition_peaks(times, signal, threshold_sigma=2.0):
    values = np.asarray(signal, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if len(values) < 3:
        return np.empty(0), np.empty(0)
    derivative = np.gradient(values, times)
    magnitude = np.abs(derivative)
    threshold = np.median(magnitude) + threshold_sigma * (
        1.4826 * np.median(np.abs(magnitude - np.median(magnitude))) + 1e-12
    )
    candidates = np.flatnonzero(magnitude >= threshold)
    if not len(candidates):
        candidates = np.argsort(magnitude)[-min(5, len(magnitude)):]
    # Keep the strongest local candidate within one median sample interval.
    spacing = max(float(np.median(np.diff(times))), 1.0)
    selected = []
    for index in candidates[np.argsort(magnitude[candidates])[::-1]]:
        if all(abs(times[index] - times[other]) > spacing for other in selected):
            selected.append(int(index))
    selected.sort()
    return times[selected], np.sign(derivative[selected])


def estimate_offset_from_signals(
    rgb_times_us,
    rgb_signal,
    event_times_us,
    event_signal,
):
    """Estimate offset and diagnostic details from two sampled 1-D signals."""
    rgb_times = np.asarray(rgb_times_us, dtype=np.float64)
    event_times = np.asarray(event_times_us, dtype=np.float64)
    rgb_values = np.asarray(rgb_signal, dtype=np.float64)
    event_values = np.asarray(event_signal, dtype=np.float64)
    if min(len(rgb_times), len(event_times)) < 4:
        raise ValueError("At least four samples are required from each sensor")
    dt = max(1.0, min(np.median(np.diff(rgb_times)), np.median(np.diff(event_times))))
    start = min(rgb_times[0], event_times[0])
    end = max(rgb_times[-1], event_times[-1])
    grid = np.arange(start, end + dt, dt)
    rgb_interp = np.interp(grid, rgb_times, rgb_values, left=rgb_values[0], right=rgb_values[-1])
    event_interp = np.interp(
        grid, event_times, event_values, left=event_values[0], right=event_values[-1]
    )
    rgb_edge = np.abs(np.gradient(rgb_interp))
    event_edge = np.abs(np.gradient(event_interp))
    rgb_edge -= rgb_edge.mean()
    event_edge -= event_edge.mean()
    correlation = np.correlate(event_edge, rgb_edge, mode="full")
    lags = np.arange(-len(rgb_edge) + 1, len(event_edge))
    coarse_offset = float(lags[int(np.argmax(correlation))] * dt)

    rgb_peaks, _ = _transition_peaks(rgb_times, rgb_values)
    event_peaks, _ = _transition_peaks(event_times, event_values)
    differences = []
    tolerance = max(3 * dt, 0.1 * (end - start))
    for rgb_peak in rgb_peaks:
        expected = rgb_peak + coarse_offset
        if len(event_peaks):
            event_peak = event_peaks[int(np.argmin(np.abs(event_peaks - expected)))]
            if abs(event_peak - expected) <= tolerance:
                differences.append(float(event_peak - rgb_peak))
    if differences:
        offset = float(np.median(differences))
        residual = float(np.sqrt(np.mean((np.asarray(differences) - offset) ** 2)))
    else:
        offset = coarse_offset
        residual = None
    peak = float(np.max(correlation))
    confidence = peak / (float(np.linalg.norm(rgb_edge) * np.linalg.norm(event_edge)) + 1e-12)
    return {
        "offset_us": offset,
        "coarse_offset_us": coarse_offset,
        "residual_rms_us": residual,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "matched_transitions": len(differences),
        "rgb_transition_times_us": rgb_peaks.tolist(),
        "event_transition_times_us": event_peaks.tolist(),
    }


def load_rgb_signal(frame_dir, roi, fps, frame_times_json=None):
    paths = sorted(
        set(Path(frame_dir).glob("*.tif")) | set(Path(frame_dir).glob("*.tiff"))
        | set(Path(frame_dir).glob("*.png")) | set(Path(frame_dir).glob("*.jpg"))
    )
    if not paths:
        raise FileNotFoundError(f"No RGB frames found in {frame_dir}")
    x, y, width, height = roi
    signal = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Could not read {path}")
        patch = image[y:y + height, x:x + width]
        if not patch.size:
            raise ValueError(f"ROI {roi} is outside {path.name} with shape {image.shape}")
        signal.append(float(patch.mean()))
    if frame_times_json:
        payload = json.loads(Path(frame_times_json).read_text())
        times = np.asarray(payload.get("frame_times_us", payload), dtype=np.float64)
    else:
        times = np.arange(len(paths), dtype=np.float64) * 1e6 / fps
    if len(times) != len(paths):
        raise ValueError("frame_times_us length does not match the number of RGB frames")
    return times, np.asarray(signal), [str(path) for path in paths]


def load_event_signal(event_h5, roi, bin_us):
    with h5py.File(event_h5, "r") as handle:
        x = handle["x"][:]
        y = handle["y"][:]
        timestamps = handle["t"][:].astype(np.float64)
    x0, y0, width, height = roi
    selected = timestamps[
        (x >= x0) & (x < x0 + width) & (y >= y0) & (y < y0 + height)
    ]
    if not len(selected):
        raise ValueError("The selected event ROI contains no events")
    edges = np.arange(timestamps.min(), timestamps.max() + bin_us, bin_us)
    counts, edges = np.histogram(selected, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts.astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-dir", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--rgb-roi", type=int, nargs=4, required=True, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--event-roi", type=int, nargs=4, required=True, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--frame-times-json")
    parser.add_argument("--event-bin-us", type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rgb_times, rgb_signal, rgb_files = load_rgb_signal(
        args.rgb_dir, args.rgb_roi, args.fps, args.frame_times_json
    )
    bin_us = args.event_bin_us or float(np.median(np.diff(rgb_times)))
    event_times, event_signal = load_event_signal(args.events, args.event_roi, bin_us)
    estimate = estimate_offset_from_signals(
        rgb_times, rgb_signal, event_times, event_signal
    )
    estimate.update({
        "schema_version": 1,
        "convention": CONVENTION,
        "method": "ROI transition derivative cross-correlation plus matched-peak refinement",
        "source_files": {
            "rgb_first": rgb_files[0],
            "rgb_last": rgb_files[-1],
            "events": str(Path(args.events).resolve()),
        },
        "timestamp_units": "microseconds",
        "rgb_roi": args.rgb_roi,
        "event_roi": args.event_roi,
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(estimate, indent=2))
    print(f"Wrote {output}: offset_us={estimate['offset_us']:.1f}, "
          f"confidence={estimate['confidence']:.3f}")


if __name__ == "__main__":
    main()
