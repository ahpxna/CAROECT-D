#!/usr/bin/env python3
"""Transfer exact SAM3 frame observations to causal detector windows.

For every SAM frame timestamp t_k, the normal dataset path creates the sample
window [t_k - delta_t, t_k) and copies only boxes/masks observed at frame k.
There is no detector-box interpolation and no future-frame influence.

Physical RGB/event pairing requires a measured sync.json artifact unless the
operator explicitly passes --allow-unsynced. Synthetic events use zero offset
by construction.
"""

import argparse
import bisect
import json
from pathlib import Path

import h5py
import numpy as np


def load_tracks(path: str):
    """Load tracks while preserving each frame observation's numeric values."""
    payload = json.loads(Path(path).read_text())
    fps = float(payload["fps"])
    explicit = payload.get("frame_times_us")
    if explicit is None:
        n_frames = int(payload.get("n_frames", 0))
        if not n_frames:
            n_frames = 1 + max(
                (int(row["frame_idx"]) for entry in payload["tracks"].values()
                 for row in entry.get("frames", [])),
                default=-1,
            )
        explicit = [round(index * 1e6 / fps) for index in range(n_frames)]
    frame_times = np.asarray(explicit, dtype=np.int64)

    tracks = {}
    base_dir = Path(path).resolve().parent
    for numeric_id, (track_id, entry) in enumerate(payload["tracks"].items()):
        observations = sorted(entry.get("frames", []), key=lambda row: int(row["frame_idx"]))
        tracks[track_id] = {
            "numeric_id": numeric_id,
            "class_id": int(entry["class_id"]),
            "class_name": entry.get("class_name"),
            "base_dir": base_dir,
            "observations": observations,
            "by_frame": {int(row["frame_idx"]): row for row in observations},
        }
    return fps, int(payload["width"]), int(payload["height"]), frame_times, tracks, payload


def load_events_h5(path: str):
    with h5py.File(path, "r") as handle:
        events = {key: handle[key][:] for key in ("x", "y", "t", "p")}
        attrs = dict(handle.attrs)
    order = np.argsort(events["t"], kind="stable")
    if not np.array_equal(order, np.arange(len(order))):
        events = {key: value[order] for key, value in events.items()}
    return events, attrs


def is_synthetic_event_file(attrs: dict) -> bool:
    return bool(attrs.get("simulator") or attrs.get("synthetic", False))


def load_sync_offset(sync_json: str | None) -> tuple[float, dict | None]:
    """Return offset under the convention event_time_us = rgb_time_us + offset_us."""
    if sync_json is None:
        return 0.0, None
    payload = json.loads(Path(sync_json).read_text())
    expected = "event_time_us = rgb_time_us + offset_us"
    if payload.get("convention") != expected:
        raise ValueError(
            f"sync.json convention must be {expected!r}, got {payload.get('convention')!r}"
        )
    if "offset_us" not in payload:
        raise ValueError("sync.json is missing offset_us")
    return float(payload["offset_us"]), payload


def resolve_clock_alignment(attrs: dict, sync_json: str | None, allow_unsynced: bool):
    """Resolve RGB->event clock mapping without permitting synthetic offsets."""
    synthetic = is_synthetic_event_file(attrs)
    if synthetic:
        if sync_json is not None:
            raise ValueError(
                "--sync-json is invalid for synthetic events: their RGB/event "
                "clock offset is zero by construction"
            )
        return True, 0.0, None

    offset_us, sync_payload = load_sync_offset(sync_json)
    if sync_payload is None and not allow_unsynced:
        raise RuntimeError(
            "Physical RGB/event pairing requires --sync-json. Pass --allow-unsynced "
            "only when deliberately accepting an unmeasured zero offset."
        )
    return False, offset_us, sync_payload


def build_causal_windows(
    frame_times_us: np.ndarray,
    tracks: dict,
    window_us: float,
    offset_us: float = 0.0,
) -> list[dict]:
    """Create [t_k-window_us, t_k) samples with exact frame-k observations."""
    if window_us <= 0:
        raise ValueError("window_us must be positive")
    frame_times = np.asarray(frame_times_us, dtype=np.float64)
    if not len(frame_times):
        return []
    clip_start_rgb_us = float(frame_times[0])
    windows = []
    for frame_idx, rgb_t_us in enumerate(frame_times):
        # A causal sample is valid only when the RGB source contains the full
        # requested history. Do not create positive labels over a padded/empty
        # pre-roll before the clip starts.
        if float(rgb_t_us) - clip_start_rgb_us < float(window_us):
            continue
        end_us = float(rgb_t_us) + float(offset_us)
        boxes = []
        for track_id in sorted(tracks):
            track = tracks[track_id]
            observation = track["by_frame"].get(frame_idx)
            if observation is None:
                continue
            # Copy the observed values directly. Do not clip, interpolate, or
            # consult frame k+1: exact frame observations are the target labels.
            box = {
                "track_id": track_id,
                "class_id": track["class_id"],
                "cx": observation["cx"],
                "cy": observation["cy"],
                "w": observation["w"],
                "h": observation["h"],
                "mask_path": observation.get("mask_path"),
                "score": observation.get("score"),
                "sam_source": observation.get("chosen_source", observation.get("source")),
            }
            boxes.append(box)
        windows.append({
            "sample_index": len(windows),
            "frame_idx": frame_idx,
            "rgb_t_us": int(rgb_t_us),
            "t_start_us": end_us - float(window_us),
            "t_end_us": end_us,
            "label_time_us": end_us,
            "boxes": boxes,
        })
    return windows


def event_index_bounds(event_times: np.ndarray, start_us: float, end_us: float) -> tuple[int, int]:
    """Indices implementing the causal half-open interval [start_us, end_us)."""
    lo = int(np.searchsorted(event_times, start_us, side="left"))
    hi = int(np.searchsorted(event_times, end_us, side="left"))
    return lo, hi


# Legacy interpolation is retained only for visual diagnostics. The normal
# dataset path above never calls these functions.
def _find_bracket(t: np.ndarray, query: float, max_gap_us: float):
    if len(t) < 2 or query < t[0] or query > t[-1]:
        return None, None
    index = bisect.bisect_right(t, query) - 1
    index = min(max(index, 0), len(t) - 2)
    lo, hi = t[index], t[index + 1]
    if hi - lo > max_gap_us:
        return None, None
    alpha = 0.0 if hi == lo else (query - lo) / (hi - lo)
    return index, alpha


def interpolate_box(track: dict, query: float, max_gap_us: float):
    rows = track["observations"]
    times = np.asarray([row["t_us"] for row in rows], dtype=np.float64)
    index, alpha = _find_bracket(times, query, max_gap_us)
    if index is None:
        return None
    a, b = rows[index], rows[index + 1]
    return tuple((1.0 - alpha) * float(a[key]) + alpha * float(b[key])
                 for key in ("cx", "cy", "w", "h"))


def _active_obs_bracket(track: dict, query: float, max_gap_us: float):
    times = np.asarray([row["t_us"] for row in track["observations"]], dtype=np.float64)
    return _find_bracket(times, query, max_gap_us)


def build_legacy_interpolated_windows(
    frame_times_us: np.ndarray,
    tracks: dict,
    window_us: float,
    max_gap_us: float,
    offset_us: float,
) -> list[dict]:
    """Debug-only interpolation, enabled solely by --legacy-interpolation."""
    output = []
    for frame_idx, rgb_t in enumerate(frame_times_us):
        end = float(rgb_t) + offset_us
        boxes = []
        for track_id in sorted(tracks):
            track = tracks[track_id]
            values = interpolate_box(track, float(rgb_t), max_gap_us)
            if values is None:
                continue
            boxes.append(dict(
                track_id=track_id,
                class_id=track["class_id"],
                **dict(zip(("cx", "cy", "w", "h"), values)),
                legacy_interpolated=True,
            ))
        output.append(dict(
            sample_index=frame_idx,
            frame_idx=frame_idx,
            rgb_t_us=int(rgb_t),
            t_start_us=end - window_us,
            t_end_us=end,
            label_time_us=end,
            boxes=boxes,
        ))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-us", type=float, required=True)
    parser.add_argument(
        "--legacy-max-gap-frames",
        "--max-gap-frames",
        dest="legacy_max_gap_frames",
        type=int,
        default=5,
        help="LEGACY DEBUG ONLY: maximum interpolation gap in RGB frames",
    )
    parser.add_argument("--sync-json")
    parser.add_argument(
        "--allow-unsynced",
        action="store_true",
        help="Explicitly accept zero RGB/event offset for a physical recording",
    )
    parser.add_argument(
        "--legacy-interpolation",
        action="store_true",
        help="Debug visualization only; interpolate boxes at sample timestamps",
    )
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    fps, width, height, frame_times, tracks, tracks_payload = load_tracks(args.tracks)
    events, attrs = load_events_h5(args.events)
    synthetic, offset_us, sync_payload = resolve_clock_alignment(
        attrs, args.sync_json, args.allow_unsynced
    )

    if args.legacy_interpolation:
        windows = build_legacy_interpolated_windows(
            frame_times, tracks, args.window_us,
            args.legacy_max_gap_frames * 1e6 / fps, offset_us,
        )
        label_semantics = "legacy_interpolated_debug"
    else:
        windows = build_causal_windows(frame_times, tracks, args.window_us, offset_us)
        label_semantics = "causal_exact_frame_observation"

    for window in windows:
        lo, hi = event_index_bounds(events["t"], window["t_start_us"], window["t_end_us"])
        window["event_index_start"] = lo
        window["event_index_end"] = hi
        window["n_events"] = hi - lo

    result = {
        "schema_version": 2,
        "fps": fps,
        "fps_provenance": tracks_payload.get(
            "fps_provenance", "camera.fps_original used to derive SAM frame timestamps"
        ),
        "width": width,
        "height": height,
        "n_frames": len(frame_times),
        "frame_times_us": [int(value) for value in frame_times],
        "window_us": float(args.window_us),
        "interval": "[t_k-window_us,t_k)",
        "label_semantics": label_semantics,
        "clock_alignment": {
            "offset_us": offset_us,
            "convention": "event_time_us = rgb_time_us + offset_us",
            "source": "synthetic_zero_by_construction" if synthetic else (
                str(Path(args.sync_json).resolve()) if args.sync_json else "explicit_unsynced_override"
            ),
        },
        "tracks_base_dir": str(Path(args.tracks).resolve().parent),
        "windows": windows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))

    if args.stats:
        counts = np.asarray([window["n_events"] for window in windows])
        labeled = sum(bool(window["boxes"]) for window in windows)
        print(
            f"{len(windows)} samples, {labeled} with labels, "
            f"{int(counts.sum()) if len(counts) else 0} windowed events, "
            f"window={args.window_us:.0f} us, offset={offset_us:.1f} us"
        )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
