#!/usr/bin/env python3
"""
raw_to_video.py — Render a .raw recording (evs_recorder_mv.cpp) as a watchable video.

HOW THIS DIFFERS FROM cevt_to_video.py
----------------------------------------
cevt_to_video.py existed to disambiguate two live hypotheses per-record (dense
accumulated frame vs. sparse EVT3.0 words) on the Arena/.cevt path, because
different records in the same file could turn out to be either. That question
does not exist here: evs_recorder_mv.cpp writes a standard Prophesee .raw file
containing ONLY real sparse (x,y,p,t) events with genuine per-event microsecond
timestamps from the sensor (see raw_to_events.py's module docstring for the
full evidence trail). There is nothing to disambiguate — every event is the
same kind of event.

What replaces the hypothesis test is windowing: sparse events have to be
grouped into fixed-duration time windows to become watchable frames at all.
--fps here controls the WINDOW WIDTH events are grouped into (1/fps seconds
of real capture time per frame) as well as the playback rate, so unlike
cevt_to_video.py's --fps (which was decoupled from any physical meaning),
here it does correspond to something real: how much wall-clock time each
frame represents. It still is not a claim about display quality — a busy
scene at high --fps can have very sparse-looking frames; that's expected,
not a bug.

RENDERING: same visual convention as cevt_to_video.py / read_evt3.py's
events_to_frame(), grayscale flavor — baseline gray 128, ON -> white (255),
OFF -> black (0). Within one window, if a pixel gets more than one event, the
LAST event in time wins (matches cevt_to_video.py's render_sparse_evt3()).
This is a visualization convenience, not the 3-channel ON-count/OFF-count/
time-surface representation build_event_dataset.py renders for training —
different purpose, don't confuse the two.

TWO DECODE BACKENDS (shared with raw_to_events.py, not reimplemented here)
----------------------------------------------------------------------------
  1. metavision_core.event_io.EventsIterator(delta_t=frame_period_us) — asks
     the SDK itself for pre-windowed chunks at exactly the playback frame
     rate. Default; efficient even for long recordings since it streams.
  2. Pure-python EVT3.0 fallback — decodes the whole file up front via
     inspect_cevt.decode_evt3() + raw_to_events.unwrap_evt3_time() (the SAME
     functions raw_to_events.py uses — see that file's comment on why a
     fourth copy of this decoder was a real, previously-shipped bug), then
     buckets events into windows with np.searchsorted, the same windowing
     pattern build_event_dataset.py already uses for training-image windows.
     Loads the entire event stream into memory first; fine for a sanity clip,
     not for an hour of footage — prefer backend 1 for those.

Usage:
    pip install opencv-python numpy   # if not already installed
    python raw_to_video.py run01.raw
    python raw_to_video.py run01.raw --output out.mp4 --fps 30
    python raw_to_video.py run01.raw --max-frames 50        # quick preview
    python raw_to_video.py run01.raw --force-fallback       # skip the SDK backend
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from raw_to_events import (  # noqa: E402 -- reuse, don't reimplement (see docstring)
    _load_sidecar_meta,
    _strip_ascii_header,
    unwrap_evt3_time,
)
from inspect_cevt import decode_evt3


def render_window(xs: np.ndarray, ys: np.ndarray, ps: np.ndarray,
                  width: int, height: int) -> np.ndarray:
    """Baseline-128 canvas, ON->255, OFF->0, last event per pixel wins.
    Same convention as cevt_to_video.py's render_sparse_evt3()."""
    frame = np.full((height, width), 128, dtype=np.uint8)
    if len(xs):
        in_bounds = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        xs, ys, ps = xs[in_bounds], ys[in_bounds], ps[in_bounds]
        frame[ys, xs] = np.where(ps > 0, 255, 0).astype(np.uint8)
    return frame


def _frames_from_sdk(path: Path, frame_period_us: int, width, height, max_frames):
    """Backend 1: ask metavision_core for pre-windowed chunks directly."""
    from metavision_core.event_io import EventsIterator
    it = EventsIterator(str(path), delta_t=frame_period_us, relative_timestamps=False)
    sdk_w, sdk_h = it.get_size()
    w = width or sdk_w
    h = height or sdk_h
    n = 0
    for ev in it:
        if max_frames and n >= max_frames:
            return
        if ev.size == 0:
            frame = np.full((h, w), 128, dtype=np.uint8)
        else:
            frame = render_window(ev["x"], ev["y"], ev["p"], w, h)
        yield frame, int(ev.size)
        n += 1


def _frames_from_fallback(path: Path, frame_period_us: int, width, height, max_frames):
    """Backend 2: decode the whole file once, then window it with
    np.searchsorted — same pattern build_event_dataset.py's render_window()
    caller already uses for training-image windows."""
    data = np.fromfile(path, dtype=np.uint8)
    hdr_text, off = _strip_ascii_header(data)
    if "EVT3" not in hdr_text and "evt3" not in hdr_text and off > 0:
        raise RuntimeError(
            "Fallback decoder only supports EVT3.0, and this file's ASCII header "
            "does not declare it. Install metavision_core (backend 1) instead of "
            f"trusting a guess.\nHeader was:\n{hdr_text}"
        )
    payload = data[off:]
    if len(payload) % 2 != 0:
        payload = payload[:-1]
    print("[info] backend 2 (pure-python EVT3.0): decoding the whole file into "
          "memory first -- fine for a sanity clip, slow/heavy for long recordings.")
    events = decode_evt3(payload.tobytes(), max_events=None)
    if not events:
        raise RuntimeError("EVT3 fallback decoded 0 events -- likely wrong encoding.")

    arr = np.asarray(events)
    xs_all = arr[:, 0].astype(np.int64)
    ys_all = arr[:, 1].astype(np.int64)
    ts_raw = arr[:, 2].astype(np.int64)
    ps_all = arr[:, 3].astype(np.int64)
    ts_all = unwrap_evt3_time(ts_raw)

    w = width or int(xs_all.max()) + 1
    h = height or int(ys_all.max()) + 1

    t_start, t_end = int(ts_all[0]), int(ts_all[-1])
    n_frames = max(1, (t_end - t_start) // frame_period_us + 1)
    edges = t_start + np.arange(n_frames + 1) * frame_period_us

    for i in range(int(n_frames)):
        if max_frames and i >= max_frames:
            return
        lo = int(np.searchsorted(ts_all, edges[i], side="left"))
        hi = int(np.searchsorted(ts_all, edges[i + 1], side="left"))
        frame = render_window(xs_all[lo:hi], ys_all[lo:hi], ps_all[lo:hi], w, h)
        yield frame, hi - lo


def main():
    ap = argparse.ArgumentParser(description="Render a .raw recording as a video")
    ap.add_argument("path", help="Path to the .raw file (from evs_recorder_mv.cpp)")
    ap.add_argument("--output", default=None, help="Output video path (default: <input>.mp4)")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="Both the window width events are grouped into (1/fps seconds of "
                         "real capture time per frame) AND the playback rate — see the "
                         "module docstring for why that's meaningful here, unlike "
                         "cevt_to_video.py's --fps")
    ap.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (quick preview)")
    ap.add_argument("--sensor-width", type=int, default=None,
                    help="Override sensor width (default: from .meta.json sidecar, else "
                         "from the decode backend)")
    ap.add_argument("--sensor-height", type=int, default=None)
    ap.add_argument("--png-dir", default=None,
                    help="Also (or instead) dump every frame as an individual PNG into this "
                         "folder — bypasses any video-codec issues entirely, use this if the "
                         "mp4 looks blank/wrong")
    ap.add_argument("--no-video", action="store_true",
                    help="Skip writing the mp4 entirely (use with --png-dir)")
    ap.add_argument("--force-fallback", action="store_true",
                    help="Skip the SDK backend, use the pure-python EVT3.0 decoder instead")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)
    out_path = args.output or str(path.with_suffix(".mp4"))

    png_dir = None
    if args.png_dir:
        png_dir = Path(args.png_dir)
        png_dir.mkdir(parents=True, exist_ok=True)

    _meta, meta_w, meta_h = _load_sidecar_meta(path)
    width = args.sensor_width or meta_w
    height = args.sensor_height or meta_h
    frame_period_us = int(round(1_000_000.0 / args.fps))

    backend = None
    frame_gen = None
    if not args.force_fallback:
        try:
            frame_gen = _frames_from_sdk(path, frame_period_us, width, height, args.max_frames)
            backend = "metavision_core"
        except ImportError:
            print("[warn] metavision_core not importable; using pure-python EVT3.0 fallback.")
    if frame_gen is None:
        frame_gen = _frames_from_fallback(path, frame_period_us, width, height, args.max_frames)
        backend = "evt3_fallback"
    print(f"[info] decoder backend: {backend}")
    print(f"[info] window = playback = {args.fps} fps  ({frame_period_us} us/frame of real capture time)")

    writer = None
    if not args.no_video:
        # NOTE: writing as COLOR (3-channel BGR) even though the content is
        # grayscale — see cevt_to_video.py's comment: cv2.VideoWriter with
        # isColor=False + mp4v is known to silently produce blank/corrupted
        # output on some systems. Converting to BGR before writing is the
        # standard, reliable workaround, kept identical here.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # width/height might still be unknown until the first frame arrives
        # (fallback backend infers them from the data itself) -- open the
        # writer lazily on frame 0 instead of guessing.
        writer_size = (width, height) if (width and height) else None
    else:
        writer_size = None

    frames_written = 0
    total_events = 0
    try:
        for frame, n_events in frame_gen:
            if writer is None and not args.no_video:
                h, w = frame.shape[:2]
                writer_size = writer_size or (w, h)
                writer = cv2.VideoWriter(out_path, fourcc, args.fps, writer_size, isColor=True)
                if not writer.isOpened():
                    print(f"Could not open video writer for {out_path} — is opencv-python "
                          "installed with a working codec backend?")
                    sys.exit(1)

            if writer is not None:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
            if png_dir is not None:
                cv2.imwrite(str(png_dir / f"frame_{frames_written:04d}.png"), frame)

            if frames_written < 10 or frames_written % 25 == 0:
                print(f"  frame {frames_written:>4}: {n_events:>7,} event(s)")
            frames_written += 1
            total_events += n_events
    finally:
        # release() in a finally block: an exception mid-loop previously left
        # the mp4 unfinalised (no moov atom) on the .cevt path — same fix here.
        if writer is not None:
            writer.release()

    print(f"\n{'=' * 60}")
    print(f"Wrote {frames_written} frame(s), {total_events:,} event(s) total -> {out_path}")
    print(f"  ({args.fps} fps = both window width and playback rate, real capture time)")
    print(f"  decoder backend: {backend}")
    print(f"{'=' * 60}")
    if frames_written == 0:
        print("\n0 frames written -- either the file has 0 events, or --max-frames 0 was "
              "passed. Check with: python raw_to_events.py <file> --max-events 10")
    else:
        print(f"\nOpen {out_path} and watch it as a visual sanity check of the scene. If it's "
              "mostly gray with occasional flickers, that can be entirely normal for a real "
              "sparse event stream at high --fps on a mostly-static scene -- try a lower "
              "--fps (wider windows) before concluding the recording is empty.")


if __name__ == "__main__":
    main()
