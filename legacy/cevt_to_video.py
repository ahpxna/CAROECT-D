#!/usr/bin/env python3
"""
cevt_to_video.py — Render a CAROECT-D .cevt recording as a watchable video.

WHY THIS EXISTS
---------------
inspect_cevt.py confirmed record #1 (921,600 bytes = exactly 1280x720) is a
DENSE accumulated frame (baseline gray 128, ON->white, OFF->black). But the
other 152 records average only ~133 KB — far smaller — so they can't be the
same kind of dense frame. Two live hypotheses, tested per-record below:

  A) DENSE  : len(payload) == width*height        -> reshape directly, done.
  B) SPARSE : otherwise                           -> try decoding as standard
              Prophesee EVT3.0 event words (see inspect_cevt.py for the same
              decoder) and rasterize the decoded (x,y,p) events onto a blank
              baseline-128 canvas the same way real EVT3.0 event frames are
              visualized (also matches this project's own read_evt3.py
              events_to_frame() convention).

Whichever hypothesis wins for a given record, this script logs it, so after
one run we'll know empirically which records are dense and which are sparse
— and get a video either way.

⚠ TIMING CAVEAT: this script's --fps controls PLAYBACK SPEED only. It has nothing to do
with the capture timebase and never did. Do not use the video to reason about timing.

The real capture timebase now lives in the container itself: CAROEVT2 recordings carry
a per-record device and/or host timestamp plus the camera's own AcquisitionFrameRate.
To see it, use the diagnostics rather than this renderer:
    python cevt_to_events.py <file>.cevt --debug-time-continuity
    python inspect_cevt.py  <file>.cevt

Note there is NO per-event microsecond timestamp anywhere in this data — the camera
emits dense accumulated frames, so sub-window ordering is destroyed in hardware. See
cevt_to_events.py's module docstring.

Usage:
    pip install opencv-python numpy   # if not already installed
    python cevt_to_video.py test.cevt
    python cevt_to_video.py test.cevt --output out.mp4 --fps 15
    python cevt_to_video.py test.cevt --max-frames 50   # quick preview
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import cv2

# Container parsing and the EVT3.0 decoder are imported from inspect_cevt rather
# than copy-pasted. They used to exist in three near-identical copies here, in
# inspect_cevt.py and in cevt_to_events.py; when the container gained CAROEVT2 the
# copies would have silently diverged, and a reader that misparses a header does
# not fail loudly — it renders convincing garbage. One definition, one place.
from inspect_cevt import (  # noqa: E402
    read_file_header,
    iter_records,
    decode_evt3,
    TS_SOURCE_NAME,
)


def render_dense(payload: bytes, width: int, height: int) -> np.ndarray:
    """Payload is already a baseline-128 accumulated frame — use directly."""
    return np.frombuffer(payload, dtype=np.uint8).reshape(height, width).copy()


def render_sparse_evt3(payload: bytes, width: int, height: int):
    """Decode as EVT3.0 and rasterize onto a fresh baseline-128 canvas.
    Returns (frame, n_events, n_in_bounds) — the counts let the caller judge
    plausibility. Note decode_evt3() from inspect_cevt takes a max_events kwarg;
    passing None decodes the whole payload."""
    frame = np.full((height, width), 128, dtype=np.uint8)
    events = decode_evt3(payload, max_events=None)
    plausible = 0
    for x, y, _t, p in events:
        if 0 <= x < width and 0 <= y < height:
            frame[y, x] = 255 if p else 0
            plausible += 1
    return frame, len(events), plausible


def main():
    ap = argparse.ArgumentParser(description="Render a .cevt recording as a video")
    ap.add_argument("path", help="Path to the .cevt file")
    ap.add_argument("--output", default=None, help="Output video path (default: <input>.mp4)")
    ap.add_argument("--fps", type=float, default=15.0,
                    help="Playback fps (NOT physically accurate — see caveat in file header)")
    ap.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (quick preview)")
    ap.add_argument("--sensor-width", type=int, default=1280)
    ap.add_argument("--sensor-height", type=int, default=720)
    ap.add_argument("--png-dir", default=None,
                    help="Also (or instead) dump every frame as an individual PNG into this "
                         "folder — bypasses any video-codec issues entirely, use this if the "
                         "mp4 looks blank/wrong")
    ap.add_argument("--no-video", action="store_true",
                    help="Skip writing the mp4 entirely (use with --png-dir)")
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

    with open(path, "rb") as f:
        header = read_file_header(f)
        # width/height in the FileHeader are 0 whenever the first buffer could not
        # be cast to IImage at capture time — fall back to the known sensor size.
        width = header["width"] or args.sensor_width
        height = header["height"] or args.sensor_height
        size_src = "from FileHeader" if header["width"] else "fallback (FileHeader had 0x0)"
        print(f"Container CAROEVT{header['version']} | frame size {width}x{height} ({size_src})")

        writer = None
        if not args.no_video:
            # NOTE: writing as COLOR (3-channel BGR) even though the content is
            # grayscale. cv2.VideoWriter with isColor=False + the mp4v codec is
            # known to silently produce blank/corrupted output on some systems
            # (a long-standing OpenCV/codec quirk) - converting to BGR before
            # writing is the standard, reliable workaround.
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, args.fps, (width, height), isColor=True)
            if not writer.isOpened():
                print(f"Could not open video writer for {out_path} — is opencv-python installed "
                      "with a working codec backend?")
                sys.exit(1)

        n_dense = 0
        n_sparse = 0
        n_sparse_empty = 0
        n_skipped = 0
        n_records = 0     # records consumed
        frames_written = 0  # frames actually rendered — NOT the same number

        try:
            for frame_id, _dev_ns, _host_ns, _ts_src, payload in iter_records(f, header):
                if args.max_frames and n_records >= args.max_frames:
                    break
                n_records += 1

                expected = width * height
                if len(payload) == expected:
                    frame = render_dense(payload, width, height)
                    n_dense += 1
                    method = "dense"
                elif len(payload) % 2 == 0:
                    frame, n_events, plausible = render_sparse_evt3(payload, width, height)
                    if n_events == 0:
                        n_sparse_empty += 1
                        method = "sparse-evt3(0 events)"
                    else:
                        n_sparse += 1
                        method = f"sparse-evt3({n_events} events, {plausible} in-bounds)"
                else:
                    # Skipped records used to still bump frame_idx, so the final
                    # "Wrote N frames" line counted frames that were never written
                    # and the PNG filenames left gaps. Counters are now separate.
                    n_skipped += 1
                    print(f"  [skip] record {n_records - 1} (frameId={frame_id}): "
                          f"{len(payload)} bytes, odd length, no method applies")
                    continue

                if writer is not None:
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
                if png_dir is not None:
                    cv2.imwrite(str(png_dir / f"frame_{frames_written:04d}.png"), frame)

                if frames_written < 10 or frames_written % 25 == 0:
                    print(f"  frame {frames_written:>4} (frameId={frame_id}): "
                          f"{len(payload):>8} bytes -> {method}")
                frames_written += 1
        finally:
            # release() in a finally block: an exception mid-loop previously left
            # the mp4 unfinalised (no moov atom), i.e. an unplayable file with no
            # indication of why.
            if writer is not None:
                writer.release()

    print(f"\n{'=' * 60}")
    print(f"Read {n_records} records, wrote {frames_written} frames -> {out_path}")
    print(f"  ({args.fps} fps is PLAYBACK SPEED only — see the caveat at the top)")
    print(f"  dense frames             : {n_dense}")
    print(f"  sparse EVT3.0 frames     : {n_sparse}  (decoded events successfully)")
    print(f"  sparse EVT3.0, 0 events  : {n_sparse_empty}  (decode ran but found nothing)")
    print(f"  skipped (odd length)     : {n_skipped}")
    print(f"{'=' * 60}")
    if n_dense and not n_sparse:
        print("\nAll renderable records were dense accumulated frames — the expected result "
              "for this camera. Open the video as a visual sanity check of the scene.")
    elif n_sparse:
        print(f"\n{n_sparse} record(s) decoded as sparse EVT3.0. That would contradict the "
              "firmware-lock finding (AcquisitionAccumulationMode IsAvailable=false), so "
              "check whether those frames show real motion or random static before "
              "believing it — random bytes decode to in-bounds coordinates quite often.")


if __name__ == "__main__":
    main()
