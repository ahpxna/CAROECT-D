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

⚠ TIMING CAVEAT: every RecordHeader.timestampNs in this recording was 0 — this is now
CONFIRMED expected behavior, not a bug (see evs_recorder.cpp's Branch-B comment and the
Step-0 --debug-buffers diagnostic: HasImageData()==false for this camera's EVT3.0
payload buffers, and Arena/IBuffer.h has no GetTimestampNs()/GetTimestamp() of its own —
those exist only on IImage/ICompressedImage). So we have NO real per-record capture time
from the RecordHeader. Playback here uses a fixed --fps you choose (default 15) — the
video's timing is NOT physically accurate to the original capture rate. Treat this
purely as a visual sanity check, not a timing-correct reconstruction. The REAL per-event
microsecond timestamp lives inside each EVT3.0 payload (TIME_LOW/TIME_HIGH words) and is
decoded properly by cevt_to_events.py, not by this script.

Usage:
    pip install opencv-python numpy   # if not already installed
    python cevt_to_video.py test.cevt
    python cevt_to_video.py test.cevt --output out.mp4 --fps 15
    python cevt_to_video.py test.cevt --max-frames 50   # quick preview
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import cv2

FILE_HEADER_FMT = "<8sQIII Q"
FILE_HEADER_SIZE = struct.calcsize(FILE_HEADER_FMT)
RECORD_HEADER_FMT = "<QQQ"
RECORD_HEADER_SIZE = struct.calcsize(RECORD_HEADER_FMT)

# Same word-type constants as inspect_cevt.py — standard public Prophesee
# EVT3.0 encoding, tested here as the hypothesis for non-dense-sized records.
EVT_ADDR_Y, EVT_ADDR_X = 0x0, 0x2
VECT_BASE_X, VECT_12, VECT_8 = 0x3, 0x4, 0x5
TIME_LOW, TIME_HIGH = 0x6, 0x8
EXT_TRIGGER, OTHERS, CONTINUED = 0xA, 0xE, 0xF


def read_file_header(f):
    raw = f.read(FILE_HEADER_SIZE)
    if len(raw) < FILE_HEADER_SIZE:
        raise ValueError("File too short for FileHeader — not a valid .cevt file?")
    magic, pixel_format, bpp, width, height, _reserved = struct.unpack(FILE_HEADER_FMT, raw)
    return {
        "magic": magic.split(b"\x00")[0].decode("ascii", errors="replace"),
        "pixel_format_or_payload_type": pixel_format,
        "bits_per_pixel": bpp,
        "width": width,
        "height": height,
    }


def iter_records(f):
    while True:
        raw = f.read(RECORD_HEADER_SIZE)
        if len(raw) == 0:
            return
        if len(raw) < RECORD_HEADER_SIZE:
            return
        frame_id, timestamp_ns, payload_size = struct.unpack(RECORD_HEADER_FMT, raw)
        payload = f.read(payload_size)
        if len(payload) < payload_size:
            return
        yield frame_id, timestamp_ns, payload


def decode_evt3(payload: bytes):
    """Same decoder as inspect_cevt.py. Returns list of (x, y, t, p)."""
    events = []
    n_words = len(payload) // 2
    cur_y, cur_p = 0, 0
    time_low, time_high, base_x = 0, 0, 0

    for i in range(n_words):
        word = struct.unpack_from("<H", payload, i * 2)[0]
        wtype = (word >> 12) & 0xF
        value = word & 0x0FFF

        if wtype == EVT_ADDR_Y:
            cur_y = value & 0x7FF
        elif wtype == EVT_ADDR_X:
            cur_p = (value >> 11) & 0x1
            x = value & 0x7FF
            events.append((x, cur_y, (time_high << 12) | time_low, cur_p))
        elif wtype == VECT_BASE_X:
            cur_p = (value >> 11) & 0x1
            base_x = value & 0x7FF
        elif wtype == VECT_12:
            for bit in range(12):
                if value & (1 << bit):
                    events.append((base_x + bit, cur_y, (time_high << 12) | time_low, cur_p))
            base_x += 12
        elif wtype == VECT_8:
            for bit in range(8):
                if value & (1 << bit):
                    events.append((base_x + bit, cur_y, (time_high << 12) | time_low, cur_p))
            base_x += 8
        elif wtype == TIME_LOW:
            time_low = value
        elif wtype == TIME_HIGH:
            time_high = value
        # EXT_TRIGGER / OTHERS / CONTINUED / unknown: not a CD event, skip

    return events


def render_dense(payload: bytes, width: int, height: int) -> np.ndarray:
    """Payload is already a baseline-128 accumulated frame — use directly."""
    return np.frombuffer(payload, dtype=np.uint8).reshape(height, width).copy()


def render_sparse_evt3(payload: bytes, width: int, height: int):
    """Decode as EVT3.0 and rasterize onto a fresh baseline-128 canvas.
    Returns (frame, n_events) — n_events lets the caller judge plausibility."""
    frame = np.full((height, width), 128, dtype=np.uint8)
    events = decode_evt3(payload)
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
        # width/height in the FileHeader were 0 for this camera (HasImageData()
        # was false at capture time) — fall back to the known sensor resolution.
        width = header["width"] or args.sensor_width
        height = header["height"] or args.sensor_height
        size_src = "from FileHeader" if header["width"] else "fallback (FileHeader had 0x0)"
        print(f"Using frame size {width}x{height} ({size_src})")

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
        frame_idx = 0

        for frame_id, _timestamp_ns, payload in iter_records(f):
            if args.max_frames and frame_idx >= args.max_frames:
                break

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
                n_skipped += 1
                print(f"  [skip] frame {frame_idx} (frameId={frame_id}): "
                      f"{len(payload)} bytes, odd length, no method applies")
                frame_idx += 1
                continue

            if writer is not None:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
            if png_dir is not None:
                cv2.imwrite(str(png_dir / f"frame_{frame_idx:04d}.png"), frame)

            if frame_idx < 10 or frame_idx % 25 == 0:
                print(f"  frame {frame_idx:>4} (frameId={frame_id}): {len(payload):>8} bytes -> {method}")
            frame_idx += 1

        if writer is not None:
            writer.release()

    print(f"\n{'=' * 60}")
    print(f"Wrote {frame_idx} frames -> {out_path}  ({args.fps} fps playback, NOT "
          "physically timed — see caveat at top of this script)")
    print(f"  dense frames             : {n_dense}")
    print(f"  sparse EVT3.0 frames     : {n_sparse}  (decoded events successfully)")
    print(f"  sparse EVT3.0, 0 events  : {n_sparse_empty}  (decode ran but found nothing)")
    print(f"  skipped (odd length)     : {n_skipped}")
    print(f"{'=' * 60}")
    print(f"\nOpen {out_path} and watch it. If the sparse-decoded frames look like real motion "
          "(not noise), the EVT3.0 hypothesis for those records is confirmed. If they look like "
          "random static, send this console output back and we'll try something else for those.")


if __name__ == "__main__":
    main()
