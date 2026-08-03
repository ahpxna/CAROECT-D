#!/usr/bin/env python3
"""
inspect_cevt.py — Read and sanity-check a CAROECT-D .cevt recording.

WHAT .cevt IS
-------------
NOT a video/image file — a custom binary container written by evs_recorder.cpp.
Layout:
    FileHeader                     (36 bytes, once)
    RecordHeader + payload bytes   (repeated once per Arena buffer)
    RecordHeader + payload bytes
    ...

FileHeader (36 bytes, packed):
    char[8]  magic          "CAROEVT1"
    uint64   pixelFormat    (or payloadType if buffer wasn't image-typed)
    uint32   bitsPerPixel
    uint32   width
    uint32   height
    uint64   reserved

RecordHeader (24 bytes, packed):
    uint64   frameId
    uint64   timestampNs    (buffer-level timestamp, NOT per-event)
    uint64   payloadSize    (bytes immediately following this header)

WHAT THIS SCRIPT DOES
---------------------
1. Parses the container losslessly (100% our own known format — safe).
2. Prints a hex dump of the first payload so you can SEE the raw bytes.
3. Attempts to decode the first payload as standard Prophesee EVT3.0
   (the well-known public 16-bit-word vectorized format). This is a
   HYPOTHESIS, not a certainty — the camera's only EventFormat entry is
   literally named "EVT3_0", which is a strong hint but not direct proof
   that Arena hands you the bit-exact standard wire format. The script
   reports x/y/t ranges from the decode attempt; if x stays < 1280 and
   y stays < 720 and timestamps trend upward, the hypothesis is likely
   correct. If not, we'll know to look elsewhere instead of guessing.

Usage:
    python inspect_cevt.py test.cevt
    python inspect_cevt.py test.cevt --max-events 20     # print first N decoded events
    python inspect_cevt.py test.cevt --no-decode         # skip EVT3.0 decode, just inspect container
"""

import argparse
import struct
import sys
from pathlib import Path

FILE_HEADER_FMT = "<8sQIII Q"   # magic, pixelFormat, bitsPerPixel, width, height, reserved
FILE_HEADER_SIZE = struct.calcsize(FILE_HEADER_FMT)

RECORD_HEADER_FMT = "<QQQ"      # frameId, timestampNs, payloadSize
RECORD_HEADER_SIZE = struct.calcsize(RECORD_HEADER_FMT)


def read_file_header(f):
    raw = f.read(FILE_HEADER_SIZE)
    if len(raw) < FILE_HEADER_SIZE:
        raise ValueError(f"File too short for FileHeader (need {FILE_HEADER_SIZE} bytes, "
                         f"got {len(raw)}). Is this really a .cevt file?")
    magic, pixel_format, bpp, width, height, reserved = struct.unpack(FILE_HEADER_FMT, raw)
    magic_str = magic.split(b"\x00")[0].decode("ascii", errors="replace")
    if magic_str != "CAROEVT1":
        print(f"  [warning] magic = {magic_str!r}, expected 'CAROEVT1' — "
              "file may be corrupt or not written by evs_recorder.cpp.")
    return {
        "magic": magic_str,
        "pixel_format_or_payload_type": pixel_format,
        "bits_per_pixel": bpp,
        "width": width,
        "height": height,
    }


def iter_records(f):
    """Yields (frame_id, timestamp_ns, payload_bytes) for every record in the file."""
    while True:
        raw = f.read(RECORD_HEADER_SIZE)
        if len(raw) == 0:
            return  # clean EOF
        if len(raw) < RECORD_HEADER_SIZE:
            print(f"  [warning] truncated RecordHeader at end of file "
                  f"({len(raw)} of {RECORD_HEADER_SIZE} bytes) — recording likely cut off mid-write.")
            return
        frame_id, timestamp_ns, payload_size = struct.unpack(RECORD_HEADER_FMT, raw)
        payload = f.read(payload_size)
        if len(payload) < payload_size:
            print(f"  [warning] truncated payload for frame {frame_id} "
                  f"(expected {payload_size}, got {len(payload)}) — recording likely cut off mid-write.")
            return
        yield frame_id, timestamp_ns, payload


def hex_dump(data: bytes, n: int = 64):
    n = min(n, len(data))
    lines = []
    for i in range(0, n, 16):
        chunk = data[i:i + 16]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"  {i:04x}:  {hex_part:<47}  {ascii_part}")
    return "\n".join(lines)


# ── EVT3.0 decode attempt (standard Prophesee public word format) ──────────
# 16-bit little-endian words; top 4 bits (15:12) = word type.
# This is the well-documented, publicly known Prophesee EVT3.0 encoding
# (used by OpenEB / Metavision and everyone downstream of it) — NOT
# something specific to this Lucid camera. We are testing whether the
# camera's payload bytes match this public spec bit-for-bit.
EVT_ADDR_Y   = 0x0
EVT_ADDR_X   = 0x2
VECT_BASE_X  = 0x3
VECT_12      = 0x4
VECT_8       = 0x5
TIME_LOW     = 0x6
TIME_HIGH    = 0x8
EXT_TRIGGER  = 0xA
OTHERS       = 0xE
CONTINUED    = 0xF


def decode_evt3(payload: bytes, max_events: int = None):
    """
    Attempt to decode a byte buffer as standard EVT3.0. Returns a list of
    (x, y, t, p) tuples. Stateful decode: tracks current y, current polarity,
    a 12-bit time-low counter and a rolling time-high base to build a full
    timestamp (matches the public EVT3.0 state machine).
    """
    events = []
    n_words = len(payload) // 2
    if len(payload) % 2 != 0:
        print(f"  [warning] payload size {len(payload)} is odd — not a whole number of "
              "16-bit words; EVT3.0 decode will ignore the trailing byte.")

    cur_y = 0
    cur_p = 0
    time_low = 0
    time_high = 0
    base_x = 0

    for i in range(n_words):
        word = struct.unpack_from("<H", payload, i * 2)[0]
        wtype = (word >> 12) & 0xF
        value = word & 0x0FFF

        if wtype == EVT_ADDR_Y:
            cur_y = value & 0x7FF
        elif wtype == EVT_ADDR_X:
            cur_p = (value >> 11) & 0x1
            x = value & 0x7FF
            t = (time_high << 12) | time_low
            events.append((x, cur_y, t, cur_p))
            if max_events and len(events) >= max_events:
                return events
        elif wtype == VECT_BASE_X:
            cur_p = (value >> 11) & 0x1
            base_x = value & 0x7FF
        elif wtype == VECT_12:
            for bit in range(12):
                if value & (1 << bit):
                    t = (time_high << 12) | time_low
                    events.append((base_x + bit, cur_y, t, cur_p))
                    if max_events and len(events) >= max_events:
                        return events
            base_x += 12
        elif wtype == VECT_8:
            for bit in range(8):
                if value & (1 << bit):
                    t = (time_high << 12) | time_low
                    events.append((base_x + bit, cur_y, t, cur_p))
                    if max_events and len(events) >= max_events:
                        return events
            base_x += 8
        elif wtype == TIME_LOW:
            time_low = value
        elif wtype == TIME_HIGH:
            time_high = value
        elif wtype in (EXT_TRIGGER, OTHERS, CONTINUED):
            pass  # not a CD event; skip
        # unknown word types are silently skipped (forward-compatibility)

    return events


def analyze_as_dense_frame(payload: bytes, width: int, height: int, record_index: int = 0):
    """
    Tests the hypothesis that payload is a DENSE HxW single-channel
    accumulated event frame (baseline=128 gray, ON pushes toward 255,
    OFF pushes toward 0) rather than a sparse list of EVT3.0 event words.
    This matches the classic event-visualization convention (also used in
    this project's own read_evt3.py: frame = np.full((h,w,3), 128, ...)).
    """
    expected_size = width * height
    print(f"Dense-frame hypothesis check ({width}x{height} = {expected_size:,} bytes expected):")

    # Always run the histogram — it's informative even if dimensions are wrong,
    # since "mostly one constant byte value" is a real, size-independent signal
    # (e.g. an all-padding or all-idle buffer looks the same regardless of how
    # you try to reshape it).
    counts = [0] * 256
    for b in payload:
        counts[b] += 1
    n = len(payload)
    baseline = counts[128]
    below = sum(counts[:128])
    above = sum(counts[129:])
    at_0 = counts[0]
    at_255 = counts[255]
    most_common_value = max(range(256), key=lambda v: counts[v])

    print(f"  payload length                : {n:,} bytes")
    print(f"  most common byte value        : {most_common_value} "
          f"({counts[most_common_value]:,} / {n:,} = {100*counts[most_common_value]/n:.2f}%)")
    print(f"  value=128 (neutral/no-change) : {baseline:,}  ({100*baseline/n:.2f}%)")
    print(f"  value <128 (OFF-leaning)      : {below:,}  ({100*below/n:.2f}%)")
    print(f"  value >128 (ON-leaning)       : {above:,}  ({100*above/n:.2f}%)")
    print(f"  value==0   (full OFF)         : {at_0:,}")
    print(f"  value==255 (full ON)          : {at_255:,}")

    if counts[most_common_value] / n > 0.99:
        print(f"  ✓ Over 99% of bytes are the SAME value ({most_common_value}) — this buffer is "
              "essentially uniform/blank, not a real image or real sparse event data.")
    elif baseline / n > 0.9:
        print("  ✓ Over 90% of bytes sit exactly at 128 — consistent with a dense "
              "accumulated event frame (mostly-idle background with some activity).")

    if len(payload) != expected_size:
        print(f"\n  Size does NOT match {width}x{height} ({expected_size:,} bytes) exactly.")
        # Suggest integer factor-pair candidates close to common aspect ratios,
        # to help pin down the real geometry if this IS a dense frame at some
        # other resolution.
        candidates = []
        for h in range(1, int(n ** 0.5) + 1):
            if n % h == 0:
                w = n // h
                candidates.append((w, h))
                candidates.append((h, w))
        candidates = sorted(set(candidates), key=lambda wh: abs(wh[0] / wh[1] - 16 / 9))
        print(f"  Factor pairs of {n:,} closest to a 16:9-ish aspect ratio:")
        for w, h in candidates[:6]:
            print(f"    {w} x {h}")
        print(f"  Re-run with --sensor-width/--sensor-height set to one of these to test it.")
        return

    try:
        from PIL import Image
        import numpy as np
        arr = np.frombuffer(payload, dtype=np.uint8).reshape(height, width)
        out_path = f"record_{record_index}_preview.png"
        Image.fromarray(arr, mode="L").save(out_path)
        print(f"\n  Saved visualization -> {out_path}  (open it and look — you should see "
              "shapes/edges wherever pixels differ from flat gray)")
    except ImportError:
        print("  (install pillow + numpy: pip install pillow numpy  -- to also save a "
              "viewable PNG of this frame)")


def main():
    ap = argparse.ArgumentParser(description="Inspect a CAROECT-D .cevt recording")
    ap.add_argument("path", help="Path to the .cevt file")
    ap.add_argument("--max-events", type=int, default=20,
                    help="Print at most this many decoded events (default 20)")
    ap.add_argument("--no-decode", action="store_true",
                    help="Skip the EVT3.0 decode attempt, just inspect the container")
    ap.add_argument("--sensor-width", type=int, default=1280)
    ap.add_argument("--sensor-height", type=int, default=720)
    ap.add_argument("--record-index", type=int, default=0,
                    help="Which record (0-based) to hex-dump/analyze in detail — "
                         "e.g. --record-index 1 to inspect the SECOND record instead "
                         "of the first (useful now that record 0 differs from the rest)")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    print(f"{'=' * 60}\n  CAROECT-D .cevt inspector — {path.name}\n{'=' * 60}\n")

    with open(path, "rb") as f:
        header = read_file_header(f)
        print("FileHeader:")
        print(f"  magic                       = {header['magic']!r}")
        print(f"  pixelFormat / payloadType   = {header['pixel_format_or_payload_type']} "
              f"(0x{header['pixel_format_or_payload_type']:x})")
        print(f"  bitsPerPixel                = {header['bits_per_pixel']}")
        print(f"  width x height              = {header['width']} x {header['height']}")
        print()

        total_records = 0
        total_payload_bytes = 0
        min_ts, max_ts = None, None
        first_payload = None
        first_frame_id = None
        first_n_sizes = []

        for frame_id, timestamp_ns, payload in iter_records(f):
            total_records += 1
            total_payload_bytes += len(payload)
            if min_ts is None or timestamp_ns < min_ts:
                min_ts = timestamp_ns
            if max_ts is None or timestamp_ns > max_ts:
                max_ts = timestamp_ns
            if total_records - 1 == args.record_index:
                first_payload = payload
                first_frame_id = frame_id
            if len(first_n_sizes) < 10:
                first_n_sizes.append(len(payload))

        print("Container summary:")
        print(f"  total records (Arena buffers) = {total_records}")
        print(f"  total payload bytes            = {total_payload_bytes:,}")
        if total_records > 0:
            print(f"  avg payload bytes/record        = {total_payload_bytes / total_records:.1f}")
        if min_ts is not None:
            print(f"  buffer timestamp range (ns)     = {min_ts} .. {max_ts}  "
                  f"(span {(max_ts - min_ts) / 1e9:.3f} s)")
        print(f"  payload sizes of first {len(first_n_sizes)} records = {first_n_sizes}")
        print()

        if first_payload is None:
            print(f"Record index {args.record_index} not found (only {total_records} "
                  "records in file) — file has only a FileHeader, or index out of range.")
            return

        print(f"Selected record (index {args.record_index}): frameId={first_frame_id}, "
              f"payload size={len(first_payload)} bytes")
        print(f"First 64 bytes of this payload (hex + ascii):")
        print(hex_dump(first_payload, 64))
        print()

        analyze_as_dense_frame(first_payload, args.sensor_width, args.sensor_height, args.record_index)
        print()

        if args.no_decode:
            return

        print(f"Attempting EVT3.0 decode of record {args.record_index} payload (hypothesis test)...")
        events = decode_evt3(first_payload, max_events=None)
        print(f"  decoded {len(events)} CD events from {len(first_payload)} bytes "
              f"({len(first_payload) / 2:.0f} words)")

        if not events:
            print("  No events decoded — either the payload truly has none, or the "
                  "EVT3.0 hypothesis is wrong for this payload. Check the hex dump above.")
            return

        xs = [e[0] for e in events]
        ys = [e[1] for e in events]
        ts = [e[2] for e in events]
        ps = [e[3] for e in events]

        print(f"  x range: {min(xs)} .. {max(xs)}  (sensor width should be 1280)")
        print(f"  y range: {min(ys)} .. {max(ys)}  (sensor height should be 720)")
        print(f"  t range: {min(ts)} .. {max(ts)}  (12-bit low + 12-bit high timer units)")
        print(f"  polarity: {sum(ps)} ON / {len(ps) - sum(ps)} OFF")

        plausible = max(xs) < 1280 and max(ys) < 720
        print()
        if plausible:
            print("  ✓ x/y stay within sensor bounds — EVT3.0 decode looks PLAUSIBLE.")
        else:
            print("  ✗ x or y exceeded sensor bounds — EVT3.0 decode looks WRONG for this "
                  "payload (either EventFormatSize=Bpe64 changes the layout, or this isn't "
                  "standard EVT3.0 wire format). Send this output back for a next step.")

        print(f"\nFirst {min(args.max_events, len(events))} decoded events (x, y, t, polarity):")
        for e in events[:args.max_events]:
            print(f"  {e}")


if __name__ == "__main__":
    main()
