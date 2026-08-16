#!/usr/bin/env python3
"""
inspect_evt3.py — Read and sanity-check a CAROECT-D .cevt recording, and home
of decode_evt3() -- the shared word-level EVT3.0 decoder also imported by
raw_to_events.py / raw_to_video.py for the CURRENT .raw pipeline.

RENAMED from inspect_cevt.py. Two reasons, not just a cleanup:
  1. Despite the CLI here being .cevt-specific (container inspection), the
     decode_evt3() function is NOT .cevt-specific -- it decodes the standard
     Prophesee EVT3.0 word format, which shows up both inside .cevt payloads
     (legacy Arena path, see legacy/) AND directly in .raw files (current
     Metavision path). raw_to_events.py's pure-python fallback backend and
     raw_to_video.py both import decode_evt3 from this file. A name implying
     "only relevant to the retired .cevt path" would be actively misleading
     about a file the ACTIVE pipeline depends on.
  2. The name "inspect.py" (dropping "cevt" entirely, as literally requested)
     would shadow Python's own stdlib `inspect` module for every script run
     from this directory -- Python puts a script's own directory at the front
     of sys.path, so `import inspect` anywhere in the project would resolve
     to this file instead of the standard library. run_v2e.py and
     run_dvsvolt.py both call inspect.signature(...) as a core part of their
     constructor-kwarg-filtering design; that would break silently/confusingly
     project-wide. inspect_evt3.py keeps "inspect" as requested while naming
     what's actually being inspected (EVT3.0 words) instead of the container
     it happened to originally live inside.

WHAT .cevt IS
-------------
NOT a video/image file — a custom binary container written by evs_recorder.cpp.
Layout:
    FileHeader                     (once)
    RecordHeader + payload bytes   (repeated once per Arena buffer)
    ...

Two container versions exist and both are read here.

CAROEVT2 FileHeader (72 bytes, packed)  — current:
    char[8]  magic          "CAROEVT2"
    uint64   pixelFormat    (or payloadType if the buffer wasn't image-typed)
    uint32   bitsPerPixel
    uint32   width
    uint32   height
    uint32   recordHeaderSize
    double   acquisitionFrameRateHz    <- read off the camera at record time
    uint64   acquisitionFrameTimeUs    <- read off the camera at record time
    uint64   hostUnixEpochNsAtStart
    uint64   deviceTimestampAtStartNs
    uint32   flags
    uint32   reserved

CAROEVT2 RecordHeader (40 bytes, packed):
    uint64   frameId
    uint64   deviceTimestampNs   (0 if the camera gave none)
    uint64   hostRecvNs          (monotonic host arrival, always valid)
    uint64   payloadSize
    uint8    timestampSource     (0=none, 1=device, 2=host)
    uint8[7] reserved

CAROEVT1 (legacy): 36-byte FileHeader, 24-byte RecordHeader
    (frameId, timestampNs, payloadSize). timestampNs was 0 in every record ever
    produced on this camera, which is why V2 exists.

WHAT THIS SCRIPT DOES
---------------------
1. Parses the container losslessly (100% our own known format — safe).
2. Prints a hex dump of a chosen payload so you can SEE the raw bytes.
3. Runs the DENSE ACCUMULATED FRAME check (analyze_as_dense_frame). This is
   the confirmed real payload format for this camera: every payload observed
   in the most recent full recording was exactly width*height bytes
   (310/310 buffers at 921,600 bytes = 1280x720), baseline 128.
4. Also attempts a standard Prophesee EVT3.0 word decode, as a FALSIFICATION
   test rather than an expectation. AcquisitionAccumulationMode is
   firmware-locked (IsAvailable=false), so the camera cannot currently emit
   sparse events at all; if this decode ever starts producing sane in-bounds
   x/y on a fresh recording, that is real news and worth chasing. Treat a
   "plausible" verdict on a 921,600-byte dense frame as coincidence, not
   evidence — random bytes decode to in-bounds coordinates surprisingly often.

   NOTE: this decodes ONE record in isolation and resets time_low/time_high at
   the start of every call, so it says nothing about time continuity ACROSS
   records. For the real timestamp diagnostic, use:
     python legacy/cevt_to_events.py <file>.cevt --debug-time-continuity

Usage:
    python inspect_evt3.py test.cevt
    python inspect_evt3.py test.cevt --max-events 20     # print first N decoded events
    python inspect_evt3.py test.cevt --no-decode         # skip EVT3.0 decode, just inspect container
"""

import argparse
import struct
import sys
from pathlib import Path

# Container layouts. Kept byte-identical to evs_recorder.cpp's packed structs and
# to cevt_to_events.py — if any of the three drift apart, every record in every
# file misparses silently, so the sizes are asserted below rather than trusted.
FILE_HEADER_V1_FMT = "<8sQIIIQ"        # magic, pixelFormat, bpp, w, h, reserved
FILE_HEADER_V1_SIZE = struct.calcsize(FILE_HEADER_V1_FMT)
RECORD_HEADER_V1_FMT = "<QQQ"          # frameId, timestampNs, payloadSize
RECORD_HEADER_V1_SIZE = struct.calcsize(RECORD_HEADER_V1_FMT)

FILE_HEADER_V2_FMT = "<8sQIIIIdQQQII"
FILE_HEADER_V2_SIZE = struct.calcsize(FILE_HEADER_V2_FMT)
RECORD_HEADER_V2_FMT = "<QQQQB7x"      # frameId, deviceTs, hostRecv, size, source, pad
RECORD_HEADER_V2_SIZE = struct.calcsize(RECORD_HEADER_V2_FMT)

assert (FILE_HEADER_V1_SIZE, RECORD_HEADER_V1_SIZE) == (36, 24), "V1 layout drifted"
assert (FILE_HEADER_V2_SIZE, RECORD_HEADER_V2_SIZE) == (72, 40), "V2 layout drifted"

TS_SOURCE_NAME = {0: "none", 1: "device", 2: "host"}


def read_file_header(f):
    peek = f.read(8)
    f.seek(0)
    if len(peek) < 8:
        raise ValueError("File too short to contain a magic number. Is this really a .cevt file?")

    if peek == b"CAROEVT2":
        raw = f.read(FILE_HEADER_V2_SIZE)
        if len(raw) < FILE_HEADER_V2_SIZE:
            raise ValueError("File truncated inside the CAROEVT2 FileHeader.")
        (_m, pf, bpp, w, h, rec_size, fps_hz, frame_us,
         host_epoch, dev_ts0, flags, _r) = struct.unpack(FILE_HEADER_V2_FMT, raw)
        if rec_size != RECORD_HEADER_V2_SIZE:
            raise ValueError(
                f"File declares recordHeaderSize={rec_size} but this script understands "
                f"{RECORD_HEADER_V2_SIZE}. Written by a newer recorder — update this script.")
        return {
            "version": 2, "magic": "CAROEVT2",
            "pixel_format_or_payload_type": pf, "bits_per_pixel": bpp,
            "width": w, "height": h,
            "record_header_size": rec_size, "record_header_fmt": RECORD_HEADER_V2_FMT,
            "frame_rate_hz": fps_hz, "frame_time_us": frame_us,
            "host_epoch_ns": host_epoch, "device_ts_at_start_ns": dev_ts0, "flags": flags,
        }

    magic_str = peek.split(b"\x00")[0].decode("ascii", errors="replace")
    if magic_str != "CAROEVT1":
        print(f"  [warning] magic = {magic_str!r}, expected 'CAROEVT1' or 'CAROEVT2' — "
              "file may be corrupt or not written by evs_recorder.cpp.")
    raw = f.read(FILE_HEADER_V1_SIZE)
    if len(raw) < FILE_HEADER_V1_SIZE:
        raise ValueError(f"File too short for FileHeader (need {FILE_HEADER_V1_SIZE} bytes, "
                         f"got {len(raw)}). Is this really a .cevt file?")
    _m, pixel_format, bpp, width, height, _reserved = struct.unpack(FILE_HEADER_V1_FMT, raw)
    return {
        "version": 1, "magic": magic_str,
        "pixel_format_or_payload_type": pixel_format, "bits_per_pixel": bpp,
        "width": width, "height": height,
        "record_header_size": RECORD_HEADER_V1_SIZE, "record_header_fmt": RECORD_HEADER_V1_FMT,
        "frame_rate_hz": 0.0, "frame_time_us": 0,
        "host_epoch_ns": 0, "device_ts_at_start_ns": 0, "flags": 0,
    }


def iter_records(f, header):
    """Yields (frame_id, device_ts_ns, host_recv_ns, ts_source, payload) per record."""
    size = header["record_header_size"]
    fmt = header["record_header_fmt"]
    v2 = header["version"] >= 2

    while True:
        raw = f.read(size)
        if len(raw) == 0:
            return  # clean EOF
        if len(raw) < size:
            print(f"  [warning] truncated RecordHeader at end of file "
                  f"({len(raw)} of {size} bytes) — recording likely cut off mid-write.")
            return
        if v2:
            frame_id, dev_ns, host_ns, payload_size, ts_source = struct.unpack(fmt, raw)
        else:
            frame_id, dev_ns, payload_size = struct.unpack(fmt, raw)
            host_ns = 0
            ts_source = 1 if dev_ns > 0 else 0
        payload = f.read(payload_size)
        if len(payload) < payload_size:
            print(f"  [warning] truncated payload for frame {frame_id} "
                  f"(expected {payload_size}, got {len(payload)}) — recording likely cut off mid-write.")
            return
        yield frame_id, dev_ns, host_ns, ts_source, payload


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
        print(f"  magic                       = {header['magic']!r}  (container v{header['version']})")
        print(f"  pixelFormat / payloadType   = {header['pixel_format_or_payload_type']} "
              f"(0x{header['pixel_format_or_payload_type']:x})")
        print(f"  bitsPerPixel                = {header['bits_per_pixel']}")
        print(f"  width x height              = {header['width']} x {header['height']}")
        if header["version"] >= 2:
            print(f"  AcquisitionFrameRate        = {header['frame_rate_hz']} Hz")
            print(f"  AcquisitionFrameTime        = {header['frame_time_us']} us")
            print(f"  device Timestamp at start   = {header['device_ts_at_start_ns']} ns")
            print(f"  flags                       = 0b{header['flags']:03b} "
                  f"(bit0=TimestampReset ok, bit1=device ts available, bit2=frame rate known)")
        else:
            print("  [note] CAROEVT1 file: no host timestamps, no camera frame rate, no")
            print("         per-record timestamp source. Re-record with the current")
            print("         evs_recorder to get measured window times.")
        print()

        total_records = 0
        total_payload_bytes = 0
        selected_payload = None
        selected_frame_id = None
        first_n_sizes = []
        size_histogram = {}
        ts_source_counts = {}
        dev_min = dev_max = None
        host_min = host_max = None
        prev_frame_id = None
        frame_id_gaps = 0

        for frame_id, dev_ns, host_ns, ts_source, payload in iter_records(f, header):
            total_records += 1
            total_payload_bytes += len(payload)
            size_histogram[len(payload)] = size_histogram.get(len(payload), 0) + 1
            ts_source_counts[ts_source] = ts_source_counts.get(ts_source, 0) + 1

            # Only track ranges over records that actually carry that clock;
            # folding the 0s from unstamped records into the min would report a
            # bogus multi-second span.
            if dev_ns:
                dev_min = dev_ns if dev_min is None else min(dev_min, dev_ns)
                dev_max = dev_ns if dev_max is None else max(dev_max, dev_ns)
            if host_ns:
                host_min = host_ns if host_min is None else min(host_min, host_ns)
                host_max = host_ns if host_max is None else max(host_max, host_ns)

            if prev_frame_id is not None and frame_id > prev_frame_id + 1:
                frame_id_gaps += 1
            prev_frame_id = frame_id

            if total_records - 1 == args.record_index:
                selected_payload = payload
                selected_frame_id = frame_id
            if len(first_n_sizes) < 10:
                first_n_sizes.append(len(payload))

        print("Container summary:")
        print(f"  total records (Arena buffers) = {total_records}")
        print(f"  total payload bytes            = {total_payload_bytes:,}")
        if total_records > 0:
            print(f"  avg payload bytes/record        = {total_payload_bytes / total_records:.1f}")
        print(f"  payload sizes of first {len(first_n_sizes)} records = {first_n_sizes}")

        # A size histogram beats "avg bytes/record" for spotting the failure that
        # actually happens here: a handful of correct-size records mixed with a
        # majority of truncated ones. The average alone hides that completely.
        expected = args.sensor_width * args.sensor_height
        print(f"  distinct payload sizes          = {len(size_histogram)}")
        for sz, count in sorted(size_histogram.items(), key=lambda kv: -kv[1])[:5]:
            tag = "  <-- matches W*H (dense frame)" if sz == expected else ""
            print(f"      {sz:>10,} bytes x {count:>5} record(s){tag}")

        print(f"  timestamp sources               = "
              f"{ {TS_SOURCE_NAME.get(k, k): v for k, v in ts_source_counts.items()} }")
        if dev_min is not None:
            print(f"  device timestamp range (ns)     = {dev_min} .. {dev_max}  "
                  f"(span {(dev_max - dev_min) / 1e9:.3f} s)")
        else:
            print("  device timestamp range          = none (no record carried a device clock)")
        if host_min is not None:
            print(f"  host recv range (ns)            = {host_min} .. {host_max}  "
                  f"(span {(host_max - host_min) / 1e9:.3f} s)")
        if frame_id_gaps:
            print(f"  frame_id gaps                   = {frame_id_gaps}  "
                  f"(buffers lost between records — real time gaps, not reordering)")
        print()

        if selected_payload is None:
            print(f"Record index {args.record_index} not found (only {total_records} "
                  "records in file) — file has only a FileHeader, or index out of range.")
            return

        print(f"Selected record (index {args.record_index}): frameId={selected_frame_id}, "
              f"payload size={len(selected_payload)} bytes")
        print(f"First 64 bytes of this payload (hex + ascii):")
        print(hex_dump(selected_payload, 64))
        print()

        analyze_as_dense_frame(selected_payload, args.sensor_width, args.sensor_height, args.record_index)
        print()

        if args.no_decode:
            return

        print(f"Attempting EVT3.0 decode of record {args.record_index} payload (hypothesis test)...")
        events = decode_evt3(selected_payload, max_events=None)
        print(f"  decoded {len(events)} CD events from {len(selected_payload)} bytes "
              f"({len(selected_payload) / 2:.0f} words)")

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
