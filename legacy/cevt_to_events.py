#!/usr/bin/env python3
"""
cevt_to_events.py — Convert .cevt (CAROECT-D container) to events.h5 and/or EVT3.0

WHAT THE CAMERA ACTUALLY PRODUCES (established, do not re-litigate without new data)
------------------------------------------------------------------------------------
LUCID Triton2 EVS (TRT009S-E) delivers DENSE ACCUMULATED FRAMES over GigE, not a
sparse event stream. Each payload is exactly W*H bytes:

    pixel == 128  -> no event during this accumulation window
    pixel == 0    -> OFF event at this pixel
    pixel == 255  -> ON  event at this pixel

"EventFormat=EVT3_0" names the sensor<->FPGA protocol, not the GigE payload
format. AcquisitionAccumulationMode — the switch that would enable real sparse
output — is firmware-locked (IsAvailable=false) and cannot be opened through
GenICam. See evs_recorder.cpp's header for the full evidence.

THE TIMESTAMP PROBLEM, AND WHAT THIS SCRIPT WILL AND WILL NOT DO
----------------------------------------------------------------
A dense accumulated frame asserts only "this pixel fired somewhere inside this
window". Sub-window ordering is destroyed inside the camera. There is therefore
NO per-event microsecond timestamp to recover — not here, not anywhere. Any code
that emits one is inventing it.

The previous version of this script invented one anyway:

    t_us = round(n_frames * 1_000_000 / args.fps)      # fps defaulted to 30

Three separate defects lived in that single line:

  1. `--fps` was an operator-typed guess. Nothing verified it against the camera.
  2. `n_frames` was a dense counter, so a dropped or corrupt buffer silently
     COMPRESSED the timeline — every event after the gap was shifted earlier by
     one frame period, permanently and invisibly.
  3. The result was written to events.h5 indistinguishable from a measured
     value. calibrate_simulator.py has a guardrail for exactly this
     (`timestamp_precision_status`), but this script never wrote that attribute,
     so the guardrail read "unknown" and passed. Fabricated time flowed into
     physical calibration completely undetected.

This version fixes the information problem rather than improving the guess:

  * It prefers MEASURED time, in this order:
        device buffer clock  >  monotonic host arrival time  >  synthesis
    CAROEVT2 recordings carry the first two per record, plus an explicit
    per-record `timestampSource` tag written by the recorder.
  * Synthesis is the last resort, is driven by the camera's own
    AcquisitionFrameRate/AcquisitionFrameTime stored in the file header (not by
    `--fps`), and is indexed by frame_id DELTA so a dropped buffer leaves a real
    gap instead of compressing the timeline.
  * Every output file states what it contains:
        timestamp_precision_status   device_buffer | host_arrival | synthesized
        t_quantization_us            width of the accumulation window
        timestamp_zero_dt_fraction   fraction of consecutive events sharing a t
        decode_method_counts         dense / undersized / oversized / empty
    calibrate_simulator.py already reads all four. Timing calibration now
    refuses to run on frame-quantised data instead of silently accepting it.

All events inside one frame share that frame's window timestamp. That is not a
bug being papered over — it is the true information content of the data, and
`t_quantization_us` tells downstream code exactly how coarse it is.

CONTAINER FORMATS READ
----------------------
  CAROEVT2 (current)  FileHeader 72 B + per record RecordHeader 40 B + payload
  CAROEVT1 (legacy)   FileHeader 36 B + per record RecordHeader 24 B + payload
Old recordings keep working; they simply have no measured time and land in the
"synthesized" bucket, correctly labelled.

OUTPUT SCHEMAS
--------------
  events.h5  : root datasets x(uint16) y(uint16) t(uint64, us) p(uint8 0/1)
               — identical to run_v2e.py / run_dvsvolt.py, one shared reader.
  .raw.evt3  : Prophesee EVT3.0 16-bit word stream, readable by OpenEB /
               Metavision. Re-encoded FROM dense frames, so it inherits the
               same frame-quantised time. It is a compatibility shim, not a
               recovery of lost precision.

Usage:
  python cevt_to_events.py rec.cevt --output-h5 events.h5
  python cevt_to_events.py rec.cevt --debug-time-continuity     # inspect first
  python cevt_to_events.py rec.cevt --output-h5 e.h5 --output-evt e.raw.evt3
"""

import argparse
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# ── Container parsing ─────────────────────────────────────────────
MAGIC_V1 = b"CAROEVT1"
MAGIC_V2 = b"CAROEVT2"

FILE_HEADER_V1_FMT = "<8sQIIIQ"
FILE_HEADER_V1_SIZE = struct.calcsize(FILE_HEADER_V1_FMT)      # 36
RECORD_HEADER_V1_FMT = "<QQQ"
RECORD_HEADER_V1_SIZE = struct.calcsize(RECORD_HEADER_V1_FMT)  # 24

# magic, pixelFormat, bitsPerPixel, width, height, recordHeaderSize,
# acquisitionFrameRateHz, acquisitionFrameTimeUs, hostUnixEpochNsAtStart,
# deviceTimestampAtStartNs, flags, reserved
FILE_HEADER_V2_FMT = "<8sQIIIIdQQQII"
FILE_HEADER_V2_SIZE = struct.calcsize(FILE_HEADER_V2_FMT)      # 72
# frameId, deviceTimestampNs, hostRecvNs, payloadSize, timestampSource, pad[7]
RECORD_HEADER_V2_FMT = "<QQQQB7x"
RECORD_HEADER_V2_SIZE = struct.calcsize(RECORD_HEADER_V2_FMT)  # 40

assert FILE_HEADER_V1_SIZE == 36, "V1 struct layout changed"
assert RECORD_HEADER_V1_SIZE == 24, "V1 struct layout changed"
assert FILE_HEADER_V2_SIZE == 72, "V2 struct layout changed - keep in sync with evs_recorder.cpp"
assert RECORD_HEADER_V2_SIZE == 40, "V2 struct layout changed - keep in sync with evs_recorder.cpp"

TS_SOURCE_NONE, TS_SOURCE_DEVICE, TS_SOURCE_HOST = 0, 1, 2
TS_SOURCE_NAME = {TS_SOURCE_NONE: "none", TS_SOURCE_DEVICE: "device", TS_SOURCE_HOST: "host"}

# FileHeader.flags bits, mirroring evs_recorder.cpp
FLAG_TIMESTAMP_RESET_OK = 1 << 0
FLAG_DEVICE_TS_AVAILABLE = 1 << 1
FLAG_FRAME_RATE_KNOWN = 1 << 2

# Fallback geometry: this sensor, used only when the header says 0x0 (which
# happens when the first buffer could not be cast to IImage at capture time).
DEFAULT_W, DEFAULT_H = 1280, 720


def read_file_header(f):
    """Reads either container version. Returns a dict with a 'version' key."""
    peek = f.read(8)
    if len(peek) < 8:
        raise ValueError("File too short to contain a magic number - not a .cevt file?")
    f.seek(0)

    if peek == MAGIC_V2:
        raw = f.read(FILE_HEADER_V2_SIZE)
        if len(raw) < FILE_HEADER_V2_SIZE:
            raise ValueError("File truncated inside the CAROEVT2 FileHeader.")
        (_magic, pf, bpp, w, h, rec_hdr_size, fps_hz, frame_us,
         host_epoch_ns, dev_ts_start_ns, flags, _res) = struct.unpack(FILE_HEADER_V2_FMT, raw)

        if rec_hdr_size != RECORD_HEADER_V2_SIZE:
            # Self-describing header: a future recorder may grow RecordHeader.
            # Refuse rather than silently misparse every record in the file.
            raise ValueError(
                f"This file declares recordHeaderSize={rec_hdr_size}, but this script only "
                f"understands {RECORD_HEADER_V2_SIZE}. The recorder that wrote it is newer "
                f"than this converter - update cevt_to_events.py.")

        return {
            "version": 2,
            "width": w or DEFAULT_W,
            "height": h or DEFAULT_H,
            "geometry_from_header": bool(w and h),
            "pixelFormat": pf,
            "bpp": bpp,
            "record_header_size": rec_hdr_size,
            "record_header_fmt": RECORD_HEADER_V2_FMT,
            "frame_rate_hz": fps_hz if fps_hz > 0 else None,
            "frame_time_us": frame_us if frame_us > 0 else None,
            "host_epoch_ns_at_start": host_epoch_ns,
            "device_ts_at_start_ns": dev_ts_start_ns,
            "flags": flags,
        }

    if peek != MAGIC_V1:
        print(f"[warning] magic={peek!r}, expected {MAGIC_V1!r} or {MAGIC_V2!r}. "
              "Parsing as CAROEVT1 and hoping for the best.", file=sys.stderr)

    raw = f.read(FILE_HEADER_V1_SIZE)
    if len(raw) < FILE_HEADER_V1_SIZE:
        raise ValueError("File too short for a CAROEVT1 FileHeader - not a .cevt file?")
    _magic, pf, bpp, w, h, _res = struct.unpack(FILE_HEADER_V1_FMT, raw)
    return {
        "version": 1,
        "width": w or DEFAULT_W,
        "height": h or DEFAULT_H,
        "geometry_from_header": bool(w and h),
        "pixelFormat": pf,
        "bpp": bpp,
        "record_header_size": RECORD_HEADER_V1_SIZE,
        "record_header_fmt": RECORD_HEADER_V1_FMT,
        "frame_rate_hz": None,
        "frame_time_us": None,
        "host_epoch_ns_at_start": 0,
        "device_ts_at_start_ns": 0,
        "flags": 0,
    }


class Record:
    """One buffer, with whatever real time the recorder managed to capture."""
    __slots__ = ("frame_id", "device_ts_ns", "host_recv_ns", "ts_source", "payload")

    def __init__(self, frame_id, device_ts_ns, host_recv_ns, ts_source, payload):
        self.frame_id = frame_id
        self.device_ts_ns = device_ts_ns
        self.host_recv_ns = host_recv_ns
        self.ts_source = ts_source
        self.payload = payload


def iter_records(f, hdr):
    """Yields Record objects for every record, for either container version."""
    size = hdr["record_header_size"]
    fmt = hdr["record_header_fmt"]
    v2 = hdr["version"] >= 2

    while True:
        raw = f.read(size)
        if len(raw) == 0:
            return  # clean EOF
        if len(raw) < size:
            print(f"[warning] truncated RecordHeader at EOF ({len(raw)}/{size} bytes) - "
                  "recording was cut off mid-write.", file=sys.stderr)
            return

        if v2:
            frame_id, dev_ns, host_ns, payload_size, ts_source = struct.unpack(fmt, raw)
        else:
            frame_id, dev_ns, payload_size = struct.unpack(fmt, raw)
            host_ns = 0
            # V1 wrote 0 whenever HasImageData() was false. A literal 0 is not a
            # timestamp, so classify it as "no time" rather than "device time of 0".
            ts_source = TS_SOURCE_DEVICE if dev_ns > 0 else TS_SOURCE_NONE

        payload = f.read(payload_size)
        if len(payload) < payload_size:
            print(f"[warning] truncated payload for frame {frame_id} "
                  f"({len(payload)}/{payload_size} bytes) - recording cut off mid-write.",
                  file=sys.stderr)
            return
        yield Record(frame_id, dev_ns, host_ns, ts_source, payload)


# ── Timebase resolution ───────────────────────────────────────────
def resolve_timebase(hdr, records_meta, fps_override):
    """
    Decides, once for the whole file, where `t` comes from.

    Returns a dict describing the decision. Nothing downstream is allowed to
    invent time without going through here, which is what makes the resulting
    status attribute trustworthy.
    """
    sources = Counter(m["ts_source"] for m in records_meta)
    n = len(records_meta)

    # --- window length ------------------------------------------------------
    # Preference order is deliberate: AcquisitionFrameTime is an exact integer
    # period straight from the camera, AcquisitionFrameRate is a float that has
    # to be inverted, and --fps is a human guess of last resort.
    frame_us = None
    frame_us_source = None
    if hdr.get("frame_time_us"):
        frame_us = float(hdr["frame_time_us"])
        frame_us_source = "camera AcquisitionFrameTime"
    elif hdr.get("frame_rate_hz"):
        frame_us = 1e6 / float(hdr["frame_rate_hz"])
        frame_us_source = "camera AcquisitionFrameRate"
    elif fps_override:
        frame_us = 1e6 / float(fps_override)
        frame_us_source = f"--fps {fps_override} (OPERATOR GUESS, not measured)"

    # --- per-record time source --------------------------------------------
    if n and sources[TS_SOURCE_DEVICE] == n:
        status, field = "device_buffer", "device_ts_ns"
    elif n and (sources[TS_SOURCE_DEVICE] + sources[TS_SOURCE_HOST]) == n \
            and sources[TS_SOURCE_HOST] > 0:
        # Mixed device/host, or all host. Host arrival time is monotonic and
        # measured; mixing it with device time would splice two unrelated clocks,
        # so use host for every record and say so.
        status, field = "host_arrival", "host_recv_ns"
    else:
        status, field = "synthesized", None

    return {
        "status": status,
        "field": field,
        "frame_us": frame_us,
        "frame_us_source": frame_us_source,
        "source_counts": {TS_SOURCE_NAME[k]: v for k, v in sources.items()},
    }


def build_frame_times_us(records_meta, timebase):
    """
    Returns one window-start time in microseconds per record.

    Measured path: take the recorded clock and rebase it so the first record is
    t=0. Both device and host clocks are monotonic, so differences are real.

    Synthesis path: index by frame_id DELTA, never by position in the file. If
    buffers 5..9 were dropped, frame_id jumps by 6 and the synthesised timeline
    jumps by six windows — preserving the gap instead of pretending the
    recording was continuous. This is the specific defect that made the old
    `n_frames * 1e6/fps` line corrupt data rather than merely approximate it.
    """
    n = len(records_meta)
    if n == 0:
        return np.empty(0, np.int64)

    field = timebase["field"]
    if field is not None:
        raw = np.array([m[field] for m in records_meta], dtype=np.int64)
        t_us = (raw - raw[0]) // 1000  # ns -> us, rebased to the first record

        # Monotonicity is an invariant of both clocks; a violation means the file
        # is damaged or the records were reordered. Surface it, do not silently
        # sort it away, because sorting would hide the corruption.
        if np.any(np.diff(t_us) < 0):
            n_bad = int(np.sum(np.diff(t_us) < 0))
            print(f"[warning] {n_bad} record(s) have a timestamp EARLIER than their "
                  f"predecessor. The recorded clock is not monotonic - the file may be "
                  f"damaged. Timestamps are being used as-is, not re-sorted.",
                  file=sys.stderr)
        return t_us

    frame_us = timebase["frame_us"]
    if frame_us is None:
        raise ValueError(
            "This recording carries no measured timestamps AND the camera's frame rate "
            "is not in the file header, so window times cannot be established.\n"
            "  * CAROEVT2 recordings normally carry both. This is a CAROEVT1 file, or "
            "the camera's AcquisitionFrameRate/AcquisitionFrameTime were unreadable.\n"
            "  * Pass --fps <hz> to synthesise timestamps anyway. They will be labelled "
            "timestamp_precision_status='synthesized' and calibrate_simulator.py will "
            "refuse to use them for timing/Eq.23 calibration - which is correct, because "
            "they are a guess.\n"
            "  * Read the true rate off the camera with:\n"
            "        ./evs_recorder --node-info AcquisitionFrameRate")

    frame_ids = np.array([m["frame_id"] for m in records_meta], dtype=np.int64)
    idx = frame_ids - frame_ids[0]
    if np.any(np.diff(idx) <= 0):
        # Non-increasing frame ids would make the synthesised timeline fold back
        # on itself. Fall back to positional indexing and say so loudly.
        print("[warning] frame_id is not strictly increasing; falling back to positional "
              "indexing for synthesis. Dropped-buffer gaps will NOT be represented, so "
              "the timeline may be compressed.", file=sys.stderr)
        idx = np.arange(n, dtype=np.int64)

    n_gaps = int(np.sum(np.diff(idx) > 1))
    if n_gaps:
        print(f"[info] {n_gaps} frame_id gap(s) detected; synthesised timestamps preserve "
              f"them as real time gaps rather than compressing the timeline.")

    return np.round(idx * frame_us).astype(np.int64)


# ── Dense frame -> (x, y, p) ──────────────────────────────────────
def extract_events(payload, width, height):
    """
    Convert one dense accumulated frame to sparse (x, y, p) arrays.

    Returns (xs, ys, ps, method) where method is one of:
        "dense"      parsed normally
        "empty"      correct size, but no pixel differed from the 128 baseline
        "undersized" / "oversized"  payload size did not match W*H

    Separating "empty" from "wrong size" matters: the previous version returned
    empty arrays for both and counted them together as `n_skip`, so a recording
    in which EVERY payload was the wrong size looked exactly like a recording of
    a completely static scene. That is the same silent-data-loss failure that
    made cevt_to_h5.py unusable.
    """
    expected = width * height
    if len(payload) != expected:
        method = "undersized" if len(payload) < expected else "oversized"
        return (np.empty(0, np.uint16), np.empty(0, np.uint16),
                np.empty(0, np.uint8), method)

    frame = np.frombuffer(payload, dtype=np.uint8)

    # flatnonzero on a boolean mask, then divmod, is markedly faster than
    # np.argwhere on a reshaped array (no intermediate 2-column index matrix),
    # which matters at 921,600 pixels x thousands of frames.
    off_idx = np.flatnonzero(frame == 0)
    on_idx = np.flatnonzero(frame == 255)

    if off_idx.size == 0 and on_idx.size == 0:
        return (np.empty(0, np.uint16), np.empty(0, np.uint16),
                np.empty(0, np.uint8), "empty")

    flat = np.concatenate([off_idx, on_idx])
    ys, xs = np.divmod(flat, width)
    ps = np.concatenate([np.zeros(off_idx.size, np.uint8),
                         np.ones(on_idx.size, np.uint8)])
    return xs.astype(np.uint16), ys.astype(np.uint16), ps, "dense"


# ── EVT3.0 encoder ───────────────────────────────────────────────
# Standard Prophesee EVT3.0: 16-bit little-endian words, top 4 bits = type.
_EVT_ADDR_Y = 0x0   # bits 10:0 = y
_EVT_ADDR_X = 0x2   # bit 11 = polarity, bits 10:0 = x
_TIME_LOW = 0x6     # bits 11:0 = lower 12 bits of the 24-bit us timestamp
_TIME_HIGH = 0x8    # bits 11:0 = upper 12 bits of the 24-bit us timestamp

_EVT3_TIME_WRAP_US = 1 << 24  # ~16.777 s


def _word(wtype, value):
    return struct.pack("<H", ((wtype & 0xF) << 12) | (value & 0xFFF))


def encode_evt3(xs, ys, ps, t_us, state):
    """
    Encode one frame's events as an EVT3.0 word stream.

    `state` carries prev_th / prev_tl / prev_y ACROSS frames. The previous
    version created a fresh encoder per frame, which re-emitted TIME_HIGH and
    TIME_LOW for every single frame and, worse, reset prev_y — producing a
    stream whose decoder state did not match a real camera's. Threading the
    state through keeps the output a valid continuous EVT3.0 stream.

    Only ADDR_X is used (one word per event); VECT_12/VECT_8 compression is
    valid but changes nothing a decoder sees.
    """
    if len(xs) == 0:
        return b""

    buf = bytearray()

    # All events in a frame share t (see module docstring), so the time words are
    # emitted once per frame rather than once per event.
    t24 = int(t_us) % _EVT3_TIME_WRAP_US
    th = (t24 >> 12) & 0xFFF
    tl = t24 & 0xFFF

    if th != state["prev_th"]:
        buf += _word(_TIME_HIGH, th)
        state["prev_th"] = th
        state["prev_tl"] = -1  # TIME_LOW must be re-emitted after TIME_HIGH changes
    if tl != state["prev_tl"]:
        buf += _word(_TIME_LOW, tl)
        state["prev_tl"] = tl

    # Sort by (y, x) so ADDR_Y is emitted once per row, as a real sensor does.
    order = np.lexsort((xs, ys))
    for x, y, p in zip(xs[order].tolist(), ys[order].tolist(), ps[order].tolist()):
        if y != state["prev_y"]:
            buf += _word(_EVT_ADDR_Y, y & 0x7FF)
            state["prev_y"] = y
        buf += _word(_EVT_ADDR_X, ((p & 1) << 11) | (x & 0x7FF))

    return bytes(buf)


# ── Diagnostics ──────────────────────────────────────────────────
def debug_time_continuity(src):
    """
    Prints, per record, every time field the container carries, so the operator
    can see with their own eyes whether the clocks are continuous BEFORE any
    conversion happens. Referenced by inspect_cevt.py and run_pipeline.sh.
    """
    with open(src, "rb") as f:
        hdr = read_file_header(f)
        print(f"Container version : CAROEVT{hdr['version']}")
        print(f"Geometry          : {hdr['width']}x{hdr['height']}"
              f"{'' if hdr['geometry_from_header'] else '  (fallback - header said 0x0)'}")
        if hdr["version"] >= 2:
            print(f"AcquisitionFrameRate : {hdr['frame_rate_hz']} Hz")
            print(f"AcquisitionFrameTime : {hdr['frame_time_us']} us")
            print(f"TimestampReset ok    : {bool(hdr['flags'] & FLAG_TIMESTAMP_RESET_OK)}")
            print(f"Device ts available  : {bool(hdr['flags'] & FLAG_DEVICE_TS_AVAILABLE)}")
            print(f"Device ts at start   : {hdr['device_ts_at_start_ns']} ns")
        else:
            print("CAROEVT1 file: carries no host time, no frame rate, and no per-record\n"
                  "timestamp source tag. Re-record with the current evs_recorder to get\n"
                  "measured window times.")
        print()
        print(f"{'idx':>5} {'frameId':>9} {'source':>7} {'deviceTs(ns)':>16} "
              f"{'hostRecv(ns)':>15} {'d_dev(us)':>11} {'d_host(us)':>11} {'bytes':>9}")

        prev_dev = prev_host = None
        for i, rec in enumerate(iter_records(f, hdr)):
            d_dev = (rec.device_ts_ns - prev_dev) / 1e3 if prev_dev is not None else float("nan")
            d_host = (rec.host_recv_ns - prev_host) / 1e3 if prev_host is not None else float("nan")
            print(f"{i:>5} {rec.frame_id:>9} {TS_SOURCE_NAME[rec.ts_source]:>7} "
                  f"{rec.device_ts_ns:>16} {rec.host_recv_ns:>15} "
                  f"{d_dev:>11.1f} {d_host:>11.1f} {len(rec.payload):>9}")
            prev_dev, prev_host = rec.device_ts_ns, rec.host_recv_ns

    print("\nHow to read this:")
    print("  * source=device  -> deviceTs is a real camera clock; d_dev is real elapsed time.")
    print("  * source=host    -> only hostRecv is real. d_host includes network/scheduling")
    print("                      jitter, so it is an upper bound on timing accuracy.")
    print("  * source=none    -> CAROEVT1 file with no time at all; t must be synthesised.")
    print("  * A steady d_host with a jittery d_dev (or vice versa) tells you which clock")
    print("    to trust. A d_host that occasionally doubles means a dropped buffer -")
    print("    cross-check against a frameId jump on the same row.")


# ── Shared reader (for training/evaluation code) ──────────────────
READER_SNIPPET = '''
# Shared event reader - same function works for h5 from cevt_to_events.py,
# run_v2e.py and run_dvsvolt.py. Copy this wherever needed.

import h5py

def load_events_h5(path):
    """Returns dict with arrays x(uint16), y(uint16), t(uint64, us), p(uint8 0/1)."""
    with h5py.File(str(path), "r") as hf:
        return {k: hf[k][:] for k in ("x", "y", "t", "p")}
'''


# ── Main ──────────────────────────────────────────────────────────
def convert(args):
    src = Path(args.input)
    if not src.exists():
        raise FileNotFoundError(src)

    if args.debug_time_continuity:
        debug_time_continuity(src)
        return

    out_h5 = Path(args.output_h5) if args.output_h5 else None
    out_evt = Path(args.output_evt) if args.output_evt else None
    if out_h5 is None and out_evt is None:
        raise ValueError("Specify --output-h5 and/or --output-evt "
                         "(or --debug-time-continuity to inspect without converting)")

    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps must be > 0")

    h5py = None
    if out_h5 is not None:
        try:
            import h5py as _h5py
            h5py = _h5py
        except ImportError:
            raise ImportError("pip install h5py  (needed for --output-h5)")

    # ── Pass 1: read headers + payloads, decide the timebase ──────
    # Two passes are needed because the timebase decision depends on ALL records
    # (are they all device-stamped? are there frame_id gaps?), and a per-record
    # decision made on the fly is exactly how the old code ended up mixing
    # measured and invented time in one file without noticing.
    with open(src, "rb") as f:
        hdr = read_file_header(f)
        W, H = hdr["width"], hdr["height"]
        print(f"Source   : {src.name}")
        print(f"Container: CAROEVT{hdr['version']}")
        print(f"Geometry : {W}x{H}  pixelFormat=0x{hdr['pixelFormat']:x}"
              f"{'' if hdr['geometry_from_header'] else '  (fallback - header said 0x0)'}")
        records = list(iter_records(f, hdr))

    if not records:
        raise ValueError(f"{src} contains no records.")

    records_meta = [{"frame_id": r.frame_id, "device_ts_ns": r.device_ts_ns,
                     "host_recv_ns": r.host_recv_ns, "ts_source": r.ts_source}
                    for r in records]

    timebase = resolve_timebase(hdr, records_meta, args.fps)
    frame_t_us = build_frame_times_us(records_meta, timebase)

    # If the camera did not report its frame rate but we DO have measured
    # timestamps, the accumulation window can be measured from the data itself:
    # the median gap between consecutive records. Median, not mean, so a single
    # dropped-buffer gap does not inflate it. This keeps t_quantization_us
    # meaningful (rather than 0 = "unknown") on host_arrival recordings, which
    # matters because it is what tells downstream code how coarse t really is.
    if timebase["frame_us"] is None and len(frame_t_us) > 2:
        deltas = np.diff(frame_t_us.astype(np.int64))
        deltas = deltas[deltas > 0]
        if deltas.size:
            timebase["frame_us"] = float(np.median(deltas))
            timebase["frame_us_source"] = "measured median inter-record delta"

    print(f"Timebase : {timebase['status']}  (per-record sources: {timebase['source_counts']})")
    if timebase["frame_us"] is not None:
        print(f"Window   : {timebase['frame_us']:.1f} us  [{timebase['frame_us_source']}]")
    if timebase["status"] == "synthesized":
        print("[warning] No measured timestamps in this file. Timestamps are SYNTHESISED "
              "from frame_id x window length. calibrate_simulator.py will refuse to use "
              "them for timing/Eq.23 calibration.", file=sys.stderr)

    # ── Pass 2: decode payloads ───────────────────────────────────
    all_x, all_y, all_t, all_p = [], [], [], []
    evt3_buf = bytearray()
    evt3_state = {"prev_th": -1, "prev_tl": -1, "prev_y": -1}
    method_counts = Counter()
    n_events = 0

    for i, rec in enumerate(records):
        xs, ys, ps, method = extract_events(rec.payload, W, H)
        method_counts[method] += 1
        t_us = int(frame_t_us[i])

        if len(xs):
            all_x.append(xs)
            all_y.append(ys)
            all_t.append(np.full(len(xs), t_us, dtype=np.uint64))
            all_p.append(ps)
            if out_evt is not None:
                evt3_buf += encode_evt3(xs, ys, ps, t_us, evt3_state)
            n_events += len(xs)

        if i == 0 or (i + 1) % 30 == 0:
            print(f"  record {i + 1:>4} (id={rec.frame_id})  t={t_us / 1e6:.3f}s  "
                  f"ev={len(xs):>7,}  total={n_events:>9,}  [{method}]")

    n_bad = method_counts["undersized"] + method_counts["oversized"]
    print(f"\nProcessed {len(records)} records  |  {n_events:,} events")
    print(f"  decode methods: {dict(method_counts)}")

    if n_bad:
        # Loud, not a one-line "[skip]". A file where most payloads are the wrong
        # size means the recording is broken (ring truncation, wrong geometry),
        # and converting it anyway produces a plausible-looking, wrong dataset.
        pct = 100.0 * n_bad / len(records)
        print(f"[warning] {n_bad} of {len(records)} records ({pct:.1f}%) did NOT match "
              f"{W}x{H} = {W * H:,} bytes and contributed ZERO events.", file=sys.stderr)
        if pct > 50:
            raise ValueError(
                f"{pct:.1f}% of records have the wrong payload size. Refusing to write an "
                f"output file that would silently contain almost no data. Check the "
                f"recorder's 'truncated=' counter, and confirm geometry with:\n"
                f"    python ../inspect_evt3.py {src}   # renamed from inspect_cevt.py, "
                f"now lives in the project root, not legacy/")

    # Fraction of consecutive events sharing an identical timestamp. With
    # frame-quantised data this is near 1.0 by construction; calibrate_simulator
    # reads it as a second, independent signal that t has no sub-window meaning.
    if n_events > 1:
        T_all = np.concatenate(all_t)
        zero_dt_fraction = float(np.mean(np.diff(T_all.astype(np.int64)) == 0))
    else:
        T_all = np.concatenate(all_t) if all_t else np.empty(0, np.uint64)
        zero_dt_fraction = 0.0

    # ── Write HDF5 ────────────────────────────────────────────────
    if out_h5 is not None:
        if not all_x:
            print("[warning] no events extracted - HDF5 not written", file=sys.stderr)
        else:
            out_h5.parent.mkdir(parents=True, exist_ok=True)
            X = np.concatenate(all_x)
            Y = np.concatenate(all_y)
            P = np.concatenate(all_p)
            with h5py.File(str(out_h5), "w") as hf:
                hf.create_dataset("x", data=X, dtype=np.uint16, compression="gzip")
                hf.create_dataset("y", data=Y, dtype=np.uint16, compression="gzip")
                hf.create_dataset("t", data=T_all, dtype=np.uint64, compression="gzip")
                hf.create_dataset("p", data=P, dtype=np.uint8, compression="gzip")

                hf.attrs["n_events"] = len(X)
                hf.attrs["source"] = str(src)
                hf.attrs["width"] = W
                hf.attrs["height"] = H
                hf.attrs["t_unit"] = "microseconds"
                hf.attrs["format"] = "cevt_dense_accumulated"
                hf.attrs["container_version"] = hdr["version"]

                # ---- The contract calibrate_simulator.py actually reads ----
                # These four attributes are the whole point of this rewrite. The
                # previous version wrote none of them, so the downstream
                # guardrail silently defaulted to "unknown" and let fabricated
                # timestamps into physical calibration.
                hf.attrs["timestamp_precision_status"] = timebase["status"]
                hf.attrs["timestamp_zero_dt_fraction"] = zero_dt_fraction
                hf.attrs["decode_method_counts"] = str(dict(method_counts))
                hf.attrs["t_quantization_us"] = (
                    float(timebase["frame_us"]) if timebase["frame_us"] else 0.0)

                hf.attrs["timestamp_source_counts"] = str(timebase["source_counts"])
                hf.attrs["timebase_origin"] = timebase["frame_us_source"] or "unknown"
                hf.attrs["host_epoch_ns_at_start"] = hdr["host_epoch_ns_at_start"]
                hf.attrs["note"] = (
                    "Dense accumulated frames: every event within one frame shares that "
                    "frame's window timestamp. t has NO sub-window meaning - see "
                    "t_quantization_us. Do not use for per-event timing/jitter "
                    "calibration unless timestamp_precision_status is 'device_buffer'.")
            print(f"-> HDF5  : {out_h5}  |  {len(X):,} events  "
                  f"|  status={timebase['status']}")

    # ── Write EVT3.0 ──────────────────────────────────────────────
    if out_evt is not None:
        out_evt.parent.mkdir(parents=True, exist_ok=True)
        out_evt.write_bytes(bytes(evt3_buf))
        print(f"-> EVT3.0: {out_evt}  |  {len(evt3_buf):,} bytes "
              f"({len(evt3_buf) // 2:,} words)")
        if timebase["frame_us"] and (frame_t_us[-1] - frame_t_us[0]) >= _EVT3_TIME_WRAP_US:
            print("  [note] recording is longer than the EVT3.0 24-bit timestamp range "
                  "(~16.78 s), so the encoded time counter wraps. Real cameras wrap too "
                  "and OpenEB/Metavision unwrap it on read.")

    print("\nShared reader (paste into train/eval code):")
    print(READER_SNIPPET)
    print("Done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Convert .cevt (CAROECT-D) to events.h5 and/or EVT3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("input", help="Path to .cevt file")
    ap.add_argument("--output-h5", default=None,
                    help="Output path for events.h5 (HDF5, same schema as the simulators)")
    ap.add_argument("--output-evt", default=None,
                    help="Output path for EVT3.0 binary (readable by OpenEB/Metavision)")
    ap.add_argument("--fps", type=float, default=None,
                    help="LAST-RESORT accumulation frame rate in Hz, used ONLY when the "
                         "recording carries neither measured timestamps nor the camera's "
                         "own AcquisitionFrameRate/AcquisitionFrameTime. Timestamps built "
                         "from this are labelled 'synthesized' and are rejected by "
                         "calibrate_simulator.py for timing work. Read the real value with "
                         "`./evs_recorder --node-info AcquisitionFrameRate` instead of "
                         "guessing.")
    ap.add_argument("--debug-time-continuity", action="store_true",
                    help="Print every time field of every record and exit without "
                         "converting. Run this before trusting t for calibration.")

    # The failure modes here are operator-actionable (wrong geometry, missing
    # frame rate, a broken recording), and their messages are written to be read.
    # A raw Python traceback buries that message under a stack dump, so catch the
    # expected exception types and exit cleanly with a non-zero code that
    # run_pipeline.sh's `set -e` can act on.
    try:
        convert(ap.parse_args())
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
