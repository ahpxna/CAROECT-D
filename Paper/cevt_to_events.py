#!/usr/bin/env python3
"""
cevt_to_events.py — Convert .cevt (CAROECT-D custom container) to events.h5 and/or EVT3.0

.cevt FORMAT (written by evs_recorder.cpp):
  FileHeader 36 bytes  : magic "CAROEVT1" | pixelFormat | bitsPerPixel | W | H | reserved
  Per-buffer           : RecordHeader 24 bytes (frameId, timestampNs, payloadSize) + payload

PAYLOAD — 3 hypotheses tried IN ORDER, per record (not globally: mixed recordings are
possible if the camera's decode setting changed mid-session):

  1. XYPT — CONFIRMED DEAD for this camera/firmware (TRT009S-E). evs_recorder.cpp
     used to request an "EvsOutputFormat=XYPT" node before recording; that node does
     not exist on this device's node map (confirmed via --dump-nodes --filter
     Output/Frame/Mode/XY, all negative — see evs_recorder.cpp comments) and the
     request was removed from evs_recorder.cpp entirely. try_decode_xypt() is kept
     here ONLY in case a future/different camera unit does support it — on THIS
     camera it will simply never match and every record falls through to #2/#3
     below. Accepted when payload_size % 16 == 0 AND >= XYPT_MIN_INBOUNDS_FRAC of
     decoded (x,y) fall inside (0..W, 0..H) — see try_decode_xypt().

  2. DENSE ACCUMULATED FRAME (old default, confirmed empirically pre-XYPT) — each
     payload = W×H uint8 frame where pixel=128 no event, pixel=0 OFF, pixel=255 ON.
     Accepted when payload_size == W*H exactly. Per-event timestamp does NOT exist in
     this format — every event in the frame shares one t reconstructed from
     frame_index / --fps (a real limitation: inter-event timing is fabricated, only
     usable for rate-based calibration, not timing-distribution calibration — see
     calibrate_simulator.py's module docstring).

  3. Standard Prophesee EVT3.0 16-bit word stream — fallback if neither of the above
     matches. "EventFormat=EVT3_0" is the sensor↔FPGA protocol name; this hypothesis
     tests whether Arena SDK passed that bitstream through verbatim instead of
     decoding it.

Pass --legacy-dense to skip the XYPT attempt entirely and force old behavior (dense
frame first, EVT3.0 fallback) — mainly useful for old recordings, since XYPT itself
is now confirmed unreachable on this camera regardless.

BEFORE TRUSTING t FOR CALIBRATION: run with --debug-time-continuity first. It has
not yet been empirically confirmed whether EVT3.0 TIME_LOW/TIME_HIGH resets to near
0 at the start of every record, or is one continuous counter that spans records
(decode_record()/try_decode_evt3_words() currently ASSUME reset-per-record). If the
diagnostic shows it's actually continuous, re-run the real conversion with
--evt3-continuous-time so absolute timestamps across records are correct.

OUTPUT SCHEMAS:
  events.h5  : datasets x(uint16) y(uint16) t(uint64,µs) p(uint8 0/1) — same as
               run_v2e.py / run_dvsvolt.py / read_evt3.py output. One shared reader.
               attrs["decode_method"] records which hypothesis won, per-file summary
               (mixed-method files are flagged).
  .raw.evt3  : standard Prophesee EVT3.0 16-bit word stream (little-endian),
               readable by Metavision SDK / OpenEB / our inspect_cevt.py decoder.

Usage:
  python cevt_to_events.py recording.cevt --debug-time-continuity   # do this FIRST
  python cevt_to_events.py recording.cevt --output-h5 events.h5
  python cevt_to_events.py recording.cevt --output-evt events.raw.evt3
  python cevt_to_events.py recording.cevt --output-h5 events.h5 --output-evt events.raw.evt3 --fps 30
  python cevt_to_events.py recording.cevt --output-h5 events.h5 --evt3-continuous-time
  python cevt_to_events.py recording.cevt --output-h5 events.h5 --legacy-dense   # force old behavior
"""

import struct
import argparse
import sys
import numpy as np
from pathlib import Path

# ── Container parsing ─────────────────────────────────────────────
FILE_HEADER_FMT  = "<8sQIII Q"
FILE_HEADER_SIZE = struct.calcsize(FILE_HEADER_FMT)  # 36
RECORD_HEADER_FMT  = "<QQQ"
RECORD_HEADER_SIZE = struct.calcsize(RECORD_HEADER_FMT)  # 24

assert FILE_HEADER_SIZE == 36, "struct layout changed"
assert RECORD_HEADER_SIZE == 24, "struct layout changed"


def read_file_header(f):
    raw = f.read(FILE_HEADER_SIZE)
    if len(raw) < FILE_HEADER_SIZE:
        raise ValueError("File too short for FileHeader — not a .cevt file?")
    magic, pf, bpp, w, h, _ = struct.unpack(FILE_HEADER_FMT, raw)
    magic_str = magic.split(b"\x00")[0].decode("ascii", errors="replace")
    if magic_str != "CAROEVT1":
        print(f"[warning] magic={magic_str!r}, expected 'CAROEVT1'", file=sys.stderr)
    # FileHeader had 0×0 geometry when HasImageData()==false at capture time;
    # fall back to the known Triton2 EVS / IMX636 sensor resolution.
    return {
        "width":       w or 1280,
        "height":      h or 720,
        "pixelFormat": pf,
        "bpp":         bpp,
    }


def iter_records(f):
    """Yield (frame_id, ts_ns, payload_bytes) for every record."""
    while True:
        raw = f.read(RECORD_HEADER_SIZE)
        if len(raw) == 0:
            return
        if len(raw) < RECORD_HEADER_SIZE:
            print("[warning] truncated RecordHeader at EOF — recording may be cut off.",
                  file=sys.stderr)
            return
        frame_id, ts_ns, payload_size = struct.unpack(RECORD_HEADER_FMT, raw)
        payload = f.read(payload_size)
        if len(payload) < payload_size:
            print(f"[warning] truncated payload for frame {frame_id}.", file=sys.stderr)
            return
        yield frame_id, ts_ns, payload


# ── Hypothesis 1: XYPT (NEW DEFAULT) — sparse, REAL per-event timestamp ──
# struct LucidXYTPPixel { float x, y, t, p; }  — 16 bytes/event, packed.
# See module docstring for why this is tried first and how it's accepted/rejected.
XYPT_MIN_INBOUNDS_FRAC = 0.90  # below this, reject the hypothesis (likely not XYPT)


def try_decode_xypt(payload: bytes, width: int, height: int):
    """Returns a dict (x,y,t,p as numpy arrays + plausible_frac) if the payload
    plausibly decodes as XYPT, else None. Never raises — a wrong guess here
    must fall through to the next hypothesis, not crash the whole conversion."""
    if len(payload) == 0 or len(payload) % 16 != 0:
        return None
    n = len(payload) // 16
    try:
        arr = np.frombuffer(payload, dtype=np.float32).reshape(n, 4)
    except ValueError:
        return None
    x, y, t, p = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height) & np.isfinite(t)
    plausible_frac = float(np.mean(in_bounds)) if n else 0.0
    if plausible_frac < XYPT_MIN_INBOUNDS_FRAC:
        return None
    xs = np.clip(np.round(x[in_bounds]), 0, width - 1).astype(np.uint16)
    ys = np.clip(np.round(y[in_bounds]), 0, height - 1).astype(np.uint16)
    ps = (p[in_bounds] > 0).astype(np.uint8)
    # NOTE unit of t not yet empirically confirmed — treated as microseconds
    # (LUCID's own convention for event timestamps). If downstream durations
    # look wrong by a constant factor (1000x / 1e6x), this is the first place
    # to check — see calibrate_simulator.py's sanity checks.
    ts = t[in_bounds]
    return dict(x=xs, y=ys, t=ts, p=ps, plausible_frac=plausible_frac, n=n)


# ── Hypothesis 3 fallback: standard Prophesee EVT3.0 word stream ─────────
# Same decoder as inspect_cevt.py / cevt_to_video.py — kept in sync manually
# (small enough that a shared-import module would be more overhead than value
# right now; if it drifts, inspect_cevt.py is the reference implementation).
_EVT3_ADDR_Y, _EVT3_ADDR_X = 0x0, 0x2
_EVT3_VECT_BASE_X, _EVT3_VECT_12, _EVT3_VECT_8 = 0x3, 0x4, 0x5
_EVT3_TIME_LOW, _EVT3_TIME_HIGH = 0x6, 0x8


def try_decode_evt3_words(payload: bytes, width: int, height: int, t_us_base: int,
                           time_state: dict = None):
    """Decode payload as standard EVT3.0 words. Returns dict like
    try_decode_xypt, or None if it doesn't look plausible (x/y out of bounds
    or zero events decoded).

    time_state: optional dict with keys "time_low"/"time_high" carried IN and
    updated OUT by the caller. Pass the SAME dict across consecutive records
    (see --evt3-continuous-time) to test/use the hypothesis that TIME_LOW/
    TIME_HIGH continue from one record to the next rather than resetting —
    see the module docstring / Stage 2 diagnostic note. When None (default),
    every record starts from time_low=time_high=0, exactly as before.

    Also returns "raw_first_time": the undecorated 24-bit
    (time_high<<12)|time_low value of the FIRST event decoded in this
    record, BEFORE t_us_base is added — this is what the continuity
    diagnostic (--debug-time-continuity) prints across records. If it stays
    near 0 at the start of every record, TIME_LOW/TIME_HIGH reset per
    record (current assumption, safe). If it keeps climbing smoothly from
    one record's last value into the next record's first value, they are
    actually one continuous counter and decode_record()/this function need
    time_state threaded through for correct absolute timestamps.
    """
    if len(payload) < 2:
        return None
    n_words = len(payload) // 2
    events = []
    cur_y = cur_p = base_x = 0
    if time_state is not None:
        time_low = time_state.get("time_low", 0)
        time_high = time_state.get("time_high", 0)
    else:
        time_low = time_high = 0
    raw_first_time = None
    for i in range(n_words):
        word = struct.unpack_from("<H", payload, i * 2)[0]
        wtype, value = (word >> 12) & 0xF, word & 0x0FFF
        if wtype == _EVT3_ADDR_Y:
            cur_y = value & 0x7FF
        elif wtype == _EVT3_ADDR_X:
            cur_p = (value >> 11) & 0x1
            x = value & 0x7FF
            t_raw = (time_high << 12) | time_low
            if raw_first_time is None:
                raw_first_time = t_raw
            events.append((x, cur_y, t_raw, cur_p))
        elif wtype == _EVT3_VECT_BASE_X:
            cur_p = (value >> 11) & 0x1
            base_x = value & 0x7FF
        elif wtype == _EVT3_VECT_12:
            for bit in range(12):
                if value & (1 << bit):
                    t_raw = (time_high << 12) | time_low
                    if raw_first_time is None:
                        raw_first_time = t_raw
                    events.append((base_x + bit, cur_y, t_raw, cur_p))
            base_x += 12
        elif wtype == _EVT3_VECT_8:
            for bit in range(8):
                if value & (1 << bit):
                    t_raw = (time_high << 12) | time_low
                    if raw_first_time is None:
                        raw_first_time = t_raw
                    events.append((base_x + bit, cur_y, t_raw, cur_p))
            base_x += 8
        elif wtype == _EVT3_TIME_LOW:
            time_low = value
        elif wtype == _EVT3_TIME_HIGH:
            time_high = value

    if time_state is not None:
        time_state["time_low"] = time_low
        time_state["time_high"] = time_high

    if not events:
        return None
    arr = np.asarray(events)
    xs, ys, ts, ps = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    if xs.max() >= width or ys.max() >= height:
        return None
    return dict(x=xs.astype(np.uint16), y=ys.astype(np.uint16),
                t=(ts.astype(np.int64) + t_us_base).astype(np.float64),
                p=ps.astype(np.uint8), plausible_frac=1.0, n=len(events),
                raw_first_time=raw_first_time)


# ── Stage 2 diagnostic: does TIME_LOW/TIME_HIGH reset every record, or is ──
# ── it one continuous counter across records? ───────────────────────────
def scan_time_continuity(path: str, width: int, height: int, max_records: int = None):
    """Print the raw (undecorated, pre-t_us_base) EVT3.0 timestamp of the
    first event in each record, for consecutive records, under BOTH
    assumptions side by side:
      reset_first   - time_low/time_high start at 0 every record (current
                       default behavior of try_decode_evt3_words / decode_record)
      running_first - time_low/time_high are threaded through as one counter
                       across every record (via a persistent time_state dict)

    How to read the output:
      - If reset_first stays small (near 0) at the start of every record,
        while running_first keeps climbing record-over-record roughly in
        step with elapsed wall time -> TIME_LOW/TIME_HIGH really do reset
        per record, and the current per-record-reset decoding is correct.
      - If reset_first ALSO climbs smoothly from one record into the next
        (i.e. record N's first raw time picks up close to where record N-1
        left off, not near 0) -> the counter is continuous across records,
        and decode_record()/convert() need to thread a persistent
        time_state through (pass --evt3-continuous-time) instead of
        resetting per record, or every record's absolute time will be off.
    """
    with open(path, "rb") as f:
        read_file_header(f)
        running_state = {"time_low": 0, "time_high": 0}
        n = 0
        print(f"{'record':>6}  {'reset_first':>12}  {'running_first':>14}  {'n_events':>8}")
        for frame_id, ts_ns, payload in iter_records(f):
            if max_records is not None and n >= max_records:
                break
            reset_result = try_decode_evt3_words(payload, width, height, 0, time_state=None)
            running_result = try_decode_evt3_words(payload, width, height, 0, time_state=running_state)
            reset_first = reset_result["raw_first_time"] if reset_result else None
            running_first = running_result["raw_first_time"] if running_result else None
            n_events = running_result["n"] if running_result else 0
            print(f"{n:>6}  {str(reset_first):>12}  {str(running_first):>14}  {n_events:>8}")
            n += 1
    print(f"\n{n} records scanned. See docstring of scan_time_continuity() for how to "
          "interpret reset_first vs running_first.")


# ── Hypothesis 2: dense accumulated frame (old default) → (x, y, t, p) ───
def extract_events_dense(payload: bytes, width: int, height: int, t_us: int):
    """
    Convert one dense accumulated frame to sparse (x,y,t,p) arrays.
    Returns 4 numpy arrays (empty if wrong size or no events).
    """
    if len(payload) != width * height:
        return (np.empty(0, np.uint16), np.empty(0, np.uint16),
                np.empty(0, np.uint64), np.empty(0, np.uint8))

    frame = np.frombuffer(payload, dtype=np.uint8).reshape(height, width)

    off_rc = np.argwhere(frame == 0)     # row,col where OFF happened
    on_rc  = np.argwhere(frame == 255)   # row,col where ON  happened

    if off_rc.size == 0 and on_rc.size == 0:
        return (np.empty(0, np.uint16), np.empty(0, np.uint16),
                np.empty(0, np.uint64), np.empty(0, np.uint8))

    xs = np.concatenate([off_rc[:, 1], on_rc[:, 1]]).astype(np.uint16)
    ys = np.concatenate([off_rc[:, 0], on_rc[:, 0]]).astype(np.uint16)
    ts = np.full(len(xs), t_us, dtype=np.uint64)
    ps = np.concatenate([np.zeros(len(off_rc), np.uint8),
                          np.ones(len(on_rc),  np.uint8)])
    return xs, ys, ts, ps


def decode_record(payload: bytes, width: int, height: int, frame_idx: int,
                   fps: float, legacy_dense: bool, evt3_time_state: dict = None):
    """Dispatch across the 3 hypotheses, in priority order (see module
    docstring). Returns (xs, ys, ts, ps, method_name).

    evt3_time_state: pass a persistent dict (shared across every call in one
    conversion) to thread EVT3.0 TIME_LOW/TIME_HIGH across records instead
    of resetting each time — only meaningful once --debug-time-continuity
    has confirmed the counter is continuous; see scan_time_continuity()."""
    t_us_fallback = int(round(frame_idx * 1_000_000.0 / fps))

    if not legacy_dense:
        xypt = try_decode_xypt(payload, width, height)
        if xypt is not None:
            return (xypt["x"], xypt["y"], xypt["t"].astype(np.uint64), xypt["p"], "xypt")

    xs, ys, ts, ps = extract_events_dense(payload, width, height, t_us_fallback)
    if len(xs):
        return xs, ys, ts, ps, "dense"

    if len(payload) == width * height:
        # Correctly-sized dense frame that happened to have 0 events this buffer.
        return xs, ys, ts, ps, "dense-empty"

    evt3 = try_decode_evt3_words(payload, width, height, t_us_fallback, time_state=evt3_time_state)
    if evt3 is not None:
        return (evt3["x"], evt3["y"], evt3["t"].astype(np.uint64), evt3["p"], "evt3-words")

    return (np.empty(0, np.uint16), np.empty(0, np.uint16),
            np.empty(0, np.uint64), np.empty(0, np.uint8), "unrecognized")


# ── EVT3.0 encoder ───────────────────────────────────────────────
# Standard Prophesee EVT3.0: 16-bit little-endian words, top 4 bits = type.
# Reference: Prophesee metavision_sdk/modules/core/include/metavision/sdk/
#            core/utils/event_traits.h (public, in OpenEB).

_EVT_ADDR_Y  = 0x0   # bits 11:0 = y (11 bits)
_EVT_ADDR_X  = 0x2   # bit  11   = polarity, bits 10:0 = x
_TIME_LOW    = 0x6   # bits 11:0 = lower 12 bits of 24-bit µs timestamp
_TIME_HIGH   = 0x8   # bits 11:0 = upper 12 bits of 24-bit µs timestamp


def _word(wtype: int, value: int) -> bytes:
    return struct.pack("<H", ((wtype & 0xF) << 12) | (value & 0xFFF))


def encode_evt3(xs: np.ndarray, ys: np.ndarray,
                ts: np.ndarray, ps: np.ndarray) -> bytes:
    """
    Encode (x,y,t,p) arrays to a Prophesee EVT3.0 binary word stream.

    Design: sort by t→y→x (scan-line order), emit TIME_HIGH / TIME_LOW
    words on change, ADDR_Y on row change, ADDR_X per event.
    We use only ADDR_X (one word per event) for correctness and simplicity —
    VECT_12/VECT_8 compression is valid but requires all events at a given
    y/t/p to be sorted and adjacent in x, which adds complexity without
    changing the readable output for our use case.

    Timestamp: EVT3.0 uses 24-bit microsecond counter split into two 12-bit
    halves (HIGH upper, LOW lower). Since our t values can exceed 2^24 µs
    (~16 s), we wrap modulo 2^24 — this matches real camera behavior where
    the timestamp counter wraps and downstream tools handle the rollover.
    """
    if len(xs) == 0:
        return b""

    buf = bytearray()
    order = np.lexsort((xs, ys, ts))
    xs, ys, ts, ps = xs[order], ys[order], ts[order], ps[order]

    prev_th, prev_tl, prev_y = -1, -1, -1

    for x, y, t, p in zip(xs.tolist(), ys.tolist(), ts.tolist(), ps.tolist()):
        t24 = t & 0xFFFFFF  # 24-bit wraparound, consistent with real cameras
        th  = (t24 >> 12) & 0xFFF
        tl  =  t24        & 0xFFF

        if th != prev_th:
            buf += _word(_TIME_HIGH, th)
            prev_th = th
            prev_tl = -1   # force TIME_LOW re-emit after HIGH changes

        if tl != prev_tl:
            buf += _word(_TIME_LOW, tl)
            prev_tl = tl

        if y != prev_y:
            buf += _word(_EVT_ADDR_Y, y & 0x7FF)
            prev_y = y

        buf += _word(_EVT_ADDR_X, ((p & 1) << 11) | (x & 0x7FF))

    return bytes(buf)


# ── Shared reader (for training/evaluation code) ──────────────────
READER_SNIPPET = '''
# ── Shared event reader — same function works for h5 from cevt_to_events.py,
#    run_v2e.py, run_dvsvolt.py, and read_evt3.py.  Copy this wherever needed.

import h5py, numpy as np

def load_events_h5(path):
    """Returns dict with arrays x(uint16), y(uint16), t(uint64,µs), p(uint8 0/1)."""
    with h5py.File(str(path), "r") as hf:
        return {k: hf[k][:] for k in ("x", "y", "t", "p")}
'''


# ── Main ──────────────────────────────────────────────────────────
def convert(args):
    src = Path(args.input)
    if not src.exists():
        raise FileNotFoundError(src)

    out_h5  = Path(args.output_h5)  if args.output_h5  else None
    out_evt = Path(args.output_evt) if args.output_evt else None

    if out_h5 is None and out_evt is None:
        raise ValueError("Specify at least --output-h5 and/or --output-evt")

    # Defer h5py import — only needed when writing HDF5
    if out_h5 is not None:
        try:
            import h5py
        except ImportError:
            raise ImportError("pip install h5py  (needed for --output-h5)")

    with open(src, "rb") as f:
        hdr = read_file_header(f)
        W, H = hdr["width"], hdr["height"]
        print(f"Source : {src.name}")
        print(f"Geometry: {W}×{H}  pixelFormat=0x{hdr['pixelFormat']:x}")

        # Collect across all frames
        all_x, all_y, all_t, all_p = [], [], [], []
        evt3_buf  = bytearray()
        n_frames  = 0
        n_events  = 0
        n_skip    = 0
        method_counts = {}
        evt3_time_state = {"time_low": 0, "time_high": 0} if args.evt3_continuous_time else None
        if args.evt3_continuous_time:
            print("[info] --evt3-continuous-time: threading TIME_LOW/TIME_HIGH across "
                  "records instead of resetting per record.")

        for frame_id, ts_ns, payload in iter_records(f):
            xs, ys, ts, ps, method = decode_record(payload, W, H, n_frames, args.fps,
                                                    args.legacy_dense, evt3_time_state)
            method_counts[method] = method_counts.get(method, 0) + 1

            if len(xs):
                all_x.append(xs); all_y.append(ys)
                all_t.append(ts); all_p.append(ps)
                if out_evt is not None:
                    evt3_buf += encode_evt3(xs, ys, ts, ps)
                n_events += len(xs)
            else:
                n_skip += 1

            n_frames += 1
            if n_frames % 30 == 0 or n_frames == 1:
                t_show = (ts[0] / 1e6) if len(ts) else (n_frames * 1_000_000.0 / args.fps) / 1e6
                print(f"  frame {n_frames:>4} (id={frame_id})  method={method:<12}  "
                      f"t~{t_show:.3f}s  ev={len(xs):>7,}  total={n_events:>9,}")

    print(f"\nProcessed {n_frames} frames  |  "
          f"{n_events:,} events  |  {n_skip} empty frames")
    print(f"Decode method breakdown: {method_counts}")
    if len(method_counts) > 1:
        print("[warning] more than one decode method matched across this file — mixed "
              "recording, or the hypothesis boundary is fuzzy. Check the breakdown above; "
              "'xypt' winning consistently is the good case (real timestamps). If 'dense' "
              "dominates, XYPT was likely not active for this recording (check evs_recorder "
              "console output for '[output-format]' at capture time).")
    primary_method = max(method_counts, key=method_counts.get) if method_counts else "none"

    # ── Write HDF5 ──────────────────────────────────────────────
    if out_h5 is not None:
        out_h5.parent.mkdir(parents=True, exist_ok=True)
        if all_x:
            X = np.concatenate(all_x)
            Y = np.concatenate(all_y)
            T = np.concatenate(all_t)
            P = np.concatenate(all_p)
            with h5py.File(str(out_h5), "w") as hf:
                hf.create_dataset("x", data=X, dtype=np.uint16, compression="gzip")
                hf.create_dataset("y", data=Y, dtype=np.uint16, compression="gzip")
                hf.create_dataset("t", data=T, dtype=np.uint64, compression="gzip")
                hf.create_dataset("p", data=P, dtype=np.uint8,  compression="gzip")
                hf.attrs["n_events"]    = len(X)
                hf.attrs["source"]      = str(src)
                hf.attrs["width"]       = W
                hf.attrs["height"]      = H
                hf.attrs["fps"]         = args.fps
                hf.attrs["t_unit"]      = "microseconds"
                hf.attrs["format"]      = f"cevt_{primary_method}"
                hf.attrs["decode_method_counts"] = str(method_counts)
                hf.attrs["note"]        = (
                    "t is REAL per-event microsecond timestamp if decode_method=xypt; "
                    "reconstructed from frame_index/fps (fabricated, rate-only-accurate) "
                    "if decode_method=dense. Schema matches run_v2e.py / run_dvsvolt.py / "
                    "read_evt3.py output.")
            print(f"→ HDF5  : {out_h5}  |  {len(X):,} events")
        else:
            print("[warning] no events extracted — HDF5 not written", file=sys.stderr)

    # ── Write EVT3.0 ────────────────────────────────────────────
    if out_evt is not None:
        out_evt.parent.mkdir(parents=True, exist_ok=True)
        out_evt.write_bytes(bytes(evt3_buf))
        print(f"→ EVT3.0: {out_evt}  |  "
              f"{len(evt3_buf):,} bytes ({len(evt3_buf)//2:,} words)")

    print("\nShared reader (paste into train/eval code):")
    print(READER_SNIPPET)
    print("✓ Done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Convert .cevt (CAROECT-D) to events.h5 and/or EVT3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("input",          help="Path to .cevt file")
    ap.add_argument("--output-h5",    default=None,
                    help="Output path for events.h5  (HDF5, same schema as simulators)")
    ap.add_argument("--output-evt",   default=None,
                    help="Output path for EVT3.0 binary (readable by OpenEB/Metavision)")
    ap.add_argument("--fps",          type=float, default=30.0,
                    help="Accumulation frame rate in Hz — used to reconstruct timestamps "
                         "from frame index when a record is NOT decoded as XYPT (default "
                         "30). Check ArenaView / camera node AcquisitionFrameRate for the "
                         "true value.")
    ap.add_argument("--legacy-dense", action="store_true",
                    help="Skip the XYPT hypothesis entirely; assume dense-CD-frame first "
                         "(old default behavior — kept for old recordings made before XYPT "
                         "support existed; XYPT itself is now confirmed DEAD for this "
                         "camera/firmware, see evs_recorder.cpp comments, so this mainly "
                         "matters for re-decoding old files)")
    ap.add_argument("--debug-time-continuity", action="store_true",
                    help="Diagnostic only (Stage 2): print the raw EVT3.0 TIME_LOW/TIME_HIGH "
                         "of the first event of each record, under both a per-record-reset "
                         "and a carried-across-records assumption, then exit without writing "
                         "any output. Use this BEFORE trusting t for calibration — see "
                         "scan_time_continuity() docstring for how to read the result.")
    ap.add_argument("--evt3-continuous-time", action="store_true",
                    help="If --debug-time-continuity confirms TIME_LOW/TIME_HIGH is one "
                         "continuous counter across records (not reset per record), pass "
                         "this to thread that state through the real conversion too, "
                         "instead of resetting time_low/time_high to 0 at the start of "
                         "every record.")
    args = ap.parse_args()
    if args.debug_time_continuity:
        with open(args.input, "rb") as f:
            hdr = read_file_header(f)
        scan_time_continuity(args.input, hdr["width"], hdr["height"])
    else:
        convert(args)
