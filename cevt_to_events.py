#!/usr/bin/env python3
"""
cevt_to_events.py — Convert .cevt (CAROECT-D custom container) to events.h5 and/or EVT3.0

.cevt FORMAT (written by evs_recorder.cpp):
  FileHeader 36 bytes  : magic "CAROEVT1" | pixelFormat | bitsPerPixel | W | H | reserved
  Per-buffer           : RecordHeader 24 bytes (frameId, timestampNs, payloadSize) + payload

PAYLOAD (confirmed empirically: Triton2 via Arena SDK delivers DENSE ACCUMULATED FRAMES):
  Each payload = W×H uint8 frame where:
    pixel = 128  → no event (neutral / no change during accumulation window)
    pixel = 0    → OFF event at this pixel
    pixel = 255  → ON event at this pixel
  NOT a Prophesee EVT3.0 bitstream — "EventFormat=EVT3_0" is the sensor↔FPGA protocol,
  not the GigE payload format. Arena SDK decodes it and delivers dense frames.

TIMESTAMPS: RecordHeader.timestampNs = 0 for all records (Arena SDK HasImageData()
  returned false → GetTimestampNs() was not callable at capture time).
  Timestamps are reconstructed from frame_index / --fps.

OUTPUT SCHEMAS:
  events.h5  : datasets x(uint16) y(uint16) t(uint64,µs) p(uint8 0/1) — same as
               run_v2e.py / run_dvsvolt.py / read_evt3.py output. One shared reader.
  .raw.evt3  : standard Prophesee EVT3.0 16-bit word stream (little-endian),
               readable by Metavision SDK / OpenEB / our inspect_cevt.py decoder.

Usage:
  python cevt_to_events.py recording.cevt --output-h5 events.h5
  python cevt_to_events.py recording.cevt --output-evt events.raw.evt3
  python cevt_to_events.py recording.cevt --output-h5 events.h5 --output-evt events.raw.evt3 --fps 30
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


# ── Dense-frame → (x, y, t, p) ───────────────────────────────────
def extract_events(payload: bytes, width: int, height: int, t_us: int):
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

        for frame_id, ts_ns, payload in iter_records(f):
            # Timestamp: ts_ns=0 in all records (capture-time limitation);
            # reconstruct from frame index × accumulation period.
            t_us = int(round(n_frames * 1_000_000.0 / args.fps))

            xs, ys, ts, ps = extract_events(payload, W, H, t_us)

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
                print(f"  frame {n_frames:>4} (id={frame_id})  "
                      f"t={t_us/1e6:.3f}s  ev={len(xs):>7,}  total={n_events:>9,}")

    print(f"\nProcessed {n_frames} frames  |  "
          f"{n_events:,} events  |  {n_skip} empty frames")

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
                hf.attrs["format"]      = "cevt_dense_accumulated"
                hf.attrs["note"]        = (
                    "t reconstructed from frame_index/fps (timestampNs=0 in source). "
                    "Schema matches run_v2e.py / run_dvsvolt.py / read_evt3.py output.")
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
                         "from frame index (default 30). Check ArenaView / camera node "
                         "AcquisitionFrameRate for the true value.")
    convert(ap.parse_args())
