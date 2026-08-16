#!/usr/bin/env python3
# ======================================================================
# RETIRED - see legacy/README.md. NOT used by run_pipeline.sh.
#
# Silently dropped every record whose payload was not exactly width*height bytes,
# logging one small [skip] line, so a broken recording produced an almost-empty
# .h5 that looked successful. Also used a nested /events/x schema unlike the rest
# of the project. Use cevt_to_events.py.
# ======================================================================

"""
cevt_to_h5.py — Convert CAROECT-D recordings to the unified events.h5 schema.

⚠ DEPRECATED — superseded by cevt_to_events.py.
This script only ever tries ONE hypothesis (dense accumulated frame) and
falls back to a fabricated frame_idx*fps timestamp for any record that
isn't exactly width*height bytes — which is now known to be wrong for
EVT3.0 payload records (the camera's confirmed real format; see
evs_recorder.cpp). cevt_to_events.py tries dense-frame AND real EVT3.0
word decode (with a --debug-time-continuity diagnostic for the real t),
and is now the one converter used by run_pipeline.sh's `real` command.
Kept here only for re-inspecting old recordings / comparison — do not
wire this into new pipeline runs.

UNIFIED SCHEMA (same as run_v2e.py / run_dvsvolt.py output, same as read_evt3.py):
  events.h5
    /events/x     uint16  [N]      pixel column
    /events/y     uint16  [N]      pixel row
    /events/t     int64   [N]      microseconds from start of clip
    /events/p     uint8   [N]      polarity: 1=ON, 0=OFF
  attrs: n_events, width, height, source, sensor_type

INPUT FORMATS SUPPORTED
-----------------------
.cevt  — evs_recorder.cpp output (dense accumulated frames, 1 byte/pixel):
           payload = H×W array, value: 128=no-event, 255=ON, 0=OFF
           Each buffer header carries frameId + timestampNs (but timestampNs=0
           for this camera because Arena's HasImageData() was false — we fall
           back to frame-count-based timestamps using the known capture fps).

events.h5 from simulator — already in the unified schema; this script can
           VERIFY and optionally re-stamp them for consistency.

WHY ONE SCHEMA
--------------
Training code, evaluation code, and sim-vs-real comparison all read the same
path. The only difference between real and simulated events is the 'source'
attribute ('triton2_real' vs 'v2e' / 'dvs_voltmeter'). This lets any
downstream script select or mix sources without format branching.

Usage:
  # Convert a real recording:
  python cevt_to_h5.py --input session01.cevt --output session01_real.h5 \\
         --fps 30.0 --width 1280 --height 720

  # Inspect a .cevt without converting:
  python cevt_to_h5.py --input session01.cevt --inspect

  # Verify a simulator .h5 has the right schema:
  python cevt_to_h5.py --verify session01_v2e.h5
"""

import argparse
import struct
import sys
import numpy as np
import h5py
from pathlib import Path

# ── .cevt container layout (matches evs_recorder.cpp) ──────────────
FILE_HEADER_FMT  = "<8sQIIIQ"   # magic(8) pixelFormat(8) bpp(4) w(4) h(4) reserved(8) = 36 B
RECORD_HEADER_FMT = "<QQQ"      # frameId(8) timestampNs(8) payloadSize(8) = 24 B
FILE_HEADER_SIZE  = struct.calcsize(FILE_HEADER_FMT)
RECORD_HEADER_SIZE = struct.calcsize(RECORD_HEADER_FMT)

MAGIC = b"CAROEVT1"

# ── unified .h5 schema keys ─────────────────────────────────────────
H5_KEYS = ("x", "y", "t", "p")


# ────────────────────────────────────────────────────────────────────
#  READ .cevt
# ────────────────────────────────────────────────────────────────────

def _read_file_header(f):
    raw = f.read(FILE_HEADER_SIZE)
    if len(raw) < FILE_HEADER_SIZE:
        raise ValueError("File too short for FileHeader — not a valid .cevt?")
    magic, pf, bpp, w, h, _res = struct.unpack(FILE_HEADER_FMT, raw)
    magic_s = magic.split(b"\x00")[0].decode("ascii", errors="replace")
    if magic_s != "CAROEVT1":
        print(f"  [warn] magic={magic_s!r} (expected 'CAROEVT1') — proceeding anyway")
    return {"pixel_format": pf, "bpp": bpp, "width": w, "height": h}


def _iter_records(f):
    while True:
        raw = f.read(RECORD_HEADER_SIZE)
        if len(raw) == 0:
            return
        if len(raw) < RECORD_HEADER_SIZE:
            print(f"  [warn] truncated RecordHeader at EOF ({len(raw)} bytes)")
            return
        frame_id, ts_ns, payload_size = struct.unpack(RECORD_HEADER_FMT, raw)
        payload = f.read(payload_size)
        if len(payload) < payload_size:
            print(f"  [warn] truncated payload for frame {frame_id} "
                  f"(expected {payload_size}, got {len(payload)})")
            return
        yield frame_id, ts_ns, payload


def cevt_to_events(path: str, width: int, height: int, fps: float):
    """
    Convert a .cevt recording to four arrays (x, y, t_us, p).

    PAYLOAD INTERPRETATION (confirmed by inspect_cevt.py analysis):
      The Triton2 EVS via Arena SDK returns DENSE ACCUMULATED FRAMES,
      not sparse EVT3.0 word streams. Each payload byte is one pixel:
        0x80 (128) = no event (background, neutral)
        0xFF (255) = ON event  (brightness increased)
        0x00 (0)   = OFF event (brightness decreased)
      Only 0x00 and 0xFF pixels become events in the output.

    TIMESTAMP STRATEGY:
      arena timestamps (timestampNs) are all 0 for this camera (HasImageData()
      was False so GetTimestampNs() was never called). We use frame-count-based
      timestamps: t_us = frame_index × (1e6 / fps). This is approximate but
      internally consistent for sim-vs-real comparison (simulators use the same
      frame-based timestamps from the TIFF sequence).
    """
    xs, ys, ts, ps = [], [], [], []
    expected_payload = width * height

    with open(path, "rb") as f:
        hdr = _read_file_header(f)
        print(f"  FileHeader: pf=0x{hdr['pixel_format']:x} bpp={hdr['bpp']} "
              f"w={hdr['width']} h={hdr['height']}")
        if hdr["width"] == 0 or hdr["height"] == 0:
            print(f"  [info] FileHeader has 0x0 — using CLI --width={width} --height={height}")

        frame_dt_us = int(round(1e6 / fps))
        frame_idx = 0

        for frame_id, ts_ns, payload in _iter_records(f):
            if len(payload) != expected_payload:
                print(f"  [skip] frame {frame_id}: payload {len(payload)} B "
                      f"!= {expected_payload} B expected — skipping")
                frame_idx += 1
                continue

            arr = np.frombuffer(payload, dtype=np.uint8).reshape(height, width)
            t_us = frame_idx * frame_dt_us

            on_ys, on_xs = np.where(arr == 255)
            if len(on_xs):
                xs.append(on_xs.astype(np.uint16))
                ys.append(on_ys.astype(np.uint16))
                ts.append(np.full(len(on_xs), t_us, dtype=np.int64))
                ps.append(np.ones(len(on_xs), dtype=np.uint8))

            off_ys, off_xs = np.where(arr == 0)
            if len(off_xs):
                xs.append(off_xs.astype(np.uint16))
                ys.append(off_ys.astype(np.uint16))
                ts.append(np.full(len(off_xs), t_us, dtype=np.int64))
                ps.append(np.zeros(len(off_xs), dtype=np.uint8))

            frame_idx += 1

    if not xs:
        raise RuntimeError("No events found — check payload format or --width/--height.")

    return (np.concatenate(xs), np.concatenate(ys),
            np.concatenate(ts), np.concatenate(ps))


# ────────────────────────────────────────────────────────────────────
#  WRITE unified events.h5
# ────────────────────────────────────────────────────────────────────

def write_h5(out_path: str, x, y, t, p, width, height, source: str, **meta):
    """Write the four arrays + attrs in the unified CAROECT-D schema."""
    with h5py.File(out_path, "w") as f:
        grp = f.create_group("events")
        grp.create_dataset("x", data=x, compression="gzip", compression_opts=4)
        grp.create_dataset("y", data=y, compression="gzip", compression_opts=4)
        grp.create_dataset("t", data=t, compression="gzip", compression_opts=4)
        grp.create_dataset("p", data=p, compression="gzip", compression_opts=4)
        f.attrs["n_events"]    = len(x)
        f.attrs["width"]       = width
        f.attrs["height"]      = height
        f.attrs["source"]      = source
        f.attrs["t_unit"]      = "microseconds"
        f.attrs["p_convention"] = "1=ON, 0=OFF"
        for k, v in meta.items():
            f.attrs[k] = str(v)
    print(f"  Wrote {len(x):,} events -> {out_path}")
    print(f"  t range: {t.min()} .. {t.max()} µs  "
          f"({(t.max()-t.min())/1e6:.2f} s)")
    print(f"  ON: {p.sum():,}  OFF: {(p==0).sum():,}")


# ────────────────────────────────────────────────────────────────────
#  VERIFY existing events.h5
# ────────────────────────────────────────────────────────────────────

def verify_h5(path: str):
    print(f"\nVerifying {path}")
    with h5py.File(path, "r") as f:
        if "events" not in f:
            print("  FAIL: no /events group")
            return False
        grp = f["events"]
        ok = True
        for k in H5_KEYS:
            if k not in grp:
                print(f"  FAIL: missing /events/{k}")
                ok = False
        if ok:
            n = len(grp["x"])
            print(f"  OK: {n:,} events  "
                  f"source={f.attrs.get('source','?')}  "
                  f"width={f.attrs.get('width','?')}  "
                  f"height={f.attrs.get('height','?')}")
            print(f"  t range: {grp['t'][0]} .. {grp['t'][-1]} µs")
        return ok


# ────────────────────────────────────────────────────────────────────
#  INSPECT .cevt without converting
# ────────────────────────────────────────────────────────────────────

def inspect_cevt(path: str):
    print(f"\nInspecting {path}")
    with open(path, "rb") as f:
        hdr = _read_file_header(f)
        print(f"  FileHeader: pf=0x{hdr['pixel_format']:x} bpp={hdr['bpp']} "
              f"w={hdr['width']} h={hdr['height']}")
        sizes, n = [], 0
        for fid, ts, payload in _iter_records(f):
            sizes.append(len(payload))
            n += 1
        sizes = np.array(sizes)
        print(f"  Records: {n}  "
              f"size min={sizes.min()} max={sizes.max()} mean={sizes.mean():.0f}")
        unique, counts = np.unique(sizes, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"    {c} records x {u} bytes")


# ────────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="CAROECT-D: .cevt -> unified events.h5")
    ap.add_argument("--input",   help=".cevt file to convert")
    ap.add_argument("--output",  help="Output .h5 path (default: same name, .h5 extension)")
    ap.add_argument("--fps",     type=float, default=30.0,
                    help="Frame rate of the recording (for timestamp generation, default 30)")
    ap.add_argument("--width",   type=int, default=1280)
    ap.add_argument("--height",  type=int, default=720)
    ap.add_argument("--source",  default="triton2_real",
                    help="Source tag written to .h5 attrs (default: triton2_real)")
    ap.add_argument("--inspect", action="store_true",
                    help="Just print .cevt stats, don't convert")
    ap.add_argument("--verify",  nargs="?", const=True, metavar="H5_PATH",
                    help="Verify schema of an existing .h5 (omit path to verify --output)")
    args = ap.parse_args()

    if args.verify and args.verify is not True:
        verify_h5(args.verify)
        return

    if not args.input:
        ap.print_help(); sys.exit(1)

    if args.inspect:
        inspect_cevt(args.input)
        return

    out = args.output or str(Path(args.input).with_suffix(".h5"))

    print(f"\nConverting {args.input} -> {out}")
    print(f"  fps={args.fps}  resolution={args.width}x{args.height}")

    x, y, t, p = cevt_to_events(args.input, args.width, args.height, args.fps)
    write_h5(out, x, y, t, p, args.width, args.height, args.source,
             fps=args.fps, source_file=Path(args.input).name)

    if args.verify:
        verify_h5(out)


if __name__ == "__main__":
    main()
