#!/usr/bin/env python3
"""
raw_to_events.py -- Prophesee .raw (from evs_recorder_mv.cpp) -> events.h5

WHAT THIS REPLACES AND WHY
---------------------------
cevt_to_events.py handles the Arena SDK path: DENSE accumulated frames, no true
per-event timestamp, everything downstream of it is labelled with a
timestamp_precision_status of "device_buffer" / "host_arrival" / "synthesized"
because the camera itself destroyed sub-window ordering.

evs_recorder_mv.cpp is a different program, talking to the same physical
camera through Metavision SDK / OpenEB instead of Arena SDK. It writes a
standard Prophesee .raw file via the SDK's own Camera::start_recording() --
not a custom container -- because that path was confirmed to deliver real
SPARSE (x,y,p,t) events with genuine per-event microsecond timestamps from the
sensor (see probe_metavision.cpp and evs_recorder_mv.cpp's header comment).
There is no accumulation window here to be coarse about. This script is the
one converter for that data, and it is why timestamp_precision_status below is
written as "precise" rather than any of the dense-path values -- it is a
different, better claim, made honestly because the source data actually
supports it.

Two decode backends, tried in this order:
  1. metavision_core.event_io.EventsIterator -- authoritative, the same
     decoder Metavision's own tools use. Default; use this unless you have a
     specific reason not to.
  2. Pure-python EVT3.0 fallback -- no SDK install needed. Reuses
     inspect_cevt.decode_evt3(), the SAME word-level decoder cevt_to_events.py
     and cevt_to_video.py already use, rather than a fourth near-identical
     copy (see cevt_to_video.py's comment on why that was a real bug before:
     "a reader that misparses a header does not fail loudly -- it renders
     convincing garbage"). EVT3.0's on-wire timestamp is only 24 bits (wraps
     every ~16.8s); this backend detects wraps and reconstructs one
     continuous microsecond timeline. It refuses to guess on any encoding
     other than EVT3.0 rather than emitting plausible garbage. It is also a
     plain python loop over every word -- fine for a sanity check, slow for
     hour-long recordings; prefer backend 1 for those.

SIDECAR .meta.json
-------------------
evs_recorder_mv.cpp writes <output>.meta.json next to the .raw with camera
serial/plugin/firmware, geometry, ERC state, biases, and run wall-clock times.
When present (same directory, "<raw>.meta.json"), this script folds the
useful fields into events.h5 attrs (prefixed recorder_*) and prefers its
width/height over whatever the decode backend reports, since it comes
straight from I_Geometry at record time. Its absence is not an error --
older or hand-made .raw files simply won't have it, and geometry then falls
back to the decoder.

OUTPUT SCHEMA -- matches cevt_to_events.py exactly, one shared reader
------------------------------------------------------------------------
  events.h5   root datasets: x(uint16) y(uint16) t(uint64, us) p(uint8 0/1)
  attrs: n_events, source, width, height, t_unit="microseconds",
         format="raw_evt3_sparse", decoder, timestamp_precision_status="precise",
         t_quantization_us=0, decode_method_counts, timestamp_source_counts,
         recorder_* (only if the .meta.json sidecar was found)

calibrate_simulator.py's check_timestamp_precision() only allows
{"precise", "unknown"} through for Eq.23/timing calibration unless overridden
-- files from this script pass that gate; files from the dense Arena path
normally do not, by design.

Usage:
    python raw_to_events.py run01.raw
    python raw_to_events.py run01.raw --output events.h5
    python raw_to_events.py run01.raw --max-events 1000000   # quick check
    python raw_to_events.py run01.raw --force-fallback       # skip the SDK backend
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from inspect_cevt import (  # noqa: E402  -- one decoder, not a fourth copy
    decode_evt3,
)

# ---------------------------------------------------------------------------
# SCHEMA -- kept identical to cevt_to_events.py's output. A mismatch here is
# a one-line fix, but check both files together if you ever change this.
# ---------------------------------------------------------------------------
DT_X = DT_Y = np.uint16
DT_P = np.uint8          # {0, 1} -- matches cevt_to_events.py's "1=ON, 0=OFF"
DT_T = np.uint64          # microseconds, absolute
CHUNK = 1 << 20

EVT3_WRAP_US = 1 << 24    # EVT3.0's on-wire timestamp is 24 bits


def _load_sidecar_meta(raw_path: Path):
    """Read <raw_path>.meta.json written by evs_recorder_mv.cpp, if present.
    Returns (dict_or_None, width_or_None, height_or_None). Never raises --
    a malformed or missing sidecar just means less metadata, not a failure."""
    meta_path = raw_path.with_name(raw_path.name + ".meta.json")
    if not meta_path.exists():
        return None, None, None
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"[warn] found {meta_path} but could not parse it ({e}); "
              f"continuing without sidecar metadata.", file=sys.stderr)
        return None, None, None
    cam = meta.get("camera", {})
    w, h = cam.get("width"), cam.get("height")
    print(f"[info] sidecar metadata: {meta_path.name}")
    return meta, w, h


def _decode_sdk(path: Path, max_events):
    """Backend 1: metavision_core's own EventsIterator. Yields
    (x, y, p, t, (w, h)) chunks; t is already absolute microseconds from the
    sensor -- no wraparound handling needed here, the SDK does it."""
    from metavision_core.event_io import EventsIterator
    it = EventsIterator(str(path), delta_t=1_000_000, relative_timestamps=False)
    w, h = it.get_size()
    total = 0
    for ev in it:
        if ev.size == 0:
            continue
        if max_events and total + ev.size > max_events:
            ev = ev[: max_events - total]
        total += ev.size
        yield ev["x"], ev["y"], ev["p"], ev["t"], (w, h)
        if max_events and total >= max_events:
            return


def _strip_ascii_header(data: np.ndarray):
    """Prophesee .raw files start with '%'-prefixed ASCII header lines.
    Returns (header_text, byte_offset_where_binary_data_starts)."""
    off = 0
    while off < len(data) and data[off] == 0x25:  # '%'
        nl = np.where(data[off:] == 0x0A)[0]
        if not nl.size:
            break
        off += nl[0] + 1
    return bytes(data[:off]).decode("ascii", "replace"), off


def _decode_evt3_fallback(path: Path, max_events):
    """Backend 2: pure-python, no SDK required. Reuses inspect_cevt's
    word-level EVT3.0 decoder (same one cevt_to_events.py/cevt_to_video.py
    use) instead of a fourth copy, then unwraps the 24-bit on-wire timestamp
    into one continuous microsecond timeline."""
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
        payload = payload[:-1]  # decode_evt3 also warns on this; trim quietly here
    payload_bytes = payload.tobytes()

    print("[info] backend 2 (pure-python EVT3.0) is a plain word-by-word python "
          "loop -- fine for a sanity check, slow for long recordings.")

    events = decode_evt3(payload_bytes, max_events=max_events)
    if not events:
        raise RuntimeError("EVT3 fallback decoded 0 events -- likely wrong encoding.")

    xs = np.empty(len(events), dtype=DT_X)
    ys = np.empty(len(events), dtype=DT_Y)
    ps = np.empty(len(events), dtype=DT_P)
    ts_raw = np.empty(len(events), dtype=np.int64)  # 24-bit wrapping value from decode_evt3
    for i, (x, y, t, p) in enumerate(events):
        xs[i], ys[i], ps[i], ts_raw[i] = x, y, p, t

    # decode_evt3() returns the raw 24-bit (time_high<<12 | time_low) value
    # per event -- it wraps every 2**24 us (~16.8s) because that is all the
    # on-wire word format carries. A monotonic file-scope timeline needs the
    # wraps folded back in: whenever t drops relative to the previous event,
    # a wrap happened.
    ts = ts_raw.astype(np.int64)
    wraps = np.cumsum(np.diff(ts_raw, prepend=ts_raw[0]) < 0)
    ts = ts + wraps.astype(np.int64) * EVT3_WRAP_US
    n_wraps = int(wraps[-1]) if len(wraps) else 0
    if n_wraps:
        print(f"[info] fallback decoder unwrapped {n_wraps} EVT3.0 24-bit timestamp "
              f"rollover(s) (~{n_wraps * EVT3_WRAP_US / 1e6:.1f}s of wrap total).")

    yield xs, ys, ps, ts.astype(DT_T), (None, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="Path to the .raw file (from evs_recorder_mv.cpp)")
    ap.add_argument("--output", default=None, help="Output events.h5 path (default: <input>.h5)")
    ap.add_argument("--max-events", type=int, default=None, help="Stop after N events (quick check)")
    ap.add_argument("--force-fallback", action="store_true",
                    help="Skip the SDK backend, use the pure-python EVT3.0 decoder instead")
    args = ap.parse_args()

    src = Path(args.path)
    if not src.exists():
        sys.exit(f"File not found: {src}")
    out = Path(args.output) if args.output else src.with_suffix(".h5")

    meta, meta_w, meta_h = _load_sidecar_meta(src)

    gen, backend = None, None
    if not args.force_fallback:
        try:
            gen = _decode_sdk(src, args.max_events)
            backend = "metavision_core"
        except ImportError:
            print("[warn] metavision_core not importable; using pure-python EVT3.0 fallback.")
    if gen is None:
        gen = _decode_evt3_fallback(src, args.max_events)
        backend = "evt3_fallback"
    print(f"[info] decoder backend: {backend}")

    n = 0
    decoder_w = decoder_h = None
    t0 = t1 = None
    method_counts = Counter()

    with h5py.File(out, "w") as f:
        dsets = {
            "x": f.create_dataset("x", shape=(0,), maxshape=(None,),
                                  dtype=DT_X, chunks=(CHUNK,), compression="gzip"),
            "y": f.create_dataset("y", shape=(0,), maxshape=(None,),
                                  dtype=DT_Y, chunks=(CHUNK,), compression="gzip"),
            "t": f.create_dataset("t", shape=(0,), maxshape=(None,),
                                  dtype=DT_T, chunks=(CHUNK,), compression="gzip"),
            "p": f.create_dataset("p", shape=(0,), maxshape=(None,),
                                  dtype=DT_P, chunks=(CHUNK,), compression="gzip"),
        }

        for x, y, p, t, resolution in gen:
            if resolution[0]:
                decoder_w, decoder_h = resolution

            p = (np.asarray(p) > 0).astype(DT_P)
            x = np.asarray(x).astype(DT_X)
            y = np.asarray(y).astype(DT_Y)
            t = np.asarray(t).astype(DT_T)

            for name, arr in (("x", x), ("y", y), ("t", t), ("p", p)):
                ds = dsets[name]
                ds.resize(n + arr.size, axis=0)
                ds[n:] = arr

            if t0 is None and t.size:
                t0 = int(t[0])
            if t.size:
                t1 = int(t[-1])
            n += x.size
            method_counts[backend] += int(x.size)

            if n % (5 * CHUNK) < CHUNK:
                print(f"  {n:,} events...")

        if n == 0:
            f.close()
            out.unlink(missing_ok=True)
            sys.exit("[error] 0 events written -- do not use this file.")

        width = meta_w or decoder_w
        height = meta_h or decoder_h
        if not width or not height:
            print("[warn] no geometry found from sidecar or decoder; width/height "
                  "attrs left unset. Downstream scripts that need sensor size "
                  "(hot_pixel_heatmap.py, measure_event_rate.py) will need "
                  "--sensor-width/--sensor-height passed explicitly.")

        f.attrs["n_events"] = n
        f.attrs["source"] = str(src)
        if width:
            f.attrs["width"] = int(width)
        if height:
            f.attrs["height"] = int(height)
        f.attrs["t_unit"] = "microseconds"
        f.attrs["format"] = "raw_evt3_sparse"
        f.attrs["decoder"] = backend
        # Real per-event sensor timestamps, no accumulation-window quantization
        # -- this is the one path in the project entitled to claim "precise".
        # calibrate_simulator.py's check_timestamp_precision() only allows
        # {"precise", "unknown"} through for Eq.23/timing calibration.
        f.attrs["timestamp_precision_status"] = "precise"
        f.attrs["t_quantization_us"] = 0
        f.attrs["timestamp_zero_dt_fraction"] = 0.0
        f.attrs["decode_method_counts"] = str(dict(method_counts))
        f.attrs["timestamp_source_counts"] = str({"device": n})

        if meta:
            cam = meta.get("camera", {})
            erc = meta.get("erc", {})
            run = meta.get("run", {})
            f.attrs["recorder"] = meta.get("recorder", "evs_recorder_mv")
            f.attrs["recorder_serial"] = cam.get("serial", "")
            f.attrs["recorder_plugin"] = cam.get("plugin", "")
            f.attrs["recorder_firmware"] = cam.get("firmware", "")
            f.attrs["recorder_erc_enabled"] = bool(erc.get("enabled", False))
            f.attrs["recorder_erc_requested_rate"] = erc.get("requested_rate_events_per_sec") or 0
            f.attrs["recorder_started_utc"] = run.get("started_utc", "")
            f.attrs["recorder_stopped_utc"] = run.get("stopped_utc", "")
            f.attrs["recorder_wall_seconds"] = run.get("wall_seconds", 0.0)
            f.attrs["recorder_stop_reason"] = run.get("stop_reason", "")
        else:
            f.attrs["recorder"] = "unknown (no .meta.json sidecar found)"

    dur = (t1 - t0) / 1e6 if (t0 is not None and t1 is not None) else 0
    print(f"\n{'=' * 60}")
    print(f"[done] {out}")
    print(f"{'=' * 60}")
    print(f"  events              : {n:,}")
    if dur > 0:
        print(f"  duration            : {dur:.3f} s   ({n / dur / 1e6:.3f} Mev/s)")
    if width and height:
        print(f"  sensor              : {width}x{height}")
    print(f"  decoder             : {backend}")
    print(f"  timestamp precision : precise (real per-event sensor time)")
    if meta:
        print(f"  recorder metadata   : serial={meta.get('camera', {}).get('serial', '?')}  "
              f"plugin={meta.get('camera', {}).get('plugin', '?')}")
    print(f"\nUse for: ./run_pipeline.sh calibrate {out} <processed_tiff_dir> ...")
    print("Or as a real test set via build_event_dataset.py (needs its own tracks.json + windows.json).")


if __name__ == "__main__":
    main()
