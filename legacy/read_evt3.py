#!/usr/bin/env python3
"""
CAROECT-D — EVT 3.0 Event Data Reader
========================================
Read real event data from LUCID Triton2 EVS (.raw files in EVT 3.0 format).
Used for: sim2real validation and final AI model testing.

This is SEPARATE from the synthetic pipeline (v2e / DVS-Voltmeter).
Real EVT 3.0 data → test the trained AI model after training on synthetic.

Requires OpenEB / MetaVision SDK:
  pip install metavision-sdk-base
  Docs: https://docs.prophesee.ai/stable/installation/index.html

EVT 3.0 format:
  Binary format from Prophesee (used in LUCID Triton2 EVS via IMX636 sensor).
  Each event: x (uint16), y (uint16), t (uint64, microseconds), p (uint8: 0/1)

Usage:
  python read_evt3.py --input recording.raw
  python read_evt3.py --input recording.raw --output events.h5 --stats
"""

import argparse
import numpy as np
from pathlib import Path


def check_openeb():
    try:
        from metavision_core.event_io import EventsIterator
        return True
    except ImportError:
        print("ERROR: OpenEB SDK not installed.")
        print("  Install: pip install metavision-sdk-base")
        print("  Or:      https://docs.prophesee.ai/stable/installation/index.html")
        print("  Note: requires x86_64 Linux or Windows — not available on ARM Mac")
        return False


def read_evt3(raw_path: str, chunk_us: int = 1_000_000) -> np.ndarray:
    """
    Decode EVT 3.0 .raw file into structured numpy array.

    Args:
        raw_path:  path to .raw file from LUCID EVS recording
        chunk_us:  read chunk size in microseconds (default: 1s chunks)

    Returns:
        numpy structured array with fields: x, y, t, p
    """
    from metavision_core.event_io import EventsIterator

    chunks = []
    for evs in EventsIterator(raw_path, delta_t=chunk_us):
        if evs is not None and len(evs) > 0:
            chunks.append(evs)

    if not chunks:
        raise ValueError(f"No events found in {raw_path}")

    return np.concatenate(chunks)


def print_stats(events: np.ndarray, path: str):
    t0, t1 = events["t"][0], events["t"][-1]
    duration_s = (t1 - t0) / 1e6
    n_on  = (events["p"] == 1).sum()
    n_off = (events["p"] == 0).sum()
    rate  = len(events) / duration_s

    print(f"\n  File:      {path}")
    print(f"  Events:    {len(events):,}")
    print(f"  Duration:  {duration_s:.2f}s  ({t0}μs → {t1}μs)")
    print(f"  Rate:      {rate/1e6:.2f} Mev/s")
    print(f"  ON:        {n_on:,}  ({100*n_on/len(events):.1f}%)")
    print(f"  OFF:       {n_off:,}  ({100*n_off/len(events):.1f}%)")
    print(f"  x range:   [{events['x'].min()}, {events['x'].max()}]")
    print(f"  y range:   [{events['y'].min()}, {events['y'].max()}]")


def save_h5(events: np.ndarray, output_path: str):
    import h5py
    with h5py.File(output_path, "w") as f:
        f.create_dataset("x", data=events["x"], compression="gzip")
        f.create_dataset("y", data=events["y"], compression="gzip")
        f.create_dataset("t", data=events["t"], compression="gzip")
        f.create_dataset("p", data=events["p"], compression="gzip")
        f.attrs["format"]  = "EVT3_decoded"
        f.attrs["n_events"] = len(events)
    print(f"  Saved → {output_path}")


def events_to_frame(events: np.ndarray, width=1280, height=720,
                    t_start=None, t_end=None) -> np.ndarray:
    """
    Accumulate events in [t_start, t_end] into a 2D event frame.
    Returns: HxWx3 uint8 image  (ON=green, OFF=red, background=gray)
    """
    if t_start is not None:
        mask = (events["t"] >= t_start) & (events["t"] < t_end)
        evs = events[mask]
    else:
        evs = events

    frame = np.full((height, width, 3), 128, dtype=np.uint8)
    on_mask  = evs["p"] == 1
    off_mask = evs["p"] == 0

    # ON events → green, OFF → red
    frame[evs["y"][on_mask],  evs["x"][on_mask]]  = [0, 200, 0]
    frame[evs["y"][off_mask], evs["x"][off_mask]] = [0, 0, 200]
    return frame


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CAROECT-D: Read EVT 3.0 real event data")
    p.add_argument("--input",  required=True,       help="EVT 3.0 .raw file from LUCID EVS")
    p.add_argument("--output", default=None,         help="Save decoded events to .h5")
    p.add_argument("--stats",  action="store_true",  help="Print statistics")
    p.add_argument("--frame",  action="store_true",  help="Save first 50ms as event frame PNG")
    args = p.parse_args()

    if not check_openeb():
        exit(1)

    print(f"\n[EVT3] Reading: {args.input}")
    events = read_evt3(args.input)

    print_stats(events, args.input)

    if args.output:
        save_h5(events, args.output)

    if args.frame:
        import cv2
        t0 = events["t"][0]
        frame = events_to_frame(events, t_start=t0, t_end=t0 + 50_000)  # first 50ms
        out_path = str(Path(args.input).with_suffix("_frame50ms.png"))
        cv2.imwrite(out_path, frame)
        print(f"  Event frame (first 50ms) → {out_path}")
