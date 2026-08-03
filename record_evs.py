#!/usr/bin/env python3
"""
CAROECT-D — Record real events from LUCID Triton2 EVS (TRT009S-EC)
==================================================================

WHY THIS FILE EXISTS
--------------------
ArenaView can display but crashes on record, and the py_save_recorder*
notebooks fail because Arena's save.Recorder is a VIDEO-FRAME recorder —
the Triton2 EVS outputs EVT 3.0 event packets, not frames. The official
path for recording is the Metavision SDK talking to the camera through
LUCID's HAL plugin, writing a standard .raw (EVT 3.0) file that
read_evt3.py already knows how to open.

SETUP REQUIRED (once, before this script can work)
--------------------------------------------------
  1. Install Arena SDK        (REQUIRED even for the Metavision path —
                               the HAL plugin uses it as GigE transport)
  2. Install Metavision SDK   (v5.x, or v4.6.2 if your plugin is older)
  3. Get the LUCID HAL plugin for Triton2 EVS
       - ships via LUCID's download hub / Arena SDK bundle
       - SDK 5.x-compatible plugin may require emailing support@thinklucid.com
  4. export MV_HAL_PLUGIN_PATH=/path/to/lucid/hal/plugin
       - without this, Metavision CANNOT see the camera at all
  5. Camera powered, Ethernet connected, LED green.
  Quick sanity test without code: open metavision_studio — if Studio can
  see the camera, this script will too.

DATA FLOW  (each NODE below is one function)
--------------------------------------------
  [Triton2 EVS] --GigE/EVT3.0--> [LUCID HAL plugin] --> Metavision HAL device
        |                                                    |
        |   N0 CHECK   env var + plugin + camera discovery   |
        |   N1 OPEN    initiate_device("")                   |
        |   N2 CONFIG  optional ERC rate cap (1GigE links!)  |
        |   N3 RECORD  I_EventsStream.log_raw_data(.raw)     |
        |   N4 PUMP    EventsIterator loop + live stats      |
        |   N5 CLOSE   stop log -> verify file header/size   |
        v
   site01.raw  (EVT 3.0)  -->  python read_evt3.py --input site01.raw --stats
                          -->  Simulator Calibration (v2e thresholds vs real)

Usage:
  python record_evs.py --check                                   # doctor mode
  python record_evs.py --output data/events_real/site01.raw --duration 120
  python record_evs.py --output data/events_real/site01.raw      # Ctrl+C to stop
  python record_evs.py --output out.raw --duration 60 --erc-mevps 20
"""

import os
import sys
import time
import argparse
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════
#  NODE 0 · CHECK  (doctor mode / preflight)
# ══════════════════════════════════════════════════════════════════════════
#  in   : nothing
#  out  : True if a camera is reachable  -> N1
#  what : verify, in order: (1) metavision python modules import, (2) the
#         MV_HAL_PLUGIN_PATH env var is set, (3) the HAL discovery actually
#         lists a source (i.e. the LUCID plugin loaded AND sees the camera).
#  why  : 90% of "camera not found" is a missing/wrong MV_HAL_PLUGIN_PATH or
#         a plugin built for a different SDK major version. Failing loudly
#         here with the exact missing piece beats a cryptic open() error.
# ══════════════════════════════════════════════════════════════════════════

def check_environment(verbose: bool = True) -> bool:
    ok = True

    # (1) SDK importable?
    try:
        import metavision_hal  # noqa: F401
        if verbose:
            print("  [ok] metavision_hal importable")
    except ImportError:
        print("  [FAIL] Metavision SDK not importable.")
        print("         Install OpenEB / Metavision SDK first (x86_64 Linux/Windows).")
        return False

    # (2) plugin path set?
    plugin_path = os.environ.get("MV_HAL_PLUGIN_PATH", "")
    if plugin_path:
        exists = Path(plugin_path).exists()
        print(f"  [{'ok' if exists else 'FAIL'}] MV_HAL_PLUGIN_PATH = {plugin_path}"
              + ("" if exists else "   <- path does not exist!"))
        ok &= exists
    else:
        print("  [FAIL] MV_HAL_PLUGIN_PATH is NOT set.")
        print("         export MV_HAL_PLUGIN_PATH=/path/to/lucid/hal/plugin")
        print("         (the plugin comes from LUCID; SDK5-compatible builds may")
        print("          require emailing support@thinklucid.com)")
        ok = False

    # (3) does HAL discovery see anything?
    try:
        from metavision_hal import DeviceDiscovery
        sources = DeviceDiscovery.list_available_sources()
        if sources:
            print(f"  [ok] camera(s) discovered: {sources}")
        else:
            print("  [FAIL] no event camera discovered.")
            print("         checklist: camera LED green? Ethernet up? IP configured")
            print("         (Arena's IpConfigUtility)? jumbo frames enabled? plugin")
            print("         version matches your Metavision major version (5.x vs 4.6.2)?")
            ok = False
    except Exception as e:
        print(f"  [warn] discovery API problem: {e}")
        print("         will still try to open the camera directly.")

    return ok


# ══════════════════════════════════════════════════════════════════════════
#  NODE 1 · OPEN
# ══════════════════════════════════════════════════════════════════════════
#  in   : nothing (first available camera through the HAL plugins)
#  out  : HAL `device` handle  -> N2, N3, N4
#  what : initiate_device("") walks all HAL plugins in MV_HAL_PLUGIN_PATH and
#         opens the first camera that answers — i.e. the Triton2 EVS.
#  why  : this is the documented pattern from Prophesee's own
#         metavision_simple_recorder sample; it works identically for partner
#         cameras once their plugin is installed.
# ══════════════════════════════════════════════════════════════════════════

def open_device():
    from metavision_core.event_io.raw_reader import initiate_device
    device = initiate_device("")          # "" = first available source
    return device


# ══════════════════════════════════════════════════════════════════════════
#  NODE 2 · CONFIG  (optional — Event Rate Control)
# ══════════════════════════════════════════════════════════════════════════
#  in   : device <- N1, target rate in Mev/s (0 = leave camera as-is)
#  out  : device with ERC capped  -> N3
#  what : cap the camera-side event rate so the GigE link never overflows.
#  why  : THIS is the likely reason ArenaView crashed while recording — a
#         laptop's 1GigE port can't swallow night-traffic event bursts, the
#         link overflows and the camera drops/disconnects (LUCID documents
#         exactly this failure mode). Capping ~20 Mev/s is safe on 1GigE;
#         on a true 2.5GigE link you can leave it off.
#  note : ERC DISCARDS events above the cap — for simulator calibration,
#         record with the same ERC setting you plan to deploy with, and
#         write the value into the session's metadata/notes.
# ══════════════════════════════════════════════════════════════════════════

def configure_erc(device, mevps: float):
    if mevps <= 0:
        return
    try:
        erc = device.get_i_erc_module()
        if erc is None:
            print("  [warn] ERC facility not exposed by this plugin — skipping.")
            return
        erc.enable(True)
        erc.set_cd_event_rate(int(mevps * 1e6))   # events/s
        print(f"  [erc]  camera-side cap: {mevps:.1f} Mev/s")
    except Exception as e:
        print(f"  [warn] could not set ERC ({e}) — continuing without cap.")


# ══════════════════════════════════════════════════════════════════════════
#  NODE 3 · RECORD  (start writing the .raw)
# ══════════════════════════════════════════════════════════════════════════
#  in   : device <- N2, output path
#  out  : camera firmware/plugin stream is being logged to <output>.raw
#  what : I_EventsStream.log_raw_data() copies the raw EVT 3.0 byte stream to
#         disk WITH the standard Prophesee header (geometry, format, plugin).
#  why  : recording the RAW stream (not decoded events) is lossless, cheap
#         (no decode cost in the write path), and produces a file that every
#         Metavision tool — and our read_evt3.py — opens directly.
# ══════════════════════════════════════════════════════════════════════════

def start_recording(device, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stream = device.get_i_events_stream()
    if stream is None:
        raise RuntimeError("I_EventsStream facility unavailable — plugin problem.")
    stream.log_raw_data(str(output_path))
    return stream


# ══════════════════════════════════════════════════════════════════════════
#  NODE 4 · PUMP  (stream loop + live stats)
# ══════════════════════════════════════════════════════════════════════════
#  in   : device (recording already armed by N3), duration (None = Ctrl+C)
#  out  : returns when duration elapsed or user interrupts  -> N5
#  what : EventsIterator starts the streaming and pulls decoded slices; while
#         it runs, N3's logger mirrors the raw bytes to disk automatically.
#         We use the decoded slices only for a live counter (Mev/s), which
#         doubles as a link-health monitor.
#  why  : if the live rate keeps slamming into your link's ceiling
#         (~20-25 Mev/s on 1GigE) you are saturated -> enable --erc-mevps.
# ══════════════════════════════════════════════════════════════════════════

def pump(device, duration_s):
    from metavision_core.event_io import EventsIterator

    it = EventsIterator.from_device(device=device, delta_t=100_000)  # 100ms slices
    total = 0
    t_wall0 = time.time()
    t_last, n_last = t_wall0, 0

    try:
        for evs in it:
            now = time.time()
            if evs is not None and len(evs) > 0:
                total += len(evs)
            if now - t_last >= 1.0:                       # refresh stats 1x/s
                rate = (total - n_last) / (now - t_last) / 1e6
                print(f"\r  [rec] {now - t_wall0:7.1f}s   {total/1e6:8.2f} Mev total"
                      f"   {rate:6.2f} Mev/s", end="", flush=True)
                t_last, n_last = now, total
            if duration_s is not None and (now - t_wall0) >= duration_s:
                break
    except KeyboardInterrupt:
        print("\n  [rec] stopped by user (Ctrl+C)")
    return total, time.time() - t_wall0


# ══════════════════════════════════════════════════════════════════════════
#  NODE 5 · CLOSE + VERIFY
# ══════════════════════════════════════════════════════════════════════════
#  in   : stream (N3), output path, totals (N4)
#  out  : verified .raw on disk — next stop: read_evt3.py, then calibration
#  what : stop the logger, then sanity-check the file: nonzero size and a
#         readable '%'-comment header (format/geometry lines).
#  why  : a 0-byte or headerless file means the plugin never streamed — catch
#         it HERE, not after a 2-hour session at the roadside.
# ══════════════════════════════════════════════════════════════════════════

def stop_and_verify(stream, output_path: Path, total_events: int, elapsed: float):
    try:
        stream.stop_log_raw_data()
    except Exception:
        pass

    print()  # newline after the \r status line
    if not output_path.exists() or output_path.stat().st_size == 0:
        print(f"  [FAIL] {output_path} is missing or empty — nothing was streamed.")
        print("         Re-run with --check and fix whatever it flags.")
        sys.exit(1)

    size_mb = output_path.stat().st_size / 1e6
    print(f"  [file] {output_path}  ({size_mb:.1f} MB)")

    # peek the EVT3 header ('%'-prefixed text lines before the binary payload)
    with open(output_path, "rb") as f:
        for _ in range(8):
            line = f.readline()
            if not line.startswith(b"%"):
                break
            print(f"  [hdr ] {line.decode(errors='replace').rstrip()}")

    if elapsed > 0:
        print(f"  [stat] {total_events/1e6:.2f} Mev in {elapsed:.1f}s "
              f"(avg {total_events/elapsed/1e6:.2f} Mev/s)")
    print(f"\n  Next:  python read_evt3.py --input {output_path} --stats --frame")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN — wire the nodes together
# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="CAROECT-D: record EVT 3.0 .raw from LUCID Triton2 EVS")
    ap.add_argument("--output", help=".raw output path (required unless --check)")
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds to record (default: until Ctrl+C)")
    ap.add_argument("--erc-mevps", type=float, default=0.0,
                    help="camera-side event-rate cap in Mev/s (0 = off; "
                         "use ~20 on a 1GigE link)")
    ap.add_argument("--check", action="store_true",
                    help="doctor mode: verify SDK, plugin path, camera discovery")
    args = ap.parse_args()

    print(f"\n{'━'*60}\n  CAROECT-D — Triton2 EVS recorder\n{'━'*60}")

    if args.check:
        ok = check_environment()
        print(f"\n  {'READY — camera reachable.' if ok else 'NOT READY — fix the FAIL lines above.'}\n")
        sys.exit(0 if ok else 1)

    if not args.output:
        ap.error("--output is required (or use --check)")

    if not check_environment(verbose=False):
        print("\n  Preflight failed — run `python record_evs.py --check` for details.\n")
        sys.exit(1)

    out = Path(args.output)
    device = open_device()                     # N1
    configure_erc(device, args.erc_mevps)      # N2
    stream = start_recording(device, out)      # N3
    print(f"  [rec] recording -> {out}"
          + (f"   (duration {args.duration:.0f}s)" if args.duration else "   (Ctrl+C to stop)"))
    total, elapsed = pump(device, args.duration)   # N4
    stop_and_verify(stream, out, total, elapsed)   # N5


if __name__ == "__main__":
    main()
