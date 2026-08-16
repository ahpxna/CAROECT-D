# legacy/ — retired scripts, kept for reference only

Nothing in this folder is wired into `run_pipeline.sh`. Each file was retired for
a specific, verified reason, recorded below so the same dead end is not
re-explored a fourth time. Nothing here is deleted: old recordings may still need
one of these to be re-read, and the reasoning is worth preserving for the
CAROECT-D write-up.

| File | Why it was retired |
|---|---|
| `xypt_to_h5.py` | Reads a `CAROXYP1` container produced by `evs_recorder --flat-xypt`. **That flag no longer exists and never worked on this camera.** The XYPT path was chased across three sessions before being closed conclusively: node `EvsOutputFormat` does not exist on the TRT009S-E node map, and neither does a `PixelFormat` node or a `LucidXYTP128f` entry (`--dump-nodes --filter Pixel` returns 3 unrelated nodes). `LucidXYTPPixel` appears in `all_arena_api.txt` because that dump documents the whole Arena/LUCID product line, not this model's firmware. No file in this format was ever written. |
| `cevt_to_h5.py` | Silently discarded every record whose payload was not exactly `width*height` bytes, logging only a small `[skip]` line. On a recording with wrong geometry or ring truncation it produced an almost-empty `.h5` that looked successful. Superseded by `cevt_to_events.py`, which classifies `dense` / `empty` / `undersized` / `oversized` separately and **refuses to write output** when more than half the records are unparseable. It also wrote a nested `/events/x` schema, unlike every other script in the project (root-level `x`). |
| `record_evs.py`, `read_evt3.py` | Both depend on the Metavision/OpenEB SDK and a `.raw` EVT3.0 file. The camera never produces a sparse EVT3.0 stream — `AcquisitionAccumulationMode` (the TimeBased/EventBased switch) is firmware-locked at `IsAvailable=false`, and no GenICam path to unlock it exists (over 2160 nodes swept). The real-event path is `evs_recorder.cpp` → `cevt_to_events.py`. |
| `pipeline.sh` | Old wrapper that stopped at `events.h5` (preprocess + v2e only) and still referenced the XYPT flags. Fully superseded by `run_pipeline.sh`, which runs preprocess → simulate → SAM3 → label transfer → dataset → train → eval. |

## If you need to re-read an old recording

`cevt_to_events.py` reads both container versions (`CAROEVT1` and `CAROEVT2`)
directly, so `cevt_to_h5.py` is not needed for that:

```bash
python cevt_to_events.py old_recording.cevt --debug-time-continuity
python cevt_to_events.py old_recording.cevt --output-h5 out.h5 --fps <hz>
```

`CAROEVT1` files carry no measured timestamps at all, so `--fps` is required and
the result is labelled `timestamp_precision_status='synthesized'`.
`calibrate_simulator.py` will correctly refuse to use it for timing or Eq.23
calibration.

## If a future firmware unlocks sparse output

Re-check with the tools already in `evs_recorder`:

```bash
./evs_recorder --node-info AcquisitionAccumulationMode   # want IsAvailable=true
./evs_recorder --node-info DeviceFirmwareVersion         # now also prints Value
./evs_recorder --set-enum AcquisitionAccumulationMode --set-enum-value EventBased
```

Only if `IsAvailable` flips to `true` is anything in this folder worth reviving.
