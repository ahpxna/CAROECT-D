# CAROECT-D Pipeline

This repository implements the paper-aligned CAROECT-D data pipeline. Capture
uses 12-bit **SDR N-RAW, not N-Log**. DaVinci Resolve decodes that SDR source and
exports an already-linear 16-bit Rec.709 RGB TIFF working signal. The Python
pipeline does not linearize it again. Geometry is applied once to a shared RGB array; only then
does the pipeline branch into annotation images and event-simulator input.

The current working colourspace is linear Rec.709. Because Rec.709 and sRGB
share D65 primaries, annotation rendering keeps the linear primaries and then
applies the IEC sRGB transfer. The event branch computes luminance with
`Y = 0.2126 R + 0.7152 G + 0.0722 B`.

## Setup

```bash
python3 -m pip install -r requirements.txt
./setup_sim.sh
./build_linux.sh
```

`evs_recorder.cpp` is the current sparse Metavision RAW acquisition path.
Arena/`.cevt` code is retained under `legacy/` only for old recordings.

## Measurement-backed radiometric calibration

Optional radiometric corrections are disabled by default. Place measured
references under the configured calibration directory:

```text
calibration/dark/          no-light residual-offset frames
calibration/flat/          homogeneous non-zero field for gain/PRNU
calibration/linearity/     at least three homogeneous non-zero levels
calibration/gray_card.tiff measured neutral reference
calibration/chessboard/    RGB intrinsic-calibration images
```

Run `python3 calibrate.py --config config.yaml`. In addition to artifacts,
the command writes `calibration_manifest.json` with acquisition/decode
settings, reference type, validity, and artifact provenance. Enabling a
correction without a valid manifest entry or artifact makes preprocessing fail
loudly. A file merely existing never enables a correction.

Physical event-camera calibration is exposed separately:

```bash
./run_pipeline.sh calibrate-camera event-only calibration/event_poses
./run_pipeline.sh calibrate-camera register calibration/event_poses calibration/rgb_poses
```

The first command estimates event intrinsics. The second performs same-session
RGB/event registration and may reuse cached event intrinsics. Unknown cached
RMS values are reported as JSON null rather than crashing.

## Synthetic pipeline and isolated conditions

In DaVinci Resolve, convert the configured SDR N-RAW source to linear Rec.709
and export RGB uint16 TIFF. Do not export N-Log, and do not feed an SDR-encoded
delivery image directly to the Python pipeline. `camera.input_transfer` remains
`linear` so preprocessing treats the TIFF values as already linear and cannot
apply inverse gamma a second time. The TIFF
count should approximately equal real scene duration times
`camera.fps_original`; `fps_export` is timeline provenance only.

```bash
./run_pipeline.sh sim data/tiff/site01 site01 train default
./run_pipeline.sh sim data/tiff/site01 site01 train calibrated
```

The orchestrator runs preprocessing, both simulators, bidirectional SAM3,
causal label transfer, and dataset rendering. Default and calibrated runs
cannot share sample directories:

```text
data/datasets/v2e/default/dt_08340us/
data/datasets/v2e/calibrated/dt_08340us/
data/datasets/dvsvolt/default/dt_08340us/
data/datasets/dvsvolt/calibrated/dt_08340us/
```

Equivalent roots are created for 16,700, 33,300, and 50,000 microseconds.
Every root snapshots the effective config, git revision, representation
manifest, and available calibration manifest.

## Causal labels and detector-window ablation

`sam3_export_tracks.py` saves complete `frame_times_us` provenance. It runs
forward and backward propagation separately, preserves both artifacts, and
merges by continuity, confidence, and finally a trajectory-based receding
tie-break. Backward propagation never wins unconditionally.

For each SAM timestamp `t_k`, `label_transfer.py` creates
`[t_k - window_us, t_k)` and copies the box/mask observed at frame k exactly.
Changing frame k+1 cannot affect sample k. Legacy box interpolation is isolated
behind `--legacy-interpolation` for visualization and is rejected by the
normal dataset builder.

All window-duration variants retain the same ordered `t_k` and label sets;
only event history changes. Window duration is independent of the 119.88-fps
simulation input.

## Physical RGB/event synchronization

Use a flash or blinking target visible to both sensors:

```bash
python estimate_rgb_event_offset.py \
  --rgb-dir sync/rgb --events sync/events.h5 \
  --rgb-roi 100 100 80 80 --event-roi 100 100 80 80 \
  --fps 119.88 --output sync/sync.json
```

The sign convention is
`event_time_us = rgb_time_us + offset_us`. The estimator combines ROI
transition cross-correlation with matched-peak refinement. Physical label
transfer refuses a silent zero offset unless `--allow-unsynced` is explicitly
given. Synthetic events use zero offset by construction.

## Event representation

The default PNG shape is the common native grid `(720, 1280, 3)`:

- red: ON count;
- green: OFF count;
- blue: last-event timestamp within the causal window.

ON/OFF counts share one fixed `count_clip` fitted from training data. The
scale is persisted in `representation.json` and reused by validation, test,
real, default-simulator, and calibrated-simulator builds. Per-window maximum
normalization is forbidden because it erases event-rate amplitude. Optional
alternate output dimensions use letterboxing and transform labels; direct
16:9-to-square scale-fill is not supported. Ultralytics performs its normal
aspect-preserving training letterbox.

## Real events

```bash
./evs_recorder --output site01.raw --duration 60
./run_pipeline.sh real site01.raw site01
```

The common HDF5 schema is `x`, `y`, `t` in microseconds, and `p` with
`1=ON, 0=OFF`. Timestamp provenance attributes are retained. Controlled
latency or inter-event-interval calibration requires precise sensor timestamps
and an explicit stimulus reference; arbitrary road footage is never treated as
timing evidence.

## Simulator calibration

The named baseline objective remains total event rate plus ON fraction:

```bash
./run_pipeline.sh calibrate data/events_real/site01.h5 data/processed/site01 \
  v2e pos_thres 0.15 0.20 0.25 0.30
./run_pipeline.sh calibrate data/events_real/site01.h5 data/processed/site01 \
  dvsvolt k1 3.0 4.0 5.3 6.5 8.0
```

`calibrate_simulator.py` also exposes traceable optional fields for
static/motion rate, controlled edge latency, and IEI summaries. Timing fields
require both precise timestamp provenance and `--stimulus-reference`.
Single-parameter Optuna/TPE remains experimental/proposed and is not required.

## Training and evaluation

Training accepts exactly one condition root and requires its manifest:

```bash
./run_pipeline.sh train data/datasets/v2e/calibrated/dt_08340us
./run_pipeline.sh eval runs/caroectd/exp/weights/best.pt \
  data/datasets/v2e/calibrated/dt_08340us/data.yaml
```

The dataset manifest is copied beside the training run so every checkpoint is
traceable. A mixed root requires explicit `--mixed` opt-in. Compare default
and calibrated models on the same real-event test definition and representation.

## Main files

- `preprocess.py`: validated corrections, one shared geometry, Rec.709 Y and sRGB annotation branches.
- `run_v2e.py`: v2e library path preserving sub-8-bit precision in float.
- `run_dvsvolt.py`: DVS-Voltmeter library path on its expected DN scale.
- `sam3_export_tracks.py`: bidirectional tracks, masks, timestamps, and merge provenance.
- `label_transfer.py`: exact frame-k labels on causal half-open windows.
- `build_event_dataset.py`: fixed-scale event representation at native aspect.
- `estimate_rgb_event_offset.py`: measured physical clock alignment.
- `calibrate_event_camera.py`: event intrinsics and RGB/event registration.
- `generate_biases.py`: validated IMX636 legacy text bias sweeps.

## Legacy

`legacy/` is retained only for compatibility with old recordings. Runtime
orchestration does not use it for current acquisition. See
`legacy/README.md` for the provenance and limitations of each retired path.
