# CAROECT-D Pipeline

Pipeline nay bam dung `Paper/CAROECT-D_Methodology.md`: moi xu ly vat ly chay tren 16-bit scene-linear, undistort/resize dung mot lan tren mang chung, roi moi fork thanh nhanh SAM sRGB va nhanh event linear.

## Setup tren Ubuntu ThinkPad

```bash
python3 -m pip install -r requirements.txt
./setup_sim.sh
```

Neu dung LUCID Triton2 EVS, cai Arena SDK truoc roi build recorder:

```bash
./build_linux.sh /path/to/ArenaSDK
./evs_recorder --list-event-formats
```

## Calibration anh

Dat file calibration vao:

```text
calibration/dark/        dark frames, lens cap
calibration/flat/        flat-field frames
calibration/gray_card.tiff
calibration/chessboard/  checkerboard images
```

Chay:

```bash
python3 calibrate.py
```

Output duoc `preprocess.py` dung truc tiep: `dark_mean.npy`, `gain_map.npy`, `wb_gains.npy`, `camera_params.npz`.

## Chay synthetic pipeline

DaVinci export `.NEV` thanh RGB 16-bit TIFF linear. Dam bao so TIFF xap xi `clip_seconds * 119.88`, khong phai `29.98`.

```bash
./run_pipeline.sh sim data/tiff/site01 site01 train
```

Neu muon tinh them per-event mask stats de debug/visualize:

```bash
LABEL_STATS=1 ./run_pipeline.sh sim data/tiff/site01 site01 train
```

Lenh nay chay:

```text
preprocess.py
  -> data/processed/site01/      16-bit linear Y for simulators
  -> data/rgb/site01/            sRGB PNG for SAM
run_v2e.py                       deterministic threshold events
run_dvsvolt.py                   stochastic Brownian-voltage events
sam3_export_tracks.py            tracks.json + mask PNGs
label_transfer.py                windows for each simulator
build_event_dataset.py           YOLO event pseudo-images
```

`pipeline.sh` van ton tai nhu wrapper cu:

```bash
./pipeline.sh data/tiff/site01 site01 train
```

## Ghi event that

Sau khi confirm camera co XYPT:

```bash
./evs_recorder --output site01.cevt --event-format EVT3_0 --strict-xypt --flat-xypt --duration 60
./run_pipeline.sh real site01.cevt site01
```

Output:

```text
data/events_real/site01.h5
```

Schema H5 chung cho real/v2e/dvsvolt la `x`, `y`, `t` microseconds, `p` with `1=ON, 0=OFF`.

## Calibrate simulator voi event that

Chua co data thi dung config default. Khi co `events_real.h5` cung scene voi TIFF da preprocess, tune tung param:

```bash
./run_pipeline.sh calibrate data/events_real/site01.h5 data/processed/site01 v2e pos_thres 0.15 0.20 0.25 0.30
./run_pipeline.sh calibrate data/events_real/site01.h5 data/processed/site01 dvsvolt k1 3.0 4.0 5.3 6.5 8.0
```

Script se backup `config.yaml.bak` roi ghi gia tri tot nhat vao `config.yaml`.

## Train va eval

```bash
./run_pipeline.sh train data/dataset
./run_pipeline.sh eval runs/caroectd/exp/weights/best.pt data/dataset/data.yaml
```

So sanh domain-transfer calibrated vs uncalibrated:

```bash
./run_pipeline.sh eval runs/calibrated/weights/best.pt data/real_test/data.yaml runs/uncalibrated/weights/best.pt
```

## File chinh

- `preprocess.py`: giu 16-bit linear, corrections, undistort/resize mot lan, fork sRGB/Y.
- `run_v2e.py`: goi v2e nhu library, giu sub-8bit precision bang float 0..255.
- `run_dvsvolt.py`: goi DVS-Voltmeter nhu library, giu thang DN dung cho `k1..k6`.
- `sam3_export_tracks.py`: export multi-class tracks, boxes, masks, timestamps.
- `label_transfer.py`: identity geometry plus shared-clock windowing, mask-aware per-event stats.
- `build_event_dataset.py`: render event windows thanh anh 3 kenh YOLO.
- `evs_recorder.cpp` + `xypt_to_h5.py`: duong real event qua Arena SDK va flat XYPT.
