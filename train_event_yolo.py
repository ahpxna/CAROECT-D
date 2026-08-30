#!/usr/bin/env python3
"""
Train YOLOv8 on the CAROECT-D three-channel event representation.

This is a thin Ultralytics wrapper, not a forked training loop. YOLOv8 is the
most reproducible image-based baseline among the event benchmarks considered.
The input channels encode ON count, OFF count, and last-event time rather than
photographic RGB, so hue/saturation/value augmentation is disabled.

Scene, view, and lighting policies are expressed by separate dataset manifests.
This trainer consumes exactly the supplied data.yaml and copies its
dataset_manifest.json into the run directory so every checkpoint is traceable.
Event-native RVT/RED experiments would require a different voxel representation
and are outside this wrapper.

Usage:
  python train_event_yolo.py --data dataset/data.yaml --epochs 100 \
      --imgsz 640 --batch 16 --model yolov8n.pt --project runs --name caroectd_v1
"""

import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="data.yaml produced by build_event_dataset.py")
    ap.add_argument("--model", default="yolov8n.pt",
                    help="Initial COCO checkpoint, or a model YAML for training from scratch")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42, help="Reproducibility seed")
    ap.add_argument("--project", default="runs/caroectd")
    ap.add_argument("--name", default="exp")
    ap.add_argument("--device", default=None, help="GPU index, cpu, or omit for automatic selection")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--patience", type=int, default=30, help="Early-stop patience (epochs)")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("pip install ultralytics")

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} does not exist; build the required dataset splits first.")
    manifest_path = data_path.parent / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} is required so every checkpoint remains traceable "
            "to one simulator/condition/window dataset.")

    print(f"{'='*64}\nTrain YOLO on the event representation\n"
          f"  data   : {data_path}\n  model  : {args.model}\n"
          f"  epochs : {args.epochs}  imgsz={args.imgsz}  batch={args.batch}  seed={args.seed}\n"
          f"{'='*64}\n")

    model = YOLO(args.model)
    model.train(
        data=str(data_path), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        seed=args.seed, project=args.project, name=args.name, resume=args.resume,
        patience=args.patience, device=args.device,
        # These are ON/OFF/time channels, not photographic RGB. Colour
        # augmentation would alter polarity semantics, so it is disabled.
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    )
    save_dir = Path(model.trainer.save_dir)
    shutil.copy2(manifest_path, save_dir / "dataset_manifest.json")
    print(f"\n✓ Complete. Weights -> {args.project}/{args.name}/weights/best.pt")
    print(f"  Evaluate: python eval_event_yolo.py --weights {args.project}/{args.name}/weights/best.pt "
          f"--data <test_data.yaml>")


if __name__ == "__main__":
    main()
