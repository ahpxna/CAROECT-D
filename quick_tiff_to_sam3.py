#!/usr/bin/env python3
"""
Quick controlled conversion from linear uint16 TIFF to SAM3 annotation TIFF.

Why TIFF rather than JPEG
-------------------------
SAM3's official image loader accepts TIFF through Pillow. An RGB uint8 TIFF is
decoded without an additional lossy DCT stage, whereas JPEG would add block
quantization after the unavoidable controlled 16-to-8-bit annotation mapping.
Sequential integer filenames also preserve SAM3's expected temporal sort.

This is still not the full preprocessing pipeline. It performs only linear
working-primary conversion plus the sRGB transfer. It intentionally skips
validated radiometric corrections, undistortion, shared resizing, and optional
stabilization. Use preprocess.py --output-rgb for official dataset production;
use this utility only for a quick SAM3 sanity check.

Usage:
  python quick_tiff_to_sam3.py input_tiffs --output-dir quick_run --prompt car
  python quick_tiff_to_sam3.py input_tiffs --output-dir frames --tiff-only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
import yaml
import json

from linear16_to_srgb8 import linear_working_to_srgb_u8, guard_reject_16bit


# ══════════════════════════════════════════════════════════════════
#  STEP 1 — controlled linear gamut conversion + sRGB transfer
# ══════════════════════════════════════════════════════════════════
#  The shared helper is also used by preprocess.py, preventing colour drift.


def convert_tiffs_to_tiff8(in_dir: Path, frames_dir: Path, limit,
                           working_primaries: str) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    tiffs = sorted(set(list(in_dir.glob("*.tif")) + list(in_dir.glob("*.tiff"))))
    if not tiffs:
        raise FileNotFoundError(f"No .tif/.tiff files found in {in_dir}")
    if limit:
        tiffs = tiffs[:limit]

    print(f"[convert] {len(tiffs)} frame(s)  {in_dir} -> {frames_dir}  (TIFF 16-bit -> TIFF 8-bit)")
    print("  Quick colour conversion only: no corrections or geometry. "
          "Use preprocess.py --output-rgb for official datasets.\n")

    out_paths = []
    for i, p in enumerate(tiffs):
        img = tifffile.imread(str(p))
        if img.dtype != np.uint16 or img.ndim != 3 or img.shape[2] != 3:
            print(f"  [skip] {p.name}: unexpected shape/dtype {img.shape}/{img.dtype} "
                  "(expected a linear uint16 HxWx3 RGB TIFF)")
            continue

        srgb = linear_working_to_srgb_u8(
            img.astype(np.float32), working_primaries)
        # Sequential integer naming (0.tiff, 1.tiff, ...) — SAM3's io_utils.py sorts
        # Integer names preserve temporal order; lexicographic sorting would
        # otherwise place 10.tiff before 9.tiff.
        out_path = frames_dir / f"{len(out_paths)}.tiff"
        tifffile.imwrite(str(out_path), srgb, photometric="rgb")
        out_paths.append(out_path)

        if i < 5 or (i + 1) % 25 == 0:
            print(f"  [{i+1:>4}/{len(tiffs)}] {out_path.name}")

    print(f"\n[convert] Done. {len(out_paths)} TIFF(s) in {frames_dir}")
    return out_paths


# ══════════════════════════════════════════════════════════════════
#  STEP 2 — reuse the verified SAM3 request/overlay implementation
# ══════════════════════════════════════════════════════════════════

def run_sam3_real(frames_dir: Path, out_dir: Path, prompt: str,
                   score_threshold: float, class_id: int):
    try:
        from sam3_video_to_labels import (
            propagate_in_video, load_sorted_frame_paths, draw_overlay,
            write_yolo_label,
        )
        from sam3.model_builder import build_sam3_video_predictor
    except ImportError as e:
        raise ImportError(
            "Could not import SAM3 or sam3_video_to_labels.py. Install the "
            f"official facebookresearch/sam3 package. Original error: {e}") from e

    labels_dir = out_dir / "labels"
    overlay_dir = out_dir / "overlay"
    labels_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[sam3] Loading SAM3 video predictor (first run downloads checkpoint)...")
    predictor = build_sam3_video_predictor()

    frame_paths = load_sorted_frame_paths(str(frames_dir))
    if not frame_paths:
        raise FileNotFoundError(f"No .tif/.tiff frames found in {frames_dir}")
    print(f"[sam3] {len(frame_paths)} frame(s): {Path(frame_paths[0]).name} .. "
          f"{Path(frame_paths[-1]).name}")

    response = predictor.handle_request(
        request=dict(type="start_session", resource_path=str(frames_dir))
    )
    session_id = response["session_id"]
    print(f"[sam3] Session started: {session_id}")

    print(f"[sam3] Prompting frame 0 with text: '{prompt}'")
    response = predictor.handle_request(
        request=dict(type="add_prompt", session_id=session_id,
                     frame_index=0, text=prompt)
    )
    frame0_out = response["outputs"]

    print("[sam3] Propagating across all frames...")
    outputs_per_frame = propagate_in_video(predictor, session_id, "both")
    outputs_per_frame[0] = outputs_per_frame.get(0, frame0_out)

    print(f"[sam3] Got outputs for {len(outputs_per_frame)} frame(s). "
          "Writing labels + overlays...")

    import cv2  # cv2.imread/imwrite handle 8-bit TIFF fine; used only for overlay drawing
    for frame_idx in sorted(outputs_per_frame.keys()):
        out = outputs_per_frame[frame_idx]
        obj_ids = out["out_obj_ids"]
        scores = out["out_probs"]
        boxes = out["out_boxes_xywh"]
        masks = out.get("out_binary_masks")

        img_bgr = cv2.imread(frame_paths[frame_idx])
        if img_bgr is None:
            print(f"  [skip] frame {frame_idx}: could not read {frame_paths[frame_idx]}")
            continue

        blended, kept = draw_overlay(img_bgr, boxes, masks, obj_ids, scores,
                                      prompt, score_threshold)
        cv2.imwrite(str(overlay_dir / f"{frame_idx}.png"), blended)
        write_yolo_label(labels_dir / f"{frame_idx}.txt", kept, class_id=class_id)

        if frame_idx < 5 or frame_idx % 25 == 0:
            print(f"  frame {frame_idx:>3}: {len(kept)}/{len(obj_ids)} object(s) "
                  f">= {score_threshold} score")

    predictor.handle_request(request=dict(type="close_session", session_id=session_id))
    predictor.shutdown()

    print("\n✓ Real SAM3 inference complete.")
    print(f"  YOLO labels -> {labels_dir}/")
    print(f"  Overlays    -> {overlay_dir}/  (inspect several box placements)")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Linear uint16 TIFF -> controlled uint8 TIFF -> SAM3")
    ap.add_argument("input", help="Folder of 16-bit RGB TIFFs (DaVinci export)")
    ap.add_argument("--output-dir", required=True,
                    help="Root output: <output-dir>/frames, /labels, /overlay")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only convert the first N frames (default: all)")
    ap.add_argument("--prompt", default="car", help="SAM3 text prompt, e.g. 'car' or 'person'")
    ap.add_argument("--score-threshold", type=float, default=0.3)
    ap.add_argument("--class-id", type=int, default=0, help="YOLO class id for this prompt")
    ap.add_argument("--tiff-only", action="store_true",
                    help="Convert to annotation TIFF only; do not run SAM3")
    ap.add_argument("--config", default="config.yaml",
                    help="Configuration declaring camera.working_primaries")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output_dir)
    frames_dir = out_dir / "frames"

    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    working_primaries = config["camera"]["working_primaries"]
    frame_paths = convert_tiffs_to_tiff8(
        in_dir, frames_dir, args.limit, working_primaries)
    (frames_dir / "conversion_metadata.json").write_text(json.dumps({
        "source_primaries": working_primaries,
        "destination_primaries": "srgb",
        "destination_transfer": "IEC 61966-2-1 sRGB",
    }, indent=2))
    if not frame_paths:
        print("No frame could be converted.", file=sys.stderr)
        sys.exit(1)

    if args.tiff_only:
        print(f"\n--tiff-only completed. Run SAM3 manually with:")
        print(f"  python sam3_video_to_labels.py {frames_dir} --prompt \"{args.prompt}\" "
              f"--output-dir {out_dir}")
        return

    run_sam3_real(frames_dir, out_dir, args.prompt, args.score_threshold, args.class_id)


if __name__ == "__main__":
    main()
