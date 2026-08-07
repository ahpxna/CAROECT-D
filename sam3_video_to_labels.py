#!/usr/bin/env python3
"""
sam3_video_to_labels.py — Full SAM3 video pipeline: text prompt -> propagate
across all frames -> YOLO labels + overlay PNGs per frame.

This fills in the ONE missing step from the user's original sam3_test.py:
add_prompt() alone only returns frame 0's result. propagate_in_video() (the
official pattern, confirmed from facebookresearch/sam3's own
sam3_video_predictor_example.ipynb) is what extends tracking across the rest
of the clip.

VERIFIED FACTS THIS SCRIPT RELIES ON (from the official notebook, not guessed):
  - predictor.handle_request(type="start_session", resource_path=folder)
    -> {"session_id": ...}. folder must contain "0.jpg","1.jpg",... (clean
    sequential integer names) - anything else triggers a lexicographic-sort
    fallback that can scramble true frame order.
  - predictor.handle_request(type="add_prompt", session_id=..., frame_index=0,
    text=...) -> {"outputs": {...}}  — frame 0 ONLY.
  - predictor.handle_stream_request(type="propagate_in_video", session_id=...)
    is a GENERATOR yielding one response per frame, each with
    response["frame_index"] and response["outputs"] — THIS is what covers
    frames 1..N.
  - Each frame's "outputs" dict contains (confirmed from real console output
    on this exact project's data): out_obj_ids, out_probs, out_boxes_xywh
    (already 0..1 NORMALIZED, matching COCO/DETR center-box convention
    [cx, cy, w, h] - SAM3's detector is DETR-based per its own documentation,
    so this is presented as the likely convention, NOT independently pixel-
    verified here - the overlay PNGs this script writes are the way to
    confirm boxes land correctly on real cars; if boxes look offset, the
    convention may be [x_min, y_min, w, h] instead and the box-drawing math
    below needs a one-line fix), out_binary_masks (H x W bool per object).

Usage:
    python sam3_video_to_labels.py /media/duolu/data_ssd1/quick_jpg \
        --prompt "car" --output-dir /media/duolu/data_ssd1/sam3_labels
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

from sam3.model_builder import build_sam3_video_predictor


def propagate_in_video(predictor, session_id):
    """Verbatim pattern from facebookresearch/sam3's own
    sam3_video_predictor_example.ipynb (Cell 10) - not our invention.
    Streams one response per frame; we collect them all into a dict."""
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


def load_sorted_frame_paths(folder: str):
    """Same loading/sorting logic as the official notebook's Cell 13 -
    integer sort on '<frame_index>.jpg', falling back to lexicographic sort
    (with a warning) if names don't match that pattern."""
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(glob.glob(os.path.join(folder, ext)))
    try:
        paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    except ValueError:
        print(f"[warning] frame names are not in '<frame_index>.jpg' format: "
              f"{paths[:5]}, falling back to lexicographic sort. "
              f"Frame order may not match true time order.")
        paths.sort()
    return paths


def draw_overlay(image_bgr, boxes_xywh_norm, masks, obj_ids, scores, prompt_label,
                  score_threshold):
    """Boxes assumed [cx, cy, w, h] normalized (DETR/COCO convention - see
    module docstring caveat). Masks assumed already at the image's own
    resolution (H, W) boolean arrays."""
    h, w = image_bgr.shape[:2]
    out = image_bgr.copy()
    overlay = out.copy()
    rng = np.random.default_rng(0)

    kept_boxes_px = []  # (obj_id, x_min, y_min, x_max, y_max) for YOLO export
    for i, obj_id in enumerate(obj_ids):
        score = float(scores[i])
        if score < score_threshold:
            continue

        color = tuple(int(c) for c in rng.integers(64, 255, 3))
        if masks is not None and i < len(masks):
            m = np.asarray(masks[i]).astype(bool)
            overlay[m] = (overlay[m] * 0.5 + np.array(color) * 0.5).astype(np.uint8)

        cx, cy, bw, bh = boxes_xywh_norm[i]
        x_min = int((cx - bw / 2) * w)
        y_min = int((cy - bh / 2) * h)
        x_max = int((cx + bw / 2) * w)
        y_max = int((cy + bh / 2) * h)
        cv2.rectangle(out, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(out, f"{prompt_label}#{obj_id} {score:.2f}", (x_min, max(0, y_min - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        kept_boxes_px.append((obj_id, x_min, y_min, x_max, y_max, cx, cy, bw, bh))

    blended = cv2.addWeighted(out, 0.6, overlay, 0.4, 0)
    return blended, kept_boxes_px


def write_yolo_label(txt_path, boxes_xywh_norm_kept, class_id=0):
    """YOLO format: class_id cx cy w h, ALL already 0..1 normalized - if
    out_boxes_xywh really is [cx,cy,w,h] normalized (see caveat above), this
    is a direct passthrough, no pixel math needed."""
    with open(txt_path, "w") as f:
        for (_obj_id, _x0, _y0, _x1, _y1, cx, cy, bw, bh) in boxes_xywh_norm_kept:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def main():
    ap = argparse.ArgumentParser(description="SAM3 video -> YOLO labels + overlays")
    ap.add_argument("video_folder", help="Folder of '0.jpg','1.jpg',... frames")
    ap.add_argument("--prompt", default="car", help="Text prompt (short noun phrase)")
    ap.add_argument("--output-dir", required=True, help="Where to write labels/ and overlay/")
    ap.add_argument("--score-threshold", type=float, default=0.3)
    ap.add_argument("--class-id", type=int, default=0, help="YOLO class id for this prompt")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    labels_dir = out_dir / "labels"
    overlay_dir = out_dir / "overlay"
    labels_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SAM3 video predictor (first run downloads checkpoint)...")
    predictor = build_sam3_video_predictor()

    frame_paths = load_sorted_frame_paths(args.video_folder)
    if not frame_paths:
        raise FileNotFoundError(f"No .jpg frames found in {args.video_folder}")
    print(f"Found {len(frame_paths)} frame(s): {Path(frame_paths[0]).name} .. "
          f"{Path(frame_paths[-1]).name}")

    response = predictor.handle_request(
        request=dict(type="start_session", resource_path=args.video_folder)
    )
    session_id = response["session_id"]
    print(f"Session started: {session_id}")

    print(f"Prompting frame 0 with text: '{args.prompt}'")
    response = predictor.handle_request(
        request=dict(type="add_prompt", session_id=session_id,
                     frame_index=0, text=args.prompt)
    )
    frame0_out = response["outputs"]

    print("Propagating across all frames (this is the step sam3_test.py was missing)...")
    outputs_per_frame = propagate_in_video(predictor, session_id)
    # frame 0 comes from add_prompt's own response; propagate_in_video covers
    # the rest, but also re-includes frame 0 in most SAM3 versions - if it's
    # present here too we let it overwrite frame0_out (should be identical).
    outputs_per_frame[0] = outputs_per_frame.get(0, frame0_out)

    print(f"Got outputs for {len(outputs_per_frame)} frame(s). Writing labels + overlays...")

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
                                      args.prompt, args.score_threshold)

        cv2.imwrite(str(overlay_dir / f"{frame_idx}.png"), blended)
        write_yolo_label(labels_dir / f"{frame_idx}.txt", kept, class_id=args.class_id)

        if frame_idx < 5 or frame_idx % 25 == 0:
            print(f"  frame {frame_idx:>3}: {len(kept)}/{len(obj_ids)} object(s) "
                  f">= {args.score_threshold} score")

    predictor.handle_request(request=dict(type="close_session", session_id=session_id))
    predictor.shutdown()

    print(f"\n✓ Done.")
    print(f"  YOLO labels -> {labels_dir}/  (one .txt per frame)")
    print(f"  Overlays    -> {overlay_dir}/  (one .png per frame, open a few to sanity-check)")
    print(f"\n  IMPORTANT: open a few overlay PNGs and check the boxes actually land on")
    print(f"  real cars. If they look shifted/offset, the box convention assumed in this")
    print(f"  script ([cx,cy,w,h] center-based) may be wrong for this SAM3 version -")
    print(f"  report back what you see and the box math gets a one-line fix.")


if __name__ == "__main__":
    main()
