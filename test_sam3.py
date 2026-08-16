#!/usr/bin/env python3
"""
test_sam3.py — First smoke test: run SAM3 text-prompted segmentation on one
CAROECT-D sRGB frame (output of preprocess.py's --output-rgb / quick_tiff_to_sam3.py).

REWRITTEN — root-caused a real crash, not a guess
---------------------------------------------------
Old version used the single-IMAGE API: build_sam3_image_model() + Sam3Processor
.set_image(). Real run on this project's hardware crashed inside SAM3's own
vitdet.py backbone:
    RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float
Wrapping set_image() in `torch.autocast(..., enabled=False)` did NOT fix it (still
crashed the same way) — that only stops autocast from casting ACTIVATIONS; it
does nothing about the WEIGHTS, which build_sam3_image_model() loads as BFloat16.
The input tensor (from Sam3Processor.set_image(), built on PIL/to_tensor) stays
Float32 and never gets cast to match — a real dtype-handling gap in the
single-image processor path itself.

FIX: switch to the VIDEO API (build_sam3_video_predictor + init_state), which is
what sam3_video_to_labels.py already uses successfully with real project data.
Confirmed from SAM3's own sam3/model/io_utils.py: `load_resource_as_video_frames`
treats a single image path as a "single-frame video"
(`load_image_as_single_frame_video`) and explicitly does `images.unsqueeze(0)
.half()` before normalizing — the exact cast the image-API path was missing.
Same model, same weights, same prompt — just the code path that's proven to
handle dtype correctly. One SAM3 entry point across the whole project now,
instead of two divergent ones.

Usage:
    python test_sam3.py frame_0000.tiff --prompt "car" --output result.png
    python test_sam3.py frame_0000.tiff --prompt "person" --output result.png
"""

import argparse

import numpy as np
import cv2

from sam3.model_builder import build_sam3_video_predictor
from linear16_to_srgb8 import guard_reject_16bit_path


def main():
    ap = argparse.ArgumentParser(description="SAM3 text-prompted segmentation smoke test")
    ap.add_argument("image", help="Path to an 8-bit sRGB TIFF frame (from preprocess.py "
                                   "--output-rgb or quick_tiff_to_sam3.py)")
    ap.add_argument("--prompt", default="car", help="Text prompt, e.g. 'car', 'person', 'truck'")
    ap.add_argument("--output", default="sam3_result.png", help="Where to save the visualization")
    ap.add_argument("--score-threshold", type=float, default=0.3,
                    help="Only draw detections above this confidence (default 0.3)")
    args = ap.parse_args()

    # Fail loud BEFORE loading the checkpoint (cheap header check) if this is an
    # unconverted 16-bit TIFF straight from DaVinci — see linear16_to_srgb8.py
    # docstring for why letting that reach PIL's convert('RGB') is dangerous.
    guard_reject_16bit_path(args.image)

    print(f"Loading SAM3 video predictor (first run downloads checkpoint, may take a while)...")
    predictor = build_sam3_video_predictor()

    print(f"Loading image as a single-frame video: {args.image}")
    # A single image path is a valid `resource_path` for start_session — io_utils.py's
    # `load_resource_as_video_frames` detects the image extension and routes it through
    # `load_image_as_single_frame_video`, which does the float16 cast that the old
    # image-API path was missing. No frame-folder / numeric-naming needed for 1 frame.
    response = predictor.handle_request(
        request=dict(type="start_session", resource_path=args.image)
    )
    session_id = response["session_id"]

    print(f"Prompting frame 0 with text: '{args.prompt}'")
    response = predictor.handle_request(
        request=dict(type="add_prompt", session_id=session_id, frame_index=0, text=args.prompt)
    )
    out = response["outputs"]

    obj_ids = out["out_obj_ids"]
    scores = out["out_probs"]
    boxes = out["out_boxes_xywh"]          # normalized [cx, cy, w, h] — see sam3_video_to_labels.py caveat
    masks = out.get("out_binary_masks")

    n_total = len(scores)
    keep = [i for i in range(n_total) if scores[i] >= args.score_threshold]
    print(f"\nFound {n_total} instance(s) matching '{args.prompt}', "
          f"{len(keep)} above score threshold {args.score_threshold}")
    for i in keep:
        print(f"  instance {i}: score={scores[i]:.3f}  box(cxcywh,norm)={boxes[i]}")

    predictor.handle_request(request=dict(type="close_session", session_id=session_id))
    predictor.shutdown()

    # ── Visualization: overlay boxes + mask silhouettes on the original image ──
    img_bgr = cv2.imread(args.image)   # cv2 reads 8-bit TIFF fine
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read {args.image} with cv2 — check it's a valid "
                                 "8-bit TIFF (uint16 TIFF straight from DaVinci will NOT work here).")
    h, w = img_bgr.shape[:2]
    overlay = img_bgr.copy()
    rng = np.random.default_rng(0)

    for i in keep:
        color = tuple(int(c) for c in rng.integers(64, 255, 3))
        if masks is not None and i < len(masks):
            m = np.asarray(masks[i]).astype(bool)
            overlay[m] = (overlay[m] * 0.5 + np.array(color) * 0.5).astype(np.uint8)

        cx, cy, bw, bh = boxes[i]
        x0 = int((cx - bw / 2) * w)
        y0 = int((cy - bh / 2) * h)
        x1 = int((cx + bw / 2) * w)
        y1 = int((cy + bh / 2) * h)
        cv2.rectangle(img_bgr, (x0, y0), (x1, y1), color, 2)
        cv2.putText(img_bgr, f"{args.prompt} {scores[i]:.2f}", (x0, max(0, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    blended = cv2.addWeighted(img_bgr, 0.6, overlay, 0.4, 0)
    cv2.imwrite(args.output, blended)
    print(f"\nSaved visualization -> {args.output}")


if __name__ == "__main__":
    main()
