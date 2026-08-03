#!/usr/bin/env python3
"""
test_sam3.py — First smoke test: run SAM3 text-prompted segmentation on one
CAROECT-D sRGB frame (output of preprocess.py's --output-rgb branch).

Verified against the OFFICIAL API in facebookresearch/sam3's README
(fetched 2026-07-xx) — not guessed.

Usage:
    python test_sam3.py frame_0000.png --prompt "car" --output result.png
    python test_sam3.py frame_0000.png --prompt "person" --output result.png
"""

import argparse
import numpy as np
import cv2
from PIL import Image

import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def main():
    ap = argparse.ArgumentParser(description="SAM3 text-prompted segmentation smoke test")
    ap.add_argument("image", help="Path to an sRGB PNG frame (from preprocess.py --output-rgb)")
    ap.add_argument("--prompt", default="car", help="Text prompt, e.g. 'car', 'person', 'truck'")
    ap.add_argument("--output", default="sam3_result.png", help="Where to save the visualization")
    ap.add_argument("--score-threshold", type=float, default=0.3,
                    help="Only draw detections above this confidence (default 0.3)")
    args = ap.parse_args()

    print(f"Loading SAM3 model (first run downloads checkpoint, may take a while)...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    print(f"Loading image: {args.image}")
    image = Image.open(args.image).convert("RGB")
    inference_state = processor.set_image(image)

    print(f"Prompting with text: '{args.prompt}'")
    output = processor.set_text_prompt(state=inference_state, prompt=args.prompt)

    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    n_total = len(scores)
    keep = [i for i, s in enumerate(scores) if s >= args.score_threshold]
    print(f"\nFound {n_total} instance(s) matching '{args.prompt}', "
          f"{len(keep)} above score threshold {args.score_threshold}")
    for i in keep:
        print(f"  instance {i}: score={scores[i]:.3f}  box={boxes[i]}")

    # ── Visualization: overlay boxes + mask silhouettes on the original image ──
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    overlay = img_bgr.copy()
    rng = np.random.default_rng(0)

    for i in keep:
        color = tuple(int(c) for c in rng.integers(64, 255, 3))
        mask = np.array(masks[i]).astype(bool)
        overlay[mask] = (overlay[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)

        x0, y0, x1, y1 = [int(v) for v in boxes[i]]
        cv2.rectangle(img_bgr, (x0, y0), (x1, y1), color, 2)
        cv2.putText(img_bgr, f"{args.prompt} {scores[i]:.2f}", (x0, max(0, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    blended = cv2.addWeighted(img_bgr, 0.6, overlay, 0.4, 0)
    cv2.imwrite(args.output, blended)
    print(f"\nSaved visualization -> {args.output}")


if __name__ == "__main__":
    main()
