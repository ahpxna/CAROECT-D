#!/usr/bin/env python3
"""
quick_tiff_to_jpg.py — TIFF -> JPG -> SAM3 THẬT (end-to-end, chạy model thật)
==============================================================================

LỊCH SỬ FILE NÀY (đọc để hiểu vì sao có 2 chế độ)
--------------------------------------------------
Bản gốc chỉ làm 1 việc: TIFF -> JPG, KHÔNG đụng vào SAM3, để người dùng tự
chạy test_sam3.py/sam3_video_to_labels.py sau. Giờ file này làm THẬT — tự
convert RỒI tự gọi SAM3 video predictor thật (propagate_in_video, đúng
pattern trong sam3_video_to_labels.py — không viết lại logic đó, IMPORT lại)
để ra labels + overlay ngay trong 1 lệnh. Đây là default. Cờ --jpg-only giữ
hành vi CŨ (chỉ convert, không đụng SAM3) cho ai chỉ cần JPG để tự xử lý.

⚠ VẪN KHÔNG PHẢI preprocess.py. Bước convert ở đây CHỈ áp sRGB encode, bỏ
qua dark/flat/WB/exposure/undistort/resize/stabilize — ảnh vào SAM3 sẽ méo
lens, sai màu, sai exposure so với dataset thật. Dùng preprocess.py --output-
rgb cho pipeline chính thức; file này để chạy được SAM3 NGAY hôm nay, không
chờ calibrate xong (đúng mục tiêu gốc: "SAM3 detect được gì không, không
chờ hardware/calib").

Usage:
    # Mặc định: convert + chạy SAM3 thật + xuất labels/overlay
    python quick_tiff_to_jpg.py /media/duolu/data_ssd1/TIFF/Export \\
        --output-dir /media/duolu/data_ssd1/quick_run --prompt "car" --limit 60

    # Chỉ convert (hành vi cũ), không đụng SAM3
    python quick_tiff_to_jpg.py /media/duolu/data_ssd1/TIFF/Export \\
        --output-dir /media/duolu/data_ssd1/quick_jpg --jpg-only --limit 30
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
import cv2


# ══════════════════════════════════════════════════════════════════
#  BƯỚC 1 — TIFF 16-bit -> JPG 8-bit (sRGB encode ONLY, xem cảnh báo trên)
# ══════════════════════════════════════════════════════════════════

def encode_srgb_u8(rgb_lin: np.ndarray) -> np.ndarray:
    """Same IEC 61966-2-1 curve as preprocess.py's N7a — the ONLY correction
    applied here. rgb_lin expected in 0..65535 scale (raw uint16 TIFF range,
    treated as already-linear, matching the project's input_transfer=linear
    default)."""
    n = np.clip(rgb_lin / 65535.0, 0.0, 1.0)
    s = np.where(n <= 0.0031308, n * 12.92, 1.055 * np.power(n, 1.0 / 2.4) - 0.055)
    return np.clip(np.rint(s * 255.0), 0, 255).astype(np.uint8)


def convert_tiffs_to_jpg(in_dir: Path, frames_dir: Path, limit, quality: int) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    tiffs = sorted(set(list(in_dir.glob("*.tif")) + list(in_dir.glob("*.tiff"))))
    if not tiffs:
        raise FileNotFoundError(f"No .tif/.tiff files found in {in_dir}")
    if limit:
        tiffs = tiffs[:limit]

    print(f"[convert] {len(tiffs)} frame(s)  {in_dir} -> {frames_dir}")
    print("  sRGB encode ONLY — no dark/flat/WB/exposure/undistort/resize. "
          "Dùng preprocess.py --output-rgb cho dataset chính thức.\n")

    out_paths = []
    for i, p in enumerate(tiffs):
        img = tifffile.imread(str(p))
        if img.dtype != np.uint16 or img.ndim != 3 or img.shape[2] != 3:
            print(f"  [skip] {p.name}: unexpected shape/dtype {img.shape}/{img.dtype}")
            continue

        srgb = encode_srgb_u8(img.astype(np.float32))
        # Sequential integer naming (0.jpg, 1.jpg, ...) — SAM3's video-frame
        # loader sorts numerically on exactly this pattern; any other naming
        # falls back to lexicographic sort, which can silently scramble frame
        # order (see sam3_video_to_labels.load_sorted_frame_paths).
        out_path = frames_dir / f"{len(out_paths)}.jpg"
        cv2.imwrite(str(out_path), srgb[:, :, ::-1],  # RGB -> BGR for cv2
                    [cv2.IMWRITE_JPEG_QUALITY, quality])
        out_paths.append(out_path)

        if i < 5 or (i + 1) % 25 == 0:
            print(f"  [{i+1:>4}/{len(tiffs)}] {out_path.name}")

    print(f"\n[convert] Done. {len(out_paths)} JPG(s) in {frames_dir}")
    return out_paths


# ══════════════════════════════════════════════════════════════════
#  BƯỚC 2 — SAM3 THẬT (import lại logic đã verify trong
#  sam3_video_to_labels.py — KHÔNG viết lại, tránh code trùng/lệch nhau)
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
            "Không import được SAM3 / sam3_video_to_labels.py. Kiểm tra: (1) đang chạy "
            "trong đúng env đã cài `sam3` package chưa (pip install từ "
            "facebookresearch/sam3), (2) file sam3_video_to_labels.py có nằm cùng thư "
            f"mục với quick_tiff_to_jpg.py không.\nLỗi gốc: {e}") from e

    labels_dir = out_dir / "labels"
    overlay_dir = out_dir / "overlay"
    labels_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[sam3] Loading SAM3 video predictor (first run downloads checkpoint)...")
    predictor = build_sam3_video_predictor()

    frame_paths = load_sorted_frame_paths(str(frames_dir))
    if not frame_paths:
        raise FileNotFoundError(f"No .jpg frames found in {frames_dir}")
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
    outputs_per_frame = propagate_in_video(predictor, session_id)
    outputs_per_frame[0] = outputs_per_frame.get(0, frame0_out)

    print(f"[sam3] Got outputs for {len(outputs_per_frame)} frame(s). "
          "Writing labels + overlays...")

    import cv2 as _cv2  # local import to keep top-level import list minimal for --jpg-only path
    for frame_idx in sorted(outputs_per_frame.keys()):
        out = outputs_per_frame[frame_idx]
        obj_ids = out["out_obj_ids"]
        scores = out["out_probs"]
        boxes = out["out_boxes_xywh"]
        masks = out.get("out_binary_masks")

        img_bgr = _cv2.imread(frame_paths[frame_idx])
        if img_bgr is None:
            print(f"  [skip] frame {frame_idx}: could not read {frame_paths[frame_idx]}")
            continue

        blended, kept = draw_overlay(img_bgr, boxes, masks, obj_ids, scores,
                                      prompt, score_threshold)
        _cv2.imwrite(str(overlay_dir / f"{frame_idx}.png"), blended)
        write_yolo_label(labels_dir / f"{frame_idx}.txt", kept, class_id=class_id)

        if frame_idx < 5 or frame_idx % 25 == 0:
            print(f"  frame {frame_idx:>3}: {len(kept)}/{len(obj_ids)} object(s) "
                  f">= {score_threshold} score")

    predictor.handle_request(request=dict(type="close_session", session_id=session_id))
    predictor.shutdown()

    print(f"\n✓ SAM3 THẬT xong.")
    print(f"  YOLO labels -> {labels_dir}/")
    print(f"  Overlays    -> {overlay_dir}/  (mở vài file .png, xem box có bám đúng xe không)")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="TIFF -> JPG -> SAM3 thật (default) hoặc chỉ convert (--jpg-only)")
    ap.add_argument("input", help="Folder of 16-bit RGB TIFFs (DaVinci export)")
    ap.add_argument("--output-dir", required=True,
                    help="Root output: <output-dir>/frames, /labels, /overlay")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only convert the first N frames (default: all)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality (default 95)")
    ap.add_argument("--prompt", default="car", help="Text prompt cho SAM3, vd 'car', 'person'")
    ap.add_argument("--score-threshold", type=float, default=0.3)
    ap.add_argument("--class-id", type=int, default=0, help="YOLO class id cho prompt này")
    ap.add_argument("--jpg-only", action="store_true",
                    help="Hành vi CŨ: chỉ convert TIFF->JPG, không gọi SAM3")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output_dir)
    frames_dir = out_dir / "frames"

    frame_paths = convert_tiffs_to_jpg(in_dir, frames_dir, args.limit, args.quality)
    if not frame_paths:
        print("Không có frame nào convert được — dừng.", file=sys.stderr)
        sys.exit(1)

    if args.jpg_only:
        print(f"\n--jpg-only: dừng ở bước convert. Chạy SAM3 thủ công bằng:")
        print(f"  python sam3_video_to_labels.py {frames_dir} --prompt \"{args.prompt}\" "
              f"--output-dir {out_dir}")
        return

    run_sam3_real(frames_dir, out_dir, args.prompt, args.score_threshold, args.class_id)


if __name__ == "__main__":
    main()
