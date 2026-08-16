#!/usr/bin/env python3
"""
quick_tiff_to_sam3.py — TIFF 16-bit -> TIFF 8-bit -> SAM3 THẬT (end-to-end)
==============================================================================
(Trước đây tên quick_tiff_to_jpg.py, xuất .jpg. ĐỔI HẲN sang TIFF 8-bit — xem
lý do dưới. File .jpg cũ đã xóa, không còn dùng nữa.)

TẠI SAO BỎ JPG, DÙNG TIFF 8-BIT
--------------------------------
SAM3 tự khai trong io_utils.py của chính nó (IMAGE_EXTS = [".jpg", ".jpeg",
".png", ".bmp", ".tiff", ".webp"]) — .tiff đã nằm sẵn trong danh sách được
chấp nhận, và hàm decode (_load_img_as_tensor) dùng PIL: Image.open(...)
.convert("RGB") — mở TIFF 8-bit RGB thì convert("RGB") gần như no-op vì đã
đúng mode rồi. Vậy JPG chỉ tổ THÊM một lớp nén có mất (quantization DCT 8x8
block) CHỒNG LÊN lượng tử hóa sRGB 16→8-bit đã có sẵn — hai lớp mất mát dồn
lại, không cần thiết. TIFF 8-bit lossless: chỉ mất đúng 1 lần (16→8-bit ở
encode_srgb_u8, không tránh được vì SAM3 cần 8-bit), không mất thêm lần nào
ở bước ghi file.

VẪN CÙNG "SCALE 16-BIT XUỐNG 8-BIT" NHƯ TRƯỚC — không có gì đổi ở bước này
(encode_srgb_u8: /65535 → sRGB curve → *255 → round → uint8). Cái đổi CHỈ
là container ghi ra đĩa: TIFF thay vì JPG.

⚠ VẪN KHÔNG PHẢI preprocess.py. Bước convert ở đây CHỈ áp sRGB encode, bỏ
qua dark/flat/WB/exposure/undistort/resize/stabilize — ảnh vào SAM3 sẽ méo
lens, sai màu, sai exposure so với dataset thật. Dùng preprocess.py --output-
rgb cho pipeline chính thức (từ giờ preprocess.py cũng xuất TIFF 8-bit, cùng
định dạng với file này — xem preprocess.py N7a). File này để chạy được SAM3
NGAY hôm nay, không chờ calibrate xong.

Usage:
    # Mặc định: convert + chạy SAM3 thật + xuất labels/overlay
    python quick_tiff_to_sam3.py /media/duolu/data_ssd1/TIFF/Export \\
        --output-dir /media/duolu/data_ssd1/quick_run --prompt "car" --limit 60

    # Chỉ convert (hành vi cũ của quick_tiff_to_jpg.py), không đụng SAM3
    python quick_tiff_to_sam3.py /media/duolu/data_ssd1/TIFF/Export \\
        --output-dir /media/duolu/data_ssd1/quick_tiff8 --tiff-only --limit 30
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile

from linear16_to_srgb8 import encode_srgb_u8, guard_reject_16bit


# ══════════════════════════════════════════════════════════════════
#  BƯỚC 1 — TIFF 16-bit -> TIFF 8-bit (sRGB encode ONLY, xem cảnh báo trên)
# ══════════════════════════════════════════════════════════════════
#  encode_srgb_u8() giờ import từ linear16_to_srgb8.py — cùng 1 hàm với
#  preprocess.py N7a, không còn copy-paste 2 nơi (xem docstring module đó).


def convert_tiffs_to_tiff8(in_dir: Path, frames_dir: Path, limit) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    tiffs = sorted(set(list(in_dir.glob("*.tif")) + list(in_dir.glob("*.tiff"))))
    if not tiffs:
        raise FileNotFoundError(f"No .tif/.tiff files found in {in_dir}")
    if limit:
        tiffs = tiffs[:limit]

    print(f"[convert] {len(tiffs)} frame(s)  {in_dir} -> {frames_dir}  (TIFF 16-bit -> TIFF 8-bit)")
    print("  sRGB encode ONLY — no dark/flat/WB/exposure/undistort/resize. "
          "Dùng preprocess.py --output-rgb cho dataset chính thức.\n")

    out_paths = []
    for i, p in enumerate(tiffs):
        img = tifffile.imread(str(p))
        if img.dtype != np.uint16 or img.ndim != 3 or img.shape[2] != 3:
            print(f"  [skip] {p.name}: unexpected shape/dtype {img.shape}/{img.dtype} "
                  "(cần TIFF 16-bit HxWx3 RGB — output của DaVinci, không phải file đã "
                  "8-bit từ lần convert trước)")
            continue

        srgb = encode_srgb_u8(img.astype(np.float32))
        # Sequential integer naming (0.tiff, 1.tiff, ...) — SAM3's io_utils.py sorts
        # frame_names bằng int(os.path.splitext(p)[0]); tên khác sẽ fallback về
        # lexicographic sort và ĐẢO LỘN thứ tự frame mà không báo lỗi gì (vd 10.tiff
        # đứng trước 9.tiff theo alphabet).
        out_path = frames_dir / f"{len(out_paths)}.tiff"
        tifffile.imwrite(str(out_path), srgb, photometric="rgb")
        out_paths.append(out_path)

        if i < 5 or (i + 1) % 25 == 0:
            print(f"  [{i+1:>4}/{len(tiffs)}] {out_path.name}")

    print(f"\n[convert] Done. {len(out_paths)} TIFF(s) in {frames_dir}")
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
            f"mục với quick_tiff_to_sam3.py không.\nLỗi gốc: {e}") from e

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
    outputs_per_frame = propagate_in_video(predictor, session_id)
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

    print(f"\n✓ SAM3 THẬT xong.")
    print(f"  YOLO labels -> {labels_dir}/")
    print(f"  Overlays    -> {overlay_dir}/  (mở vài file .png, xem box có bám đúng xe không)")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="TIFF 16-bit -> TIFF 8-bit -> SAM3 thật (default) hoặc chỉ convert (--tiff-only)")
    ap.add_argument("input", help="Folder of 16-bit RGB TIFFs (DaVinci export)")
    ap.add_argument("--output-dir", required=True,
                    help="Root output: <output-dir>/frames, /labels, /overlay")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only convert the first N frames (default: all)")
    ap.add_argument("--prompt", default="car", help="Text prompt cho SAM3, vd 'car', 'person'")
    ap.add_argument("--score-threshold", type=float, default=0.3)
    ap.add_argument("--class-id", type=int, default=0, help="YOLO class id cho prompt này")
    ap.add_argument("--tiff-only", action="store_true",
                    help="Chỉ convert TIFF 16-bit -> TIFF 8-bit, không gọi SAM3 "
                         "(tương đương --jpg-only của bản cũ, giờ ra .tiff)")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output_dir)
    frames_dir = out_dir / "frames"

    frame_paths = convert_tiffs_to_tiff8(in_dir, frames_dir, args.limit)
    if not frame_paths:
        print("Không có frame nào convert được — dừng.", file=sys.stderr)
        sys.exit(1)

    if args.tiff_only:
        print(f"\n--tiff-only: dừng ở bước convert. Chạy SAM3 thủ công bằng:")
        print(f"  python sam3_video_to_labels.py {frames_dir} --prompt \"{args.prompt}\" "
              f"--output-dir {out_dir}")
        return

    run_sam3_real(frames_dir, out_dir, args.prompt, args.score_threshold, args.class_id)


if __name__ == "__main__":
    main()
