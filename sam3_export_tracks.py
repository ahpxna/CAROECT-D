#!/usr/bin/env python3
"""
sam3_export_tracks.py — Chạy SAM3 video predictor, xuất tracks.json GIỮ TRACK_ID
xuyên suốt các frame kèm timestamp — mảnh còn thiếu để label_transfer.py hoạt động.

TẠI SAO FILE NÀY PHẢI TỒN TẠI (gap cụ thể trong pipeline cũ)
-------------------------------------------------------------
sam3_video_to_labels.py, với mỗi frame, ghi một .txt kiểu YOLO:
    class_id cx cy w h
write_yolo_label() KHÔNG nhận obj_id — track_id bị vứt bỏ ngay tại chỗ ghi file.

Hậu quả: label_transfer.py (script mới, xem file đó) cần biết "box ở frame i
và box ở frame i+1 có phải CÙNG MỘT xe hay không" để nội suy tuyến tính vị trí
tại đúng timestamp của từng event — đây chính là phương trình

    α = (t_k - t_i) / (t_{i+1} - t_i)
    B_j(t_k) = (1-α)·B_j(t_i) + α·B_j(t_{i+1})

đã thống nhất trong phần lý thuyết của project. KHÔNG có track_id -> không biết
"j" nào ứng với "j" nào giữa 2 frame -> không viết được phương trình trên.

File này KHÔNG viết lại logic SAM3 — import lại propagate_in_video(),
load_sorted_frame_paths(), draw_overlay(), write_yolo_label() từ
sam3_video_to_labels.py (đúng pattern quick_tiff_to_jpg.py đã dùng), chỉ THÊM
bước ghi tracks.json giữ đúng thông tin track_id + timestamp mà bước ghi YOLO
cũ đã làm mất.

OUTPUT SCHEMA (tracks.json)
----------------------------
{
  "fps": 119.88,                  # fps_original — PHẢI khớp config.yaml lúc quay
  "width": 1280, "height": 720,
  "prompt": "car",
  "tracks": {
    "<obj_id>": {
      "class_id": 0,
      "frames": [                 # CHỈ những frame track thật sự xuất hiện
        {"frame_idx": 0, "t_us": 0,    "cx":.., "cy":.., "w":.., "h":.., "score":..},
        {"frame_idx": 1, "t_us": 8342, "cx":.., "cy":.., "w":.., "h":.., "score":..},
        ...
      ]
    }
  }
}
cx,cy,w,h là NORMALIZED [0,1] (giống quy ước out_boxes_xywh của SAM3, xem
caveat trong sam3_video_to_labels.py — nếu box lệch khi vẽ overlay, quy ước
có thể là [x_min,y_min,w,h] chứ không phải center-based, sửa 1 dòng ở đó rồi
chạy lại file này). Khoảng trống trong "frames" (track biến mất do occlusion
rồi xuất hiện lại) được label_transfer.py tự phát hiện qua --max-gap-frames,
KHÔNG nội suy xuyên qua khoảng trống quá lớn.

Usage:
  python sam3_export_tracks.py frames_dir/ --prompt "car" --output-dir out/ \
      --config config.yaml --class-id 0 --also-yolo
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from sam3_video_to_labels import (
    propagate_in_video, load_sorted_frame_paths, draw_overlay, write_yolo_label,
)
from sam3.model_builder import build_sam3_video_predictor


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _as_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _mask_to_uint8(mask, height: int, width: int):
    m = _as_numpy(mask)
    if m.ndim == 3:
        m = np.squeeze(m)
    if m.shape != (height, width):
        m = cv2.resize(m.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8) * 255


def _xywh_to_xyxy(box: dict):
    cx, cy, w, h = box["cx"], box["cy"], box["w"], box["h"]
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def _box_iou(a: dict, b: dict) -> float:
    ax0, ay0, ax1, ay1 = _xywh_to_xyxy(a)
    bx0, by0, bx1, by1 = _xywh_to_xyxy(b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def merge_duplicate_observations(tracks: dict, iou_threshold: float) -> tuple[dict, int]:
    """Frame-wise high-IoU NMS across all SAM3 prompt sessions.

    SAM3 is run once per class prompt in this wrapper, so the same physical
    object can appear in multiple prompt sessions. We keep the highest-score
    observation per frame and drop lower-score duplicates; track IDs remain
    stable for the remaining observations.
    """
    by_frame: dict[int, list[tuple[str, int, dict]]] = {}
    for key, entry in tracks.items():
        for obs_idx, obs in enumerate(entry["frames"]):
            by_frame.setdefault(int(obs["frame_idx"]), []).append((key, obs_idx, obs))

    drop: set[tuple[str, int]] = set()
    for rows in by_frame.values():
        rows = sorted(rows, key=lambda r: float(r[2].get("score", 0.0)), reverse=True)
        kept: list[tuple[str, int, dict]] = []
        for key, obs_idx, obs in rows:
            duplicate = any(_box_iou(obs, kept_obs) >= iou_threshold for _, _, kept_obs in kept)
            if duplicate:
                drop.add((key, obs_idx))
            else:
                kept.append((key, obs_idx, obs))

    if not drop:
        return tracks, 0

    merged = {}
    for key, entry in tracks.items():
        frames = [obs for obs_idx, obs in enumerate(entry["frames"]) if (key, obs_idx) not in drop]
        if frames:
            kept_entry = dict(entry)
            kept_entry["frames"] = frames
            merged[key] = kept_entry
    return merged, len(drop)


def _run_one_prompt(predictor, video_folder: str, frame_paths: list[str], prompt: str,
                    class_id: int, fps: float, width: int, height: int,
                    score_threshold: float, out_dir: Path, also_yolo: bool):
    labels_dir, overlay_dir, masks_dir = out_dir / "labels", out_dir / "overlay", out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPrompt '{prompt}' (class_id={class_id})")
    response = predictor.handle_request(
        request=dict(type="start_session", resource_path=video_folder))
    session_id = response["session_id"]
    print(f"  Session started: {session_id}")

    response = predictor.handle_request(
        request=dict(type="add_prompt", session_id=session_id, frame_index=0, text=prompt))
    frame0_out = response["outputs"]

    outputs_per_frame = propagate_in_video(predictor, session_id)
    outputs_per_frame[0] = outputs_per_frame.get(0, frame0_out)

    tracks: dict = {}
    yolo_by_frame: dict[int, list] = {}
    for frame_idx in sorted(outputs_per_frame.keys()):
        out = outputs_per_frame[frame_idx]
        obj_ids, scores, boxes = out["out_obj_ids"], out["out_probs"], out["out_boxes_xywh"]
        masks = out.get("out_binary_masks")
        t_us = round(frame_idx * 1e6 / fps)

        img_bgr = cv2.imread(frame_paths[frame_idx]) if frame_idx < len(frame_paths) else None
        if also_yolo and img_bgr is not None:
            blended, kept = draw_overlay(img_bgr, boxes, masks, obj_ids, scores,
                                          prompt, score_threshold)
            cv2.imwrite(str(overlay_dir / f"{Path(frame_paths[frame_idx]).stem}_{class_id}.png"), blended)
            yolo_by_frame.setdefault(frame_idx, []).extend((class_id, row) for row in kept)

        n_kept = 0
        for i, obj_id in enumerate(obj_ids):
            score = float(scores[i])
            if score < score_threshold:
                continue
            n_kept += 1
            cx, cy, bw, bh = [float(v) for v in boxes[i]]
            raw_obj_id = int(obj_id)
            key = f"{class_id}_{raw_obj_id}"
            mask_rel = None
            if masks is not None and i < len(masks):
                mask_name = f"{frame_idx:06d}_c{class_id}_o{raw_obj_id}.png"
                mask_rel = f"masks/{mask_name}"
                cv2.imwrite(str(masks_dir / mask_name), _mask_to_uint8(masks[i], height, width))
            entry = tracks.setdefault(key, {
                "class_id": class_id,
                "class_name": prompt,
                "raw_obj_id": raw_obj_id,
                "frames": [],
            })
            entry["frames"].append({
                "frame_idx": int(frame_idx), "t_us": int(t_us),
                "cx": cx, "cy": cy, "w": bw, "h": bh,
                "score": score, "mask_path": mask_rel,
            })

        if frame_idx < 5 or frame_idx % 25 == 0:
            print(f"  frame {frame_idx:>4}: {n_kept}/{len(obj_ids)} object(s) kept")

    predictor.handle_request(request=dict(type="close_session", session_id=session_id))
    return tracks, yolo_by_frame


def main():
    ap = argparse.ArgumentParser(
        description="SAM3 video -> tracks.json (giữ track_id + timestamp, xem module docstring)")
    ap.add_argument("video_folder", help="Folder '0.jpg','1.jpg',... (output preprocess.py --output-rgb)")
    ap.add_argument("--prompt", default=None,
                    help="Một prompt đơn. Nếu bỏ trống và dùng --all-classes, lấy class list từ config.")
    ap.add_argument("--all-classes", action="store_true",
                    help="Chạy toàn bộ label_transfer.class_names trong config.yaml")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--score-threshold", type=float, default=0.3)
    ap.add_argument("--class-id", type=int, default=0)
    ap.add_argument("--also-yolo", action="store_true",
                    help="Cũng ghi labels/*.txt + overlay/*.png y hệt sam3_video_to_labels.py "
                         "(sanity-check bằng mắt song song với tracks.json)")
    ap.add_argument("--merge-iou", type=float, default=None,
                    help="NMS IoU threshold để gộp duplicate cross-prompt observations. "
                         "Mặc định lấy label_transfer.cross_class_iou hoặc 0.85.")
    ap.add_argument("--no-merge", action="store_true",
                    help="Tắt postprocess NMS cross-prompt.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    fps = float(cfg["camera"]["fps_original"])      # KHÔNG phải fps_export — xem config.yaml
    W, H = int(cfg["camera"]["width"]), int(cfg["camera"]["height"])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir, overlay_dir = out_dir / "labels", out_dir / "overlay"
    if args.also_yolo:
        labels_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = load_sorted_frame_paths(args.video_folder)
    if not frame_paths:
        raise FileNotFoundError(f"No .jpg/.jpeg/.png frames in {args.video_folder}")
    print(f"{len(frame_paths)} frame(s) @ fps_original={fps}  "
          f"(duration={len(frame_paths)/fps:.2f}s)")

    if args.all_classes:
        prompts = list(cfg.get("label_transfer", {}).get("class_names", []))
        if not prompts:
            raise ValueError("--all-classes cần label_transfer.class_names trong config.yaml")
        prompt_class_pairs = [(p, i) for i, p in enumerate(prompts)]
    else:
        prompt = args.prompt or "car"
        prompt_class_pairs = [(prompt, args.class_id)]

    print("Loading SAM3 video predictor (first run downloads checkpoint)...")
    predictor = build_sam3_video_predictor()

    tracks: dict = {}
    yolo_by_frame: dict[int, list] = {}
    for prompt, class_id in prompt_class_pairs:
        prompt_tracks, prompt_yolo = _run_one_prompt(
            predictor, args.video_folder, frame_paths, prompt, class_id, fps, W, H,
            args.score_threshold, out_dir, args.also_yolo)
        tracks.update(prompt_tracks)
        for frame_idx, rows in prompt_yolo.items():
            yolo_by_frame.setdefault(frame_idx, []).extend(rows)

    predictor.shutdown()

    for entry in tracks.values():
        entry["frames"].sort(key=lambda r: r["frame_idx"])

    merge_iou = args.merge_iou
    if merge_iou is None:
        merge_iou = float(cfg.get("label_transfer", {}).get("cross_class_iou", 0.85))
    n_dropped = 0
    if not args.no_merge and merge_iou > 0:
        tracks, n_dropped = merge_duplicate_observations(tracks, merge_iou)
        print(f"\nCross-prompt NMS: dropped {n_dropped} duplicate observation(s) "
              f"(IoU >= {merge_iou:.2f})")

    if args.also_yolo:
        yolo_by_frame = {}
        for key, entry in tracks.items():
            for obs in entry["frames"]:
                yolo_by_frame.setdefault(int(obs["frame_idx"]), []).append((entry["class_id"], obs))
        for frame_idx, rows in yolo_by_frame.items():
            with open(labels_dir / f"{frame_idx}.txt", "w") as f:
                for class_id, obs in rows:
                    f.write(f"{class_id} {obs['cx']:.6f} {obs['cy']:.6f} "
                            f"{obs['w']:.6f} {obs['h']:.6f}\n")

    payload = dict(fps=fps, width=W, height=H,
                   prompts=[p for p, _ in prompt_class_pairs],
                   merge_iou=merge_iou, n_merged_duplicate_observations=n_dropped,
                   tracks=tracks)
    tracks_path = out_dir / "tracks.json"
    tracks_path.write_text(json.dumps(payload, indent=2))

    n_tracks = len(tracks)
    n_obs = sum(len(t["frames"]) for t in tracks.values())
    print(f"\n✓ {tracks_path}  ({n_tracks} track(s), {n_obs} box-observation(s) total)")
    if args.also_yolo:
        print(f"  YOLO/overlay (sanity check) -> {labels_dir}/ , {overlay_dir}/")
        print(f"  Mở vài overlay PNG, xác nhận box bám đúng xe TRƯỚC KHI chạy label_transfer.py —")
        print(f"  nếu SAM3 sai ở đây, nhãn event sinh ra sau cũng sai theo.")
    print(f"\n  Tiếp theo: python label_transfer.py --tracks {tracks_path} --events <events.h5> "
          f"--output <windows.json>")


if __name__ == "__main__":
    main()
