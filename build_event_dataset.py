#!/usr/bin/env python3
"""
build_event_dataset.py — Gộp events.h5 + windows.json (label_transfer.py) thành
1 dataset dạng YOLO (images/ + labels/) để train_event_yolo.py tiêu thụ.

VÌ SAO CẦN "RENDER" EVENT THÀNH ẢNH GIẢ (pseudo-image)
---------------------------------------------------------
Event thô là 1 danh sách (x,y,t,p) rời rạc, không phải lưới pixel — không đưa
thẳng vào YOLO được. Ta cần một REPRESENTATION gộp theo cửa sổ thời gian
[t_start,t_end) thành 1 "ảnh" H×W×3, đúng cách RVT/RED/YOLOv8 (3 baseline
paper eTraM dùng) xử lý input. Representation ở đây (time-surface hybrid,
3 kênh, không cần huấn luyện thêm gì để tính):

    Kênh R = đếm event ON  trong cửa sổ, tại mỗi pixel, chuẩn hoá [0,255]
    Kênh G = đếm event OFF trong cửa sổ, tại mỗi pixel, chuẩn hoá [0,255]
    Kênh B = "time surface" = (t_event_gần_nhất_trong_cửa_sổ - t_start) / (t_end-t_start),
             chuẩn hoá [0,255] — pixel càng "mới" (event gần cuối cửa sổ) càng sáng

Trực giác: kênh R/G cho biết "ở đâu có chuyển động và theo hướng sáng/tối
nào" (giống 2 kênh ON/OFF cơ bản nhất của mọi event representation trong
literature — ESIM/v2e/DVS-Voltmeter đều polarity-tách ON/OFF), kênh B cho
biết "chuyển động đó xảy ra sớm hay muộn trong cửa sổ" (time surface — ý
tưởng gốc từ HOTS, Lagorce et al. — cho phép model phân biệt hướng chuyển
động trong 1 ảnh tĩnh 3 kênh, không cần voxel grid nhiều kênh phức tạp hơn).

3 kênh này tương thích TRỰC TIẾP với checkpoint YOLOv8 pretrained (conv đầu
vào 3-channel, không cần sửa kiến trúc) — train_event_yolo.py fine-tune từ
đó thay vì train from scratch, đúng thực hành phổ biến trong literature khi
dữ liệu event ít hơn ImageNet nhiều bậc.

SPLIT POLICY
-------------
CAROECT-D outline định nghĩa 3 kiểu split (scene / view / lighting) — vì
project CHƯA có site data thật, script này KHÔNG tự đoán site nào thuộc scene
nào. Người gọi (hoặc run_pipeline.sh) quyết định --split train|val|test cho
MỖI lần gọi ứng với 1 site/session cụ thể — gọi lại nhiều lần cho nhiều
site để dựng đủ 3 tập theo đúng policy đang muốn test (chi tiết: xem
README_PIPELINE.md phần "Dataset splits").

Usage:
  python build_event_dataset.py --events events.h5 --windows windows.json \
      --output dataset/ --split train --site-id site01 --img-size 640 640

  # Ảnh mẫu vài cửa sổ để sanity-check bằng mắt (không ghi vào dataset):
  python build_event_dataset.py --events events.h5 --windows windows.json \
      --output /tmp/preview --split train --site-id preview --preview 5
"""

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml


def load_events_h5(path):
    with h5py.File(str(path), "r") as hf:
        return {k: hf[k][:] for k in ("x", "y", "t", "p")}


# ══════════════════════════════════════════════════════════════════════════
#  N1 · RENDER 1 CỬA SỔ EVENT -> ẢNH 3 KÊNH  (công thức ở module docstring)
# ══════════════════════════════════════════════════════════════════════════

def render_window(events: dict, idx_lo: int, idx_hi: int, t_start: float, t_end: float,
                   width: int, height: int) -> np.ndarray:
    x = events["x"][idx_lo:idx_hi].astype(np.int64)
    y = events["y"][idx_lo:idx_hi].astype(np.int64)
    t = events["t"][idx_lo:idx_hi].astype(np.float64)
    p = events["p"][idx_lo:idx_hi]

    on_count = np.zeros((height, width), dtype=np.float32)
    off_count = np.zeros((height, width), dtype=np.float32)
    time_surface = np.zeros((height, width), dtype=np.float32)

    if len(x) > 0:
        on_mask = p > 0
        np.add.at(on_count, (y[on_mask], x[on_mask]), 1.0)
        np.add.at(off_count, (y[~on_mask], x[~on_mask]), 1.0)

        dur = max(t_end - t_start, 1.0)
        t_norm = np.clip((t - t_start) / dur, 0.0, 1.0)
        # time surface: MỖI pixel giữ timestamp CHUẨN HOÁ của event GẦN CUỐI
        # cửa sổ nhất (không phải trung bình) — dùng np.maximum.at để "event
        # sau ghi đè event trước" theo đúng thứ tự thời gian (events đã sort).
        np.maximum.at(time_surface, (y, x), t_norm)

    def norm_u8(ch, ref_max=None):
        m = ref_max if ref_max is not None else max(ch.max(), 1.0)
        return np.clip(ch / m * 255.0, 0, 255).astype(np.uint8)

    # ON/OFF count chuẩn hoá theo giá trị max CỦA CHÍNH CỬA SỔ ĐÓ (không phải
    # theo toàn clip) — mỗi window là 1 sample độc lập, giống cách mọi frame
    # ảnh thường tự chuẩn hoá exposure của chính nó.
    R = norm_u8(on_count)
    G = norm_u8(off_count)
    B = (time_surface * 255.0).astype(np.uint8)
    return np.stack([R, G, B], axis=-1)  # HxWx3, kênh thứ tự R,G,B


# ══════════════════════════════════════════════════════════════════════════
#  N2 · GHI DATASET (YOLO format: images/*.png + labels/*.txt cùng tên)
# ══════════════════════════════════════════════════════════════════════════

def write_data_yaml(dataset_root: Path, class_names: list):
    """Idempotent — chỉ ghi 1 lần / cập nhật nếu class_names thay đổi.
    ultralytics YOLO đọc trực tiếp file này qua --data."""
    data_yaml = dataset_root / "data.yaml"
    payload = dict(
        path=str(dataset_root.resolve()),
        train="train/images", val="val/images", test="test/images",
        names={i: n for i, n in enumerate(class_names)},
    )
    with open(data_yaml, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return data_yaml


def _box_to_xyxy_pixels(box: dict, img_w: int, img_h: int):
    cx, cy, bw, bh = box["cx"], box["cy"], box["w"], box["h"]
    x0 = max(0.0, (cx - bw / 2) * img_w)
    y0 = max(0.0, (cy - bh / 2) * img_h)
    x1 = min(float(img_w), (cx + bw / 2) * img_w)
    y1 = min(float(img_h), (cy + bh / 2) * img_h)
    return x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)


def _rectangle_segmentation(x: float, y: float, w: float, h: float):
    return [[x, y, x + w, y, x + w, y + h, x, y + h]]


def _mask_to_coco_segmentation(mask_path: Path, img_w: int, img_h: int):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None, 0.0
    if mask.shape[:2] != (img_h, img_w):
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentations = []
    area = 0.0
    for contour in contours:
        if len(contour) < 3:
            continue
        contour = contour.reshape(-1, 2).astype(np.float32)
        area += float(cv2.contourArea(contour))
        poly = contour.reshape(-1).tolist()
        if len(poly) >= 6:
            segmentations.append(poly)
    return segmentations or None, area


def _add_coco_annotation(coco: dict, ann_id: int, image_id: int, box: dict,
                         tracks_base_dir: Path | None, img_w: int, img_h: int):
    x, y, w, h = _box_to_xyxy_pixels(box, img_w, img_h)
    segmentation = None
    area = 0.0
    mask_rel = box.get("mask_path")
    if mask_rel and tracks_base_dir is not None:
        segmentation, area = _mask_to_coco_segmentation(tracks_base_dir / mask_rel, img_w, img_h)
    if not segmentation:
        segmentation = _rectangle_segmentation(x, y, w, h)
        area = w * h
    coco["annotations"].append(dict(
        id=ann_id,
        image_id=image_id,
        category_id=int(box["class_id"]),
        bbox=[float(x), float(y), float(w), float(h)],
        area=float(area),
        segmentation=segmentation,
        iscrowd=0,
        track_id=str(box.get("track_id", "")),
        source_mask=str(mask_rel or ""),
    ))
    return ann_id + 1


def process(args):
    events = load_events_h5(args.events)
    payload = json.loads(Path(args.windows).read_text())
    windows, W, H = payload["windows"], payload["width"], payload["height"]
    tracks_base_dir = Path(payload["tracks_base_dir"]) if payload.get("tracks_base_dir") else None

    out_root = Path(args.output)
    split_dir = out_root / args.split
    img_dir, lbl_dir = split_dir / "images", split_dir / "labels"
    ann_dir = out_root / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    img_w, img_h = args.img_size
    t_sorted = events["t"]  # đã sort theo t trong run_v2e.py/run_dvsvolt.py/cevt_to_events.py
    coco = dict(
        images=[],
        annotations=[],
        categories=[dict(id=i, name=name) for i, name in enumerate(args.class_names)],
    )
    image_id = 1
    ann_id = 1

    n_written = 0
    n_empty = 0
    preview_n = args.preview or 0

    for wi, win in enumerate(windows):
        t0, t1 = win["t_start_us"], win["t_end_us"]
        idx_lo = int(np.searchsorted(t_sorted, t0, side="left"))
        idx_hi = int(np.searchsorted(t_sorted, t1, side="left"))

        if idx_hi <= idx_lo and not win["boxes"]:
            n_empty += 1
            if preview_n <= 0:
                continue  # bỏ qua cửa sổ hoàn toàn trống (không event, không box)
            # với --preview vẫn render để thấy cả cửa sổ "trống thật" trông ra sao

        img = render_window(events, idx_lo, idx_hi, t0, t1, W, H)
        if (img_w, img_h) != (W, H):
            img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA)

        name = f"{args.site_id}_{wi:06d}"
        cv2.imwrite(str(img_dir / f"{name}.png"), img)
        if args.export_coco:
            coco["images"].append(dict(
                id=image_id,
                file_name=f"{args.split}/images/{name}.png",
                width=img_w,
                height=img_h,
                t_start_us=float(t0),
                t_end_us=float(t1),
                site_id=args.site_id,
            ))

        with open(lbl_dir / f"{name}.txt", "w") as f:
            for b in win["boxes"]:
                f.write(f"{b['class_id']} {b['cx']:.6f} {b['cy']:.6f} "
                        f"{b['w']:.6f} {b['h']:.6f}\n")
                if args.export_coco:
                    ann_id = _add_coco_annotation(
                        coco, ann_id, image_id, b, tracks_base_dir, img_w, img_h)
        if args.export_coco:
            image_id += 1
        n_written += 1

        if preview_n > 0 and n_written >= preview_n:
            print(f"[preview] Đã ghi {n_written} ảnh mẫu -> {img_dir}/  (dừng do --preview)")
            break

        if n_written % 500 == 0:
            print(f"  [{n_written}] window {wi}: {idx_hi-idx_lo} event(s), "
                  f"{len(win['boxes'])} box(es)")

    print(f"\n✓ split='{args.split}' site='{args.site_id}': {n_written} sample(s) ghi, "
          f"{n_empty} cửa sổ trống bị bỏ qua")
    print(f"  images -> {img_dir}/\n  labels -> {lbl_dir}/")

    if not args.preview:
        yaml_path = write_data_yaml(out_root, args.class_names)
        print(f"  data.yaml -> {yaml_path}")
        if args.export_coco:
            coco_path = ann_dir / f"instances_{args.split}_{args.site_id}.json"
            coco_path.write_text(json.dumps(coco, indent=2))
            print(f"  COCO segmentation -> {coco_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", required=True, help="events.h5 (sim hoặc real)")
    ap.add_argument("--windows", required=True, help="windows.json (từ label_transfer.py)")
    ap.add_argument("--output", required=True, help="Thư mục gốc dataset (chứa train/val/test/)")
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--site-id", required=True,
                    help="Tiền tố tên file, PHẢI unique giữa các lần gọi khác nhau "
                         "(vd site01_day, site01_night, site02...) để không ghi đè")
    ap.add_argument("--img-size", type=int, nargs=2, default=[640, 640], metavar=("W", "H"))
    ap.add_argument("--class-names", nargs="+",
                    default=["car", "truck", "bus", "motorcycle", "person",
                             "bicycle", "micromobility", "other"],
                    help="Danh sách tên class theo đúng thứ tự class_id trong tracks.json "
                         "(khớp 8 class trong CAROECT-D outline / eTraM)")
    ap.add_argument("--preview", type=int, default=0,
                    help="Chỉ render N ảnh đầu để xem thử, KHÔNG ghi vào dataset chính thức")
    ap.add_argument("--export-coco", action="store_true",
                    help="Xuất COCO instances JSON, dùng SAM mask_path nếu windows.json có mask.")
    args = ap.parse_args()
    process(args)


if __name__ == "__main__":
    main()
