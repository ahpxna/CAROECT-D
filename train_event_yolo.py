#!/usr/bin/env python3
"""
train_event_yolo.py — Train YOLOv8 trên event-pseudo-image dataset
(build_event_dataset.py output). Wrapper mỏng quanh ultralytics — KHÔNG viết
lại training loop, đúng nguyên tắc "không fork, chỉ gọi làm library" đã áp
dụng nhất quán cho v2e/DVS-Voltmeter trong project này.

VÌ SAO DÙNG YOLOv8 (không phải model event-native như RVT/RED)
------------------------------------------------------------------
eTraM (Verma et al., ASU) benchmark chính xác 3 model: RVT, RED, YOLOv8 — và
YOLOv8 là baseline dễ tái lập nhất, không cần kiến trúc event-native tự
implement. Vì representation ở build_event_dataset.py đã render event thành
ảnh 3-kênh chuẩn, YOLOv8 pretrained trên ImageNet/COCO dùng được NGAY làm
điểm khởi tạo (transfer learning) — hợp lý khi dataset event luôn nhỏ hơn
nhiều bậc so với ảnh RGB thường có sẵn.

Nếu sau này muốn thử RVT/RED (event-native, xử lý voxel grid nhiều kênh thay
vì ảnh 3 kênh), cần sửa build_event_dataset.py để xuất voxel grid N kênh thay
vì render_window() 3 kênh hiện tại — không đụng vào file này.

CÁCH ĐỌC KẾT QUẢ CHO 3 KIỂU SPLIT CỦA PROJECT
-------------------------------------------------
CAROECT-D định nghĩa 3 split (scene / view / lighting) — mỗi split là một
LẦN gọi eval_event_yolo.py riêng trên val/test tương ứng (dataset khác nhau
theo policy đang test, dựng bằng build_event_dataset.py --split test với
site-id thuộc nhóm test đó). File train này KHÔNG biết gì về policy — chỉ
train trên đúng data.yaml được đưa vào.

Usage:
  python train_event_yolo.py --data dataset/data.yaml --epochs 100 \
      --imgsz 640 --batch 16 --model yolov8n.pt --project runs --name caroectd_v1
"""

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="data.yaml (từ build_event_dataset.py)")
    ap.add_argument("--model", default="yolov8n.pt",
                    help="Checkpoint khởi tạo — yolov8n/s/m/l/x.pt (pretrained COCO) hoặc "
                         "'yolov8n.yaml' để train from-scratch (không khuyến nghị, xem docstring)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42, help="Khớp seed dùng ở simulator.seed trong config.yaml")
    ap.add_argument("--project", default="runs/caroectd")
    ap.add_argument("--name", default="exp")
    ap.add_argument("--device", default=None, help="'0' cho GPU đầu tiên, 'cpu' để ép CPU, "
                    "để trống = ultralytics tự chọn")
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
            f"{data_path} không tồn tại — chạy build_event_dataset.py cho ít nhất "
            "train + val trước (mỗi split một lần gọi, xem --split trong file đó).")

    print(f"{'='*64}\nTrain YOLO trên event-pseudo-image dataset\n"
          f"  data   : {data_path}\n  model  : {args.model}\n"
          f"  epochs : {args.epochs}  imgsz={args.imgsz}  batch={args.batch}  seed={args.seed}\n"
          f"{'='*64}\n")

    model = YOLO(args.model)
    model.train(
        data=str(data_path), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        seed=args.seed, project=args.project, name=args.name, resume=args.resume,
        patience=args.patience, device=args.device,
        # channels ảnh input là 3 (R=ON count, G=OFF count, B=time-surface) —
        # KHÔNG phải RGB thật, nên augmentation màu sắc kiểu photographic
        # (hsv_h/hsv_s/hsv_v) không có ý nghĩa vật lý ở đây -> tắt để tránh
        # augment sai bản chất dữ liệu (đảo "màu" event = đảo polarity giả).
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    )
    print(f"\n✓ Xong. Weights -> {args.project}/{args.name}/weights/best.pt")
    print(f"  Đánh giá: python eval_event_yolo.py --weights {args.project}/{args.name}/weights/best.pt "
          f"--data <test_data.yaml>")


if __name__ == "__main__":
    main()
