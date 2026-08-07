#!/usr/bin/env python3
"""
eval_event_yolo.py — Đánh giá model đã train, gồm 3 split policy của CAROECT-D
và bài kiểm domain-transfer (Δ_domain — bằng chứng định lượng cho novelty
"calibrate simulator bằng event thật" đã bàn trong phần lý thuyết project).

3 CÁCH DÙNG
------------
1. Đánh giá thường (mAP trên 1 tập test):
     python eval_event_yolo.py --weights runs/exp/weights/best.pt --data test_data.yaml

2. So sánh domain-transfer: model train trên sim ĐÃ calibrate vs model train
   trên sim với tham số MẶC ĐỊNH (chưa chạy calibrate_simulator.py), cả 2 test
   trên CÙNG 1 tập real event (vd dataset dựng từ events_real.h5 quay bằng
   evs_recorder.cpp, hoặc format eTraM/TUMTraf Event nếu đã convert sang cùng
   schema events.h5 + windows.json):
     python eval_event_yolo.py --weights runs/calibrated/weights/best.pt \
         --baseline-weights runs/uncalibrated/weights/best.pt \
         --data real_test_data.yaml

   In ra đúng con số Δ_domain = mAP(calibrated) - mAP(uncalibrated) trên cùng
   real test set — đây là con số "ăn tiền" để chứng minh novelty bằng số,
   không chỉ bằng lời, khi bảo vệ đồ án.

3. Nhiều split cùng lúc (scene/view/lighting) — gọi script 3 lần với --data
   trỏ tới 3 data.yaml khác nhau (mỗi cái build từ site-id thuộc đúng nhóm
   test tương ứng, xem README_PIPELINE.md).
"""

import argparse
import json


def run_val(weights: str, data: str, imgsz: int, device):
    from ultralytics import YOLO
    model = YOLO(weights)
    metrics = model.val(data=data, imgsz=imgsz, device=device)
    return dict(
        map50_95=float(metrics.box.map),
        map50=float(metrics.box.map50),
        map75=float(metrics.box.map75),
        precision=float(metrics.box.mp),
        recall=float(metrics.box.mr),
        per_class_map50_95=[float(v) for v in metrics.box.maps],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="best.pt của model chính cần đánh giá")
    ap.add_argument("--data", required=True, help="data.yaml của tập test/val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--baseline-weights", default=None,
                    help="(tuỳ chọn) weights của model baseline (vd train trên sim KHÔNG "
                         "calibrate) để tính Δ_domain trên cùng --data")
    ap.add_argument("--output-json", default=None, help="Ghi kết quả ra file JSON")
    args = ap.parse_args()

    print(f"{'='*64}\nĐánh giá: {args.weights}\n  data: {args.data}\n{'='*64}")
    main_metrics = run_val(args.weights, args.data, args.imgsz, args.device)
    print(f"\nmAP50-95 = {main_metrics['map50_95']:.4f}   mAP50 = {main_metrics['map50']:.4f}   "
          f"Precision = {main_metrics['precision']:.4f}   Recall = {main_metrics['recall']:.4f}")

    result = dict(main=main_metrics)

    if args.baseline_weights:
        print(f"\n{'='*64}\nĐánh giá baseline (để so Δ_domain): {args.baseline_weights}\n{'='*64}")
        base_metrics = run_val(args.baseline_weights, args.data, args.imgsz, args.device)
        print(f"\nmAP50-95 = {base_metrics['map50_95']:.4f}   mAP50 = {base_metrics['map50']:.4f}")

        delta_50_95 = main_metrics["map50_95"] - base_metrics["map50_95"]
        delta_50 = main_metrics["map50"] - base_metrics["map50"]
        print(f"\n{'='*64}\nΔ_domain (main - baseline), cùng test set '{args.data}'\n{'='*64}")
        print(f"  Δ mAP50-95 = {delta_50_95:+.4f}")
        print(f"  Δ mAP50    = {delta_50:+.4f}")
        if delta_50_95 > 0:
            print(f"\n  -> Model chính TỐT HƠN baseline trên chính test set này.")
            print(f"     Nếu baseline = train trên sim tham số MẶC ĐỊNH và main = train trên")
            print(f"     sim ĐÃ calibrate bằng events_real.h5 (calibrate_simulator.py), đây là")
            print(f"     bằng chứng định lượng cho việc calibrate simulator theo event thật")
            print(f"     thực sự cải thiện domain-transfer — con số cho slide bảo vệ.")
        else:
            print(f"\n  -> Model chính KHÔNG tốt hơn baseline trên test set này — cần xem lại")
            print(f"     quy trình calibrate (đủ điều kiện sáng chưa, real events có đủ dài "
                  f"không) trước khi kết luận calibrate không có tác dụng.")
        result["baseline"] = base_metrics
        result["delta_domain"] = dict(map50_95=delta_50_95, map50=delta_50)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Kết quả -> {args.output_json}")


if __name__ == "__main__":
    main()
