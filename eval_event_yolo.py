#!/usr/bin/env python3
"""
Evaluate a trained event-representation YOLO model.

A normal invocation reports detection metrics for one explicit data.yaml.
For the paper's domain-transfer comparison, --weights is the calibrated-
simulator model and --baseline-weights is the default-simulator model. Both are
evaluated on exactly the same real-event test definition, so delta_domain is a
controlled difference rather than a cross-dataset comparison.

Scene, view, and lighting policies remain separate invocations with separate
traceable dataset manifests.

Usage:
  python eval_event_yolo.py --weights runs/exp/weights/best.pt --data test_data.yaml
  python eval_event_yolo.py --weights runs/calibrated/weights/best.pt \
      --baseline-weights runs/default/weights/best.pt --data real_test_data.yaml
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
    ap.add_argument("--weights", required=True, help="Primary model checkpoint")
    ap.add_argument("--data", required=True, help="Shared test/validation data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--baseline-weights", default=None,
                    help="Default-simulator baseline checkpoint for delta_domain")
    ap.add_argument("--output-json", default=None, help="Write metrics to JSON")
    args = ap.parse_args()

    print(f"{'='*64}\nEvaluate: {args.weights}\n  data: {args.data}\n{'='*64}")
    main_metrics = run_val(args.weights, args.data, args.imgsz, args.device)
    print(f"\nmAP50-95 = {main_metrics['map50_95']:.4f}   mAP50 = {main_metrics['map50']:.4f}   "
          f"Precision = {main_metrics['precision']:.4f}   Recall = {main_metrics['recall']:.4f}")

    result = dict(main=main_metrics)

    if args.baseline_weights:
        print(f"\n{'='*64}\nEvaluate baseline: {args.baseline_weights}\n{'='*64}")
        base_metrics = run_val(args.baseline_weights, args.data, args.imgsz, args.device)
        print(f"\nmAP50-95 = {base_metrics['map50_95']:.4f}   mAP50 = {base_metrics['map50']:.4f}")

        delta_50_95 = main_metrics["map50_95"] - base_metrics["map50_95"]
        delta_50 = main_metrics["map50"] - base_metrics["map50"]
        print(f"\n{'='*64}\nDelta_domain (main - baseline), shared test set '{args.data}'\n{'='*64}")
        print(f"  Δ mAP50-95 = {delta_50_95:+.4f}")
        print(f"  Δ mAP50    = {delta_50:+.4f}")
        if delta_50_95 > 0:
            print("\n  -> The calibrated-condition model outperforms the default baseline "
                  "on this same real-event test definition.")
        else:
            print("\n  -> The primary model does not outperform this baseline. "
                  "Review calibration evidence and test coverage before drawing conclusions.")
        result["baseline"] = base_metrics
        result["delta_domain"] = dict(map50_95=delta_50_95, map50=delta_50)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults -> {args.output_json}")


if __name__ == "__main__":
    main()
