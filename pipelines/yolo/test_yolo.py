import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "train_implant" / "exp" / "weights" / "best.pt"
DEFAULT_DATA = PROJECT_ROOT / "data" / "Dental Implants 2.0.v3i.yolov8" / "data.yaml"
DEFAULT_TEST_PROJECT = PROJECT_ROOT / "runs" / "test_implant"


def resolve_device(device_arg: str) -> str:
    device_arg = (device_arg or "auto").strip().lower()

    if device_arg in {"", "auto"}:
        if torch.cuda.is_available():
            return "0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if device_arg == "cpu":
        return "cpu"

    if device_arg == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if device_arg.replace(",", "").isdigit():
        if torch.cuda.is_available():
            return device_arg
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    return device_arg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained YOLO implant model on a test/val split"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help="Path to trained model (.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA),
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='e.g. "0" / "0,1" / "cpu" / "mps" / "auto"',
    )
    parser.add_argument(
        "--project",
        type=str,
        default=str(DEFAULT_TEST_PROJECT),
        help="Where to save validation artifacts",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="exp",
        help="Run folder name under project",
    )
    parser.add_argument("--plots", action="store_true", help="Save PR/F1/confusion plots")
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Export COCO-format JSON report when supported",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default="",
        help="Optional path to save a simple JSON summary of the metrics",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_report(results, class_names, split, data, weights):
    report = {
        "weights": str(weights),
        "data": str(data),
        "split": split,
        "metrics": {},
        "class_metrics": {},
    }

    # Common dict export format (Ultralytics often keeps a summary dict)
    results_dict = getattr(results, "results_dict", None)
    if isinstance(results_dict, dict):
        for k, v in results_dict.items():
            num = _safe_float(v)
            if num is None:
                continue
            report["metrics"][str(k)] = num

    # Box metrics are often nested under `box`
    box_metrics = getattr(results, "box", None)
    if box_metrics is not None:
        metric_pairs = {
            "mAP50_95": getattr(box_metrics, "map", None),
            "mAP50": getattr(box_metrics, "map50", None),
            "mAP75": getattr(box_metrics, "map75", None),
            "precision": getattr(box_metrics, "mp", None),
            "recall": getattr(box_metrics, "mr", None),
            "f1": getattr(box_metrics, "mf1", None),
        }
        for k, v in metric_pairs.items():
            num = _safe_float(v)
            if num is not None:
                report["metrics"][k] = num

        class_map = getattr(box_metrics, "maps", None)
        if class_map is not None:
            for idx, ap in enumerate(class_map):
                cls_name = class_names.get(idx, str(idx)) if hasattr(class_names, "get") else str(idx)
                num = _safe_float(ap)
                if num is None:
                    continue
                report["class_metrics"][str(cls_name)] = num

    return report


def print_metrics(report):
    metrics = report["metrics"]
    if not metrics:
        return

    keys = [
        "mAP50_95",
        "mAP50",
        "mAP75",
        "precision",
        "recall",
        "f1",
    ]
    print("\n=== Evaluation Summary ===")
    for k in keys:
        if k in metrics:
            print(f"{k:10s}: {metrics[k]:.4f}")

    # Fallback values from results_dict keys if present
    fallback = [k for k in metrics.keys() if k not in keys]
    if fallback:
        for k in sorted(fallback):
            print(f"{k:10s}: {metrics[k]:.4f}")


def main():
    args = parse_args()
    resolved_device = resolve_device(args.device)
    weights_path = Path(args.weights).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    project_path = Path(args.project).expanduser()
    model = YOLO(str(weights_path))

    results = model.val(
        data=str(data_path),
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        batch=args.batch,
        imgsz=args.imgsz,
        device=resolved_device,
        project=str(project_path),
        name=args.name,
        save_json=args.save_json,
        plots=args.plots,
        verbose=args.verbose,
    )

    report = build_report(
        results,
        model.names,
        split=args.split,
        data=str(data_path),
        weights=str(weights_path),
    )
    print_metrics(report)

    if args.report_json:
        out = Path(args.report_json).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nSaved metrics report: {out}")


if __name__ == "__main__":
    main()
