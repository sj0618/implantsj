import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YOLO_DATA = PROJECT_ROOT / "data" / "Dental Implants 2.0.v3i.yolov8" / "data.yaml"
DEFAULT_YOLO_WEIGHTS = PROJECT_ROOT / "weights" / "yolov8s.pt"
DEFAULT_TRAIN_PROJECT = PROJECT_ROOT / "runs" / "train_implant"


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
        if not torch.cuda.is_available():
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device_arg

    return device_arg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 implant manufacturer detector"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_YOLO_DATA),
        help="Path to YOLO data.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_YOLO_WEIGHTS),
        help="Base model: yolov8n.pt / yolov8s.pt / yolov8m.pt / custom .pt",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='e.g. "0" / "0,1" / "cpu" / "mps" / "auto"',
    )
    parser.add_argument("--project", type=str, default=str(DEFAULT_TRAIN_PROJECT))
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cache", type=str, default="", help='e.g. "ram"')
    parser.add_argument("--use-mosaic", action="store_true")
    parser.add_argument("--use-amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    resolved_device = resolve_device(args.device)
    data_path = Path(args.data).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    project_path = Path(args.project).expanduser()

    model = YOLO(str(model_path))
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=resolved_device,
        project=str(project_path),
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        cache=args.cache if args.cache else False,
        mosaic=args.use_mosaic,
        amp=args.use_amp,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
