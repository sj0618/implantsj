from __future__ import annotations

import argparse
import importlib
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_CROP_SIZE = 512


@dataclass(frozen=True, slots=True)
class CropResult:
    source: Path
    saved: int
    skipped: int


def resolve_device(device_arg: str) -> str:
    torch = importlib.import_module("torch")
    value = (device_arg or "auto").strip().lower()
    if value in {"", "auto"}:
        if torch.cuda.is_available():
            return "0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if value == "cpu":
        return "cpu"
    if value == "mps":
        return "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    if value.replace(",", "").isdigit():
        if torch.cuda.is_available():
            return value
        return "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    return value


def sanitize_token(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return cleaned or "item"


def iter_image_files(root: Path) -> Iterator[Path]:
    if root.is_file():
        if root.suffix.lower() in IMAGE_SUFFIXES:
            yield root
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def load_roots(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in {".yaml", ".yml"}:
        yaml = importlib.import_module("yaml")
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
        roots: list[Path] = []
        for key in ("train", "val", "test"):
            value = data.get(key)
            if not isinstance(value, str):
                continue
            root = (source.parent / value).expanduser().resolve()
            if root.exists():
                roots.append(root)
        return roots
    return [source.expanduser().resolve()]


def infer_split_name(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "train" in parts:
        return "train"
    if "valid" in parts or "val" in parts:
        return "valid"
    if "test" in parts:
        return "test"
    return "unspecified"


def expand_square_crop(image: Image.Image, box: Sequence[float], pad: float) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + pad)
    side_px = max(1, int(math.ceil(side)))
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    right = left + side_px
    bottom = top + side_px

    crop_left = max(0, left)
    crop_top = max(0, top)
    crop_right = min(width, right)
    crop_bottom = min(height, bottom)
    canvas = Image.new("RGB", (side_px, side_px), (0, 0, 0))
    if crop_right > crop_left and crop_bottom > crop_top:
        crop = image.crop((crop_left, crop_top, crop_right, crop_bottom))
        canvas.paste(crop, (crop_left - left, crop_top - top))
    return canvas


def crop_detection(image: Image.Image, box: Sequence[float], crop_size: int, pad: float) -> Image.Image:
    return expand_square_crop(image, box, pad).resize((crop_size, crop_size), Image.Resampling.BICUBIC)


def chunked(items: Sequence[Path], size: int) -> Iterator[list[Path]]:
    step = max(1, size)
    for start in range(0, len(items), step):
        yield list(items[start:start + step])


def relative_token(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path.name
    if isinstance(rel, Path):
        text = rel.with_suffix("").as_posix()
    else:
        text = str(rel)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def build_output_path(output_root: Path, source_path: Path, root: Path, class_name: str, det_index: int, conf: float) -> Path:
    split = infer_split_name(source_path)
    class_dir = sanitize_token(class_name)
    stem = sanitize_token(source_path.stem)
    digest = relative_token(source_path, root)
    filename = f"{stem}__{digest}__det{det_index:02d}_conf{conf:.2f}.jpg"
    return output_root / split / class_dir / filename


def resolve_class_name(names: object, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def collect_images(source: Path) -> list[tuple[Path, Path]]:
    roots = load_roots(source)
    pairs: list[tuple[Path, Path]] = []
    for root in roots:
        for image_path in iter_image_files(root):
            pairs.append((root, image_path))
    return pairs


def predict_and_crop(
    model,
    images: list[tuple[Path, Path]],
    output_root: Path,
    crop_size: int,
    pad: float,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    batch_size: int,
    save_empty: bool,
) -> list[CropResult]:
    image_paths = [path for _, path in images]
    root_by_path = {path: root for root, path in images}
    results: list[CropResult] = []

    for batch in chunked(image_paths, batch_size):
        preds = model.predict(
            source=[str(path) for path in batch],
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        for result in preds:
            image_path = Path(result.path)
            root = root_by_path.get(image_path, image_path.parent)
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    xyxy = []
                else:
                    xyxy_attr = getattr(boxes, "xyxy", None)
                    xyxy = xyxy_attr.cpu().tolist() if xyxy_attr is not None else []
                if not xyxy:
                    if save_empty:
                        out_path = build_output_path(output_root, image_path, root, "empty", 0, 0.00)
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        image.resize((crop_size, crop_size), Image.Resampling.BICUBIC).save(out_path, quality=95)
                        results.append(CropResult(source=image_path, saved=1, skipped=0))
                    else:
                        results.append(CropResult(source=image_path, saved=0, skipped=1))
                    continue

                cls_attr = getattr(boxes, "cls", None)
                conf_attr = getattr(boxes, "conf", None)
                cls_ids = cls_attr.cpu().tolist() if cls_attr is not None else [0] * len(xyxy)
                confs = conf_attr.cpu().tolist() if conf_attr is not None else [0.0] * len(xyxy)
                saved = 0
                for det_index, (box, cls_id, score) in enumerate(zip(xyxy, cls_ids, confs), start=1):
                    class_name = resolve_class_name(getattr(model, "names", {}), int(cls_id))
                    crop = crop_detection(image, box, crop_size, pad)
                    out_path = build_output_path(output_root, image_path, root, class_name, det_index, float(score))
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    crop.save(out_path, quality=95)
                    saved += 1
                results.append(CropResult(source=image_path, saved=saved, skipped=0))

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop YOLO-detected implant regions into fixed 512x512 images.")
    parser.add_argument("--source", required=True, help="Image root, single image, or YOLO data.yaml")
    parser.add_argument("--weights", required=True, help="Path to YOLO weights (.pt)")
    parser.add_argument("--output-root", required=True, help="Directory for cropped images")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    parser.add_argument("--crop-size", type=int, default=DEFAULT_CROP_SIZE)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--pad", type=float, default=0.15, help="Extra context around each box, as a fraction")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-empty", action="store_true", help="Save resized full images when no boxes are detected")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    images = collect_images(source)
    if not images:
        raise RuntimeError(f"No image files found under {source}")

    YOLO = importlib.import_module("ultralytics").YOLO
    model = YOLO(str(weights))
    device = resolve_device(args.device)
    results = predict_and_crop(
        model=model,
        images=images,
        output_root=output_root,
        crop_size=args.crop_size,
        pad=args.pad,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=device,
        batch_size=args.batch_size,
        save_empty=args.save_empty,
    )

    total_saved = sum(item.saved for item in results)
    total_skipped = sum(item.skipped for item in results)
    print(f"Processed images: {len(results)}")
    print(f"Saved crops: {total_saved}")
    print(f"Skipped images: {total_skipped}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
