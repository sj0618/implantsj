from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import yaml
from PIL import Image, ImageFilter

from pipelines.yolo.crop_yolo import expand_square_crop


SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass(frozen=True)
class LabeledBox:
    image_path: Path
    label: str
    box: tuple[float, float, float, float]


def sanitize_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return cleaned or "unknown"


def _class_names(data_yaml: Path) -> dict[int, str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = data.get("names", {})
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    raise ValueError(f"data.yaml has no usable names field: {data_yaml}")


def _txt_boxes(path: Path, image_size: tuple[int, int], names: dict[int, str]) -> Iterable[tuple[str, tuple[float, float, float, float]]]:
    width, height = image_size
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        values = raw_line.strip().split()
        if not values:
            continue
        try:
            class_id = int(float(values[0]))
            coords = [float(value) for value in values[1:]]
        except ValueError as exc:
            raise ValueError(f"invalid YOLO annotation {path}:{line_number}") from exc
        if class_id not in names:
            raise ValueError(f"unknown class id {class_id} in {path}:{line_number}")
        if len(coords) == 4:
            cx, cy, box_width, box_height = coords
            x1 = (cx - box_width / 2.0) * width
            y1 = (cy - box_height / 2.0) * height
            x2 = (cx + box_width / 2.0) * width
            y2 = (cy + box_height / 2.0) * height
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            xs = coords[0::2]
            ys = coords[1::2]
            x1, x2 = min(xs) * width, max(xs) * width
            y1, y2 = min(ys) * height, max(ys) * height
        else:
            raise ValueError(f"unsupported YOLO annotation {path}:{line_number}")
        yield names[class_id], (x1, y1, x2, y2)


def _json_boxes(path: Path) -> Iterable[tuple[str, tuple[float, float, float, float]]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    boxes = data.get("boxes", []) if isinstance(data, dict) else []
    for index, item in enumerate(boxes, start=1):
        try:
            label = str(item["label"])
            cx = float(item["x"])
            cy = float(item["y"])
            width = float(item["width"])
            height = float(item["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid JSON box {path} item {index}") from exc
        yield label, (cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)


def _collect_boxes(data_root: Path, names: dict[int, str]) -> list[LabeledBox]:
    records: list[LabeledBox] = []
    for source_split in ("train", "valid", "val", "test"):
        image_root = data_root / source_split / "images"
        label_root = data_root / source_split / "labels"
        if not image_root.is_dir() or not label_root.is_dir():
            continue
        for image_path in sorted(path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
            relative = image_path.relative_to(image_root)
            txt_path = (label_root / relative).with_suffix(".txt")
            json_path = (label_root / relative).with_suffix(".json")
            with Image.open(image_path) as opened:
                size = opened.size
            if txt_path.is_file():
                boxes = _txt_boxes(txt_path, size, names)
            elif json_path.is_file():
                boxes = _json_boxes(json_path)
            else:
                raise FileNotFoundError(f"missing label for image: {image_path}")
            records.extend(LabeledBox(image_path.resolve(), str(label), tuple(box)) for label, box in boxes)
    if not records:
        raise RuntimeError(f"no labeled boxes found under {data_root}")
    return records


def _split_records(records: Sequence[LabeledBox], seed: int) -> dict[str, list[LabeledBox]]:
    labels_by_image: dict[Path, set[str]] = defaultdict(set)
    for record in records:
        labels_by_image[record.image_path].add(record.label)
    ambiguous = {
        str(path): sorted(labels)
        for path, labels in labels_by_image.items()
        if len(labels) > 1
    }
    if ambiguous:
        sample = dict(list(sorted(ambiguous.items()))[:5])
        raise ValueError(f"source images contain multiple class labels: {sample}")

    by_label: dict[str, dict[Path, list[LabeledBox]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_label[record.label][record.image_path].append(record)
    output = {split: [] for split in SPLITS}
    rng = random.Random(seed)
    for label in sorted(by_label):
        image_groups = sorted(by_label[label].values(), key=lambda group: str(group[0].image_path))
        rng.shuffle(image_groups)
        if len(image_groups) < 3:
            raise ValueError(
                f"label {label!r} needs at least 3 source images for leakage-free "
                f"train/valid/test splits; got {len(image_groups)}"
            )
        n_valid = max(1, round(len(image_groups) * 0.1))
        n_test = max(1, round(len(image_groups) * 0.1))
        if n_valid + n_test >= len(image_groups):
            n_valid = n_test = 1
        n_train = len(image_groups) - n_valid - n_test
        grouped_splits = {
            "train": image_groups[:n_train],
            "valid": image_groups[n_train:n_train + n_valid],
            "test": image_groups[n_train + n_valid:],
        }
        for split, groups in grouped_splits.items():
            output[split].extend(record for group in groups for record in group)
    return output


def build_dataset(
    data_root: Path,
    output_root: Path,
    data_yaml: Path,
    *,
    output_size: int = 224,
    sharpen: bool = True,
    square: bool = True,
    seed: int = 42,
) -> dict[str, dict[str, int]]:
    data_root = Path(data_root).resolve()
    output_root = Path(output_root).resolve()
    names = _class_names(Path(data_yaml))
    split_records = _split_records(_collect_boxes(data_root, names), seed)
    counters: dict[str, int] = defaultdict(int)
    summary = {split: defaultdict(int) for split in SPLITS}
    for split in SPLITS:
        for record in split_records[split]:
            counters[record.label] += 1
            class_name = sanitize_label(record.label)
            target = output_root / split / class_name / f"{class_name}_{counters[record.label]:03d}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(record.image_path) as opened:
                image = opened.convert("RGB")
                crop = expand_square_crop(image, record.box, 0.0) if square else image.crop(record.box)
                crop = crop.resize((output_size, output_size), Image.Resampling.BICUBIC)
                if sharpen:
                    crop = crop.filter(ImageFilter.SHARPEN)
                crop.save(target, quality=95)
            summary[split][record.label] += 1
    return {split: dict(counts) for split, counts in summary.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop labeled YOLO boxes and create train/valid/test ImageFolder data.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--output-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-sharpen", action="store_true")
    parser.add_argument("--no-square", action="store_true")
    args = parser.parse_args()
    summary = build_dataset(
        Path(args.data_root),
        Path(args.output_root),
        Path(args.data_yaml),
        output_size=args.output_size,
        sharpen=not args.no_sharpen,
        square=not args.no_square,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
