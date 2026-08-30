#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "pipelines" / "yolo" / "crop_labeled.py"
spec = importlib.util.spec_from_file_location("crop_labeled_implants", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load module: {MODULE_PATH}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_txt_and_json_boxes_are_split_and_numbered_continuously(tmp_path: Path) -> None:
    data_root = tmp_path / "source"
    image_dir = data_root / "train" / "images"
    label_dir = data_root / "train" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (data_root / "data.yaml").write_text("nc: 1\nnames: ['IMPLANCE']\n", encoding="utf-8")

    for index in range(5):
        Image.new("RGB", (100, 100), "white").save(image_dir / f"txt_{index}.jpg")
        annotation = (
            "0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6"
            if index == 4
            else "0 0.5 0.5 0.2 0.2"
        )
        (label_dir / f"txt_{index}.txt").write_text(annotation, encoding="utf-8")
    for index in range(5):
        Image.new("RGB", (100, 100), "white").save(image_dir / f"json_{index}.jpg")
        (label_dir / f"json_{index}.json").write_text(json.dumps({"boxes": [
            {"label": "IMPLANCE", "x": "50", "y": "50", "width": "20", "height": "20"}
        ]}), encoding="utf-8")

    output_root = tmp_path / "cropped"
    summary = module.build_dataset(
        data_root,
        output_root,
        data_root / "data.yaml",
        output_size=224,
        sharpen=True,
        square=True,
    )

    assert [summary[split]["IMPLANCE"] for split in module.SPLITS] == [8, 1, 1]
    files = sorted(output_root.rglob("*.jpg"))
    assert len(files) == 10
    assert {path.stem for path in files} == {f"IMPLANCE_{number:03d}" for number in range(1, 11)}
    assert {Image.open(path).size for path in files} == {(224, 224)}


def test_multiple_boxes_from_one_image_never_cross_splits(tmp_path: Path) -> None:
    records = []
    for index in range(3):
        image_path = (tmp_path / f"source_{index}.jpg").resolve()
        records.append(module.LabeledBox(image_path, "A", (0, 0, 10, 10)))
        if index == 0:
            records.append(module.LabeledBox(image_path, "A", (10, 10, 20, 20)))

    split_records = module._split_records(records, seed=42)
    memberships = {
        split
        for split, items in split_records.items()
        if any(item.image_path == records[0].image_path for item in items)
    }
    assert len(memberships) == 1
