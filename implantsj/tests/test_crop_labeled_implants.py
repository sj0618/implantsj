#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "yolo" / "crop_labeled_implants.py"
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

    Image.new("RGB", (100, 100), "white").save(image_dir / "txt.jpg")
    (label_dir / "txt.txt").write_text("\n".join(
        ["0 0.5 0.5 0.2 0.2"] * 4 + ["0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6"]
    ), encoding="utf-8")
    Image.new("RGB", (100, 100), "white").save(image_dir / "json.jpg")
    (label_dir / "json.json").write_text(json.dumps({"boxes": [
        {"label": "IMPLANCE", "x": "50", "y": "50", "width": "20", "height": "20"}
        for _ in range(5)
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
