#!/usr/bin/env python3
from __future__ import annotations

import sys
import importlib.util
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "pipelines" / "yolo" / "crop_yolo.py"
spec = importlib.util.spec_from_file_location("crop_implant_yolo", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load module: {MODULE_PATH}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

build_output_path = module.build_output_path
crop_detection = module.crop_detection
expand_square_crop = module.expand_square_crop
sanitize_token = module.sanitize_token
predict_and_crop = module.predict_and_crop


def make_image(width: int = 100, height: int = 80) -> Image.Image:
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[:, :] = (30, 60, 90)
    array[20:60, 30:70] = (220, 50, 50)
    return Image.fromarray(array, mode="RGB")


def test_square_crop_and_resize_returns_fixed_output_size() -> None:
    image = make_image()
    crop = crop_detection(image, (30.0, 20.0, 70.0, 60.0), crop_size=512, pad=0.25)
    assert crop.size == (512, 512)


def test_expand_square_crop_keeps_square_geometry() -> None:
    image = make_image()
    square = expand_square_crop(image, (30.0, 20.0, 70.0, 60.0), pad=0.25)
    assert square.size[0] == square.size[1]
    assert square.size[0] >= 40


def test_build_output_path_uses_split_class_and_unique_name() -> None:
    source = Path("/data/Dental Implants 2.0.v3i.yolov8/train/images/Bicon-1.jpg")
    root = Path("/data/Dental Implants 2.0.v3i.yolov8/train/images")
    out = build_output_path(Path("/tmp/crops"), source, root, "Bicon", 1, 0.9876)
    assert out.parts[-3:-1] == ("train", "Bicon")
    assert out.name.startswith("Bicon-1__")
    assert out.name.endswith("_conf0.99.jpg")
    assert sanitize_token("Bicon / ITI") == "Bicon_ITI"


class _ArrayLike:
    def __init__(self, values):
        self._values = values

    def cpu(self):
        return self

    def tolist(self):
        return list(self._values)


class _FakeModel:
    names = {0: "implant"}

    def __init__(self, result):
        self._result = result

    def predict(self, **kwargs):
        return [self._result]


def test_predict_and_crop_writes_512_crop_file() -> None:
    with tempfile.TemporaryDirectory(prefix="yolo_crop_") as tmp:
        tmp_dir = Path(tmp)
        source_root = tmp_dir / "train" / "images"
        source_root.mkdir(parents=True, exist_ok=True)
        image_path = source_root / "sample.jpg"
        make_image().save(image_path)

        result = SimpleNamespace(
            path=str(image_path),
            boxes=SimpleNamespace(
                xyxy=_ArrayLike([[30.0, 20.0, 70.0, 60.0]]),
                cls=_ArrayLike([0.0]),
                conf=_ArrayLike([0.91]),
            ),
        )
        model = _FakeModel(result)

        outputs = predict_and_crop(
            model=model,
            images=[(source_root, image_path)],
            output_root=tmp_dir / "out",
            crop_size=512,
            pad=0.15,
            conf=0.25,
            iou=0.45,
            imgsz=640,
            device="cpu",
            batch_size=1,
            save_empty=False,
        )

        assert outputs[0].saved == 1
        saved_files = list((tmp_dir / "out").rglob("*.jpg"))
        assert len(saved_files) == 1
        assert Image.open(saved_files[0]).size == (512, 512)


def main() -> None:
    test_square_crop_and_resize_returns_fixed_output_size()
    test_expand_square_crop_keeps_square_geometry()
    test_build_output_path_uses_split_class_and_unique_name()
    test_predict_and_crop_writes_512_crop_file()
    print("OK: yolo crop helper tests passed")


if __name__ == "__main__":
    main()
