#!/usr/bin/env python3
"""Smoke test for 7-label SupCon checkpoint -> 3-label metric evaluation."""
from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


class TinyBackbone(nn.Module):
    num_features = 8

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(3, self.num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).flatten(1)
        return self.fc(x)


def install_fake_timm() -> None:
    fake_timm = types.ModuleType("timm")

    def create_model(*args, **kwargs):
        return TinyBackbone()

    fake_timm.create_model = create_model
    sys.modules["timm"] = fake_timm


def install_simple_transform(eval_module) -> None:
    def build_transforms(image_size: int = 224, train: bool = False, strong_aug: bool = True):
        def transform(img: Image.Image) -> torch.Tensor:
            arr = np.asarray(img.resize((image_size, image_size)), dtype=np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))
            return torch.from_numpy(arr)

        return transform

    eval_module.build_transforms = build_transforms


def write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((10, 10, 3), value, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def make_3label_data(data_root: Path) -> None:
    values = {"Bego": 40, "Bicon": 130, "ITI": 220}
    folder_names = {"Bego": "bego", "Bicon": "Bicon", "ITI": "ITI"}
    for split in ["train", "valid", "test"]:
        for label, value in values.items():
            folder = data_root / split / folder_names[label]
            for idx in range(2):
                write_rgb(folder / f"{label}_{idx}.png", value + idx)


def make_supcon_checkpoint(eval_module, checkpoint: Path) -> None:
    model = eval_module.ProjectorMetricModel("tiny_vit", embedding_dim=4)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": {
                "model_name": "tiny_vit",
                "image_size": 8,
                "embedding_dim": 4,
            },
            "class_to_idx": {
                "ADIN": 0,
                "Dentium": 1,
                "DIONAVI": 2,
                "MIS": 3,
                "NORIS": 4,
                "nobel": 5,
                "osstem": 6,
            },
        },
        checkpoint,
    )


def main() -> None:
    install_fake_timm()

    import evaluate_3label_metric_from_7label as eval_module

    install_simple_transform(eval_module)

    with tempfile.TemporaryDirectory(prefix="supcon7_to_3label_") as tmp:
        tmp_dir = Path(tmp)
        data_root = tmp_dir / "3label"
        manifest = tmp_dir / "manifests" / "3label.csv"
        output_root = tmp_dir / "outputs"
        checkpoint = tmp_dir / "checkpoints" / "supcon_7label.pt"

        make_3label_data(data_root)
        make_supcon_checkpoint(eval_module, checkpoint)

        old_argv = sys.argv[:]
        try:
            sys.argv = [
                "evaluate_3label_metric_from_7label.py",
                "--checkpoint",
                str(checkpoint),
                "--data-root",
                str(data_root),
                "--manifest",
                str(manifest),
                "--output-root",
                str(output_root),
                "--experiment-name",
                "smoke_supcon7_to_3label",
                "--model-name",
                "tiny_vit",
                "--image-size",
                "8",
                "--proj-output-dim",
                "4",
                "--batch-size",
                "2",
                "--num-workers",
                "0",
                "--allow-cpu",
            ]
            eval_module.main()
        finally:
            sys.argv = old_argv

        run_dirs = sorted(output_root.glob("*_smoke_supcon7_to_3label"))
        assert len(run_dirs) == 1, f"Expected one output run, got {run_dirs}"
        run_dir = run_dirs[0]

        args_json = json.loads((run_dir / "args.json").read_text(encoding="utf-8"))
        assert args_json["model_kind"] == "metric_model_projector"
        assert args_json["use_projection"] is True
        for split in ["valid", "test"]:
            assert (run_dir / "metrics" / f"{split}_prototype_metrics.json").is_file()
            assert (run_dir / "metrics" / f"{split}_prototype_predictions.csv").is_file()
            assert (run_dir / "confusion_matrices" / f"{split}_prototype.csv").is_file()

    print("OK: 7-label SupCon checkpoint -> 3-label metric evaluation smoke test passed")


if __name__ == "__main__":
    main()
