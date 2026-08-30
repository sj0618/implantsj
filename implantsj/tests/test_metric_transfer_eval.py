#!/usr/bin/env python3
"""Smoke test for generic metric checkpoint -> held-out manifest evaluation."""
from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipelines" / "eval"))
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


def install_simple_transform() -> None:
    import src.data as data_module

    def build_transforms(image_size: int = 224, train: bool = False, strong_aug: bool = True):
        def transform(img: Image.Image) -> torch.Tensor:
            arr = np.asarray(img.resize((image_size, image_size)), dtype=np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))
            return torch.from_numpy(arr)

        return transform

    data_module.build_transforms = build_transforms


def write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((10, 10, 3), value, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def make_manifest(manifest: Path) -> None:
    rows = []
    novel_values = {"NovelA": 30, "NovelB": 120, "NovelC": 220}
    for split in ["train", "valid", "test"]:
        for label, base in novel_values.items():
            for idx in range(2):
                img_path = manifest.parent / "images" / split / label / f"{label}_{idx}.png"
                write_rgb(img_path, base + idx)
                rows.append({"path": str(img_path.resolve()), "label": label, "split": split})
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest, index=False)


def make_metric_checkpoint(checkpoint: Path) -> None:
    from src.models import MetricModel

    model = MetricModel("tiny_vit", embedding_dim=4, pretrained=False)
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
                "Seen1": 0,
                "Seen2": 1,
            },
        },
        checkpoint,
    )


def main() -> None:
    install_fake_timm()
    install_simple_transform()

    import evaluate_metric_transfer as eval_module

    with tempfile.TemporaryDirectory(prefix="metric_transfer_eval_") as tmp:
        tmp_dir = Path(tmp)
        manifest = tmp_dir / "manifests" / "unseen15_like.csv"
        output_root = tmp_dir / "outputs"
        checkpoint = tmp_dir / "checkpoints" / "metric_seen27.pt"

        make_manifest(manifest)
        make_metric_checkpoint(checkpoint)

        old_argv = sys.argv[:]
        try:
            sys.argv = [
                "evaluate_metric_transfer.py",
                "--checkpoint",
                str(checkpoint),
                "--manifest",
                str(manifest),
                "--output-root",
                str(output_root),
                "--experiment-name",
                "smoke_metric_transfer",
                "--batch-size",
                "2",
                "--num-workers",
                "0",
                "--allow-cpu",
                "--require-disjoint",
            ]
            eval_module.main()
        finally:
            sys.argv = old_argv

        run_dirs = sorted(output_root.glob("*_smoke_metric_transfer"))
        assert len(run_dirs) == 1, f"Expected one output run, got {run_dirs}"
        run_dir = run_dirs[0]

        args_json = json.loads((run_dir / "args.json").read_text(encoding="utf-8"))
        assert args_json["eval_labels"] == ["NovelA", "NovelB", "NovelC"]
        assert args_json["overlap_labels"] == []
        for split in ["valid", "test"]:
            assert (run_dir / "metrics" / f"{split}_prototype_metrics.json").is_file()
            assert (run_dir / "metrics" / f"{split}_prototype_predictions.csv").is_file()
            assert (run_dir / "confusion_matrices" / f"{split}_prototype.csv").is_file()

    print("OK: generic metric transfer evaluation smoke test passed")


if __name__ == "__main__":
    main()
