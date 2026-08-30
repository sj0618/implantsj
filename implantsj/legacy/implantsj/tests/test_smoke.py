from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from implantsj.evaluation import evaluate, extract_embeddings  # noqa: E402
from implantsj.manifests import build_manifests, build_parser as manifest_parser  # noqa: E402
from implantsj.models import ModelSpec, create_adapter, save_initial_checkpoint  # noqa: E402
from implantsj.training import build_parser as training_parser, train  # noqa: E402
from implantsj.visualization import build_parser as visual_parser, create_tsne  # noqa: E402


def write_image(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.random.default_rng(seed).integers(0, 256, (40, 40, 3), dtype=np.uint8)
    Image.fromarray(values).save(path)


def make_data(root: Path) -> tuple[Path, Path, Path, Path]:
    labels = ["Class A", "Class B", "Class C"]
    train_root = root / "images" / "train"
    test_root = root / "images" / "test"
    gt_root = root / "images" / "gt"
    seed = 0
    for label in labels:
        for index in range(4):
            write_image(train_root / label / f"{index}.png", seed)
            seed += 1
        for index in range(2):
            write_image(test_root / label / f"{index}.png", seed)
            seed += 1
        write_image(gt_root / label / "prototype_0.png", seed)
        write_image(gt_root / label / "prototype_1.png", seed + 100)
        seed += 1
    closed = root / "closed.txt"
    closed.write_text("Class A\nClass B\n", encoding="utf-8")
    manifests = root / "manifests"
    args = manifest_parser().parse_args(
        [
            "--train-root",
            str(train_root),
            "--test-root",
            str(test_root),
            "--gt-root",
            str(gt_root),
            "--closed-labels",
            str(closed),
            "--output-dir",
            str(manifests),
            "--valid-fraction",
            "0.25",
            "--seed",
            "11",
        ]
    )
    build_manifests(args)
    return manifests, train_root, test_root, gt_root


def train_one(manifests: Path, output: Path, loss: str) -> Path:
    args = training_parser().parse_args(
        [
            "--experiment-name",
            f"smoke-{loss}",
            "--model",
            "resnet18",
            "--no-pretrained",
            "--train-data",
            str(manifests / "train.csv"),
            "--valid-data",
            str(manifests / "valid.csv"),
            "--test-data",
            str(manifests / "test.csv"),
            "--scope",
            "all",
            "--loss",
            loss,
            "--epochs",
            "1",
            "--batch-size",
            "6",
            "--samples-per-class",
            "2",
            "--input-size",
            "32",
            "--projection-dim",
            "8",
            "--projection-hidden-dim",
            "16",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--output-dir",
            str(output),
        ]
    )
    return train(args)


def test_cpu_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    manifests, _, _, _ = make_data(tmp_path)
    ce = train_one(manifests, tmp_path / "ce", "ce")
    supcon = train_one(manifests, tmp_path / "supcon", "supcon")

    for run in (ce, supcon):
        assert (run / "checkpoints" / "last.pt").is_file()
        assert (run / "checkpoints" / "best.pt").is_file()
        assert (run / "history.csv").is_file()
        assert (run / "curves.png").is_file()
        assert (run / "model_spec.json").is_file()
        history = pd.read_csv(run / "history.csv")
        assert {"test_loss", "test_accuracy"} <= set(history)
    supcon_history = pd.read_csv(supcon / "history.csv")
    assert {
        "expected_cross_image_positive_fraction",
        "actual_cross_image_positive_fraction",
    } <= set(supcon_history)

    caches = {}
    for split in ("support", "gt", "valid", "test"):
        caches[split] = extract_embeddings(
            model_spec_path=ce / "model_spec.json",
            manifest_path=manifests / f"{split}.csv",
            output_path=tmp_path / "embeddings" / f"{split}.npz",
            batch_size=6,
            num_workers=0,
            device="cpu",
        )
    supcon_cache = extract_embeddings(
        model_spec_path=supcon / "model_spec.json",
        manifest_path=manifests / "valid.csv",
        output_path=tmp_path / "embeddings" / "supcon_valid.npz",
        batch_size=6,
        num_workers=0,
        device="cpu",
    )
    assert supcon_cache.is_file()

    eval_args = argparse.Namespace(
        output_dir=str(tmp_path / "evaluation"),
        labels_json=str(manifests / "labels.json"),
        support_embeddings=str(caches["support"]),
        gt_embeddings=str(caches["gt"]),
        query_embeddings=str(caches["test"]),
        allow_noncanonical_labels=True,
        allow_incomplete_query=False,
        prototypes="gt,mean,ss",
        distances="euclidean,cosine",
        ss_count=1,
        ss_fraction=None,
        seed=11,
        model_name="smoke-ce",
        query_name="test",
    )
    metrics_path = evaluate(eval_args)
    metrics = pd.read_csv(metrics_path)
    assert len(metrics) == 18
    assert {
        "embedding_norm_mean",
        "embedding_norm_std",
        "mean_dimension_std",
        "near_zero_dimension_count",
    } <= set(pd.read_csv(tmp_path / "evaluation" / "geometry.csv"))

    tsne_args = visual_parser().parse_args(
        [
            "tsne",
            "--model-name",
            "smoke-ce",
            "--embedding",
            f"train={caches['support']}",
            "--embedding",
            f"valid={caches['valid']}",
            "--embedding",
            f"test={caches['test']}",
            "--labels-json",
            str(manifests / "labels.json"),
            "--output-dir",
            str(tmp_path / "tsne"),
            "--max-points",
            "12",
            "--seed",
            "11",
        ]
    )
    create_tsne(tsne_args)
    for name in (
        "tsne.png",
        "tsne_coordinates.csv",
        "sampled_paths.txt",
        "tsne_metadata.json",
    ):
        assert (tmp_path / "tsne" / name).is_file()
    metadata = json.loads((tmp_path / "tsne" / "tsne_metadata.json").read_text())
    assert all(set(item) == {"split", "path"} for item in metadata["inputs"])
    assert list(pd.read_csv(tmp_path / "tsne" / "tsne_coordinates.csv")) == [
        "path",
        "label",
        "split",
        "set",
        "x",
        "y",
    ]


def test_local_initial_checkpoint_roundtrip(tmp_path):
    import torch

    checkpoint = save_initial_checkpoint(
        "resnet18", tmp_path / "base.pt", pretrained=False
    )
    adapter = create_adapter(
        ModelSpec(
            name="base",
            architecture="resnet18",
            checkpoint=str(checkpoint),
            checkpoint_format="ckpt",
            pretrained=False,
            input_size=32,
        ),
        "cpu",
    ).load()
    assert adapter.encode(torch.rand(2, 3, 32, 32)).shape == (2, 512)
