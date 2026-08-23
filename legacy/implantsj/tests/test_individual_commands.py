from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from implantsj.workflow.individual import build_parser, main
from implantsj.workflow.schema import ManifestError, ManifestResolver


def _image(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(
        0, 256, size=(24, 24, 3), dtype=np.uint8
    )
    Image.fromarray(pixels).save(path)


def _yaml(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return path


def _data_manifests(root: Path) -> tuple[Path, Path]:
    seed = 0
    for label in ("Closed A", "Closed B"):
        for index in range(8):
            _image(root / "known" / label / f"{index}.png", seed)
            seed += 1
        for index in range(2):
            _image(root / "prototype" / label / f"{index}.png", seed)
            seed += 1
    for label in ("Open A", "Open B"):
        for index in range(2):
            _image(root / "unknown" / label / f"{index}.png", seed)
            seed += 1
    dataset = _yaml(
        root / "manifests" / "dataset.yaml",
        {
            "schema_version": 1,
            "kind": "dataset",
            "id": "tiny",
            "duplicate_policy": "error",
            "sources": [
                {
                    "id": "known",
                    "type": "folder_label",
                    "role": "sample",
                    "path": str(root / "known"),
                },
                {
                    "id": "prototype",
                    "type": "folder_label",
                    "role": "prototype",
                    "path": str(root / "prototype"),
                },
                {
                    "id": "unknown",
                    "type": "folder_label",
                    "role": "unknown",
                    "path": str(root / "unknown"),
                },
            ],
            "labels": {"mode": "derived", "aliases": {}},
            "label_spaces": {
                "closed": ["Closed A", "Closed B"],
                "open": ["Open A", "Open B"],
            },
        },
    )
    split = _yaml(
        root / "manifests" / "split.yaml",
        {
            "schema_version": 1,
            "kind": "split",
            "id": "seed42",
            "seed": 42,
            "label_reference_view": "train",
            "split": {
                "strategy": "stratified_hash",
                "seed": 42,
                "ratios": {"train": 0.5, "validation": 0.25, "test": 0.25},
            },
            "views": [
                {
                    "id": "train",
                    "from": "all",
                    "usage": "train",
                    "filters": {"source_role": "sample", "split": "train"},
                },
                {
                    "id": "validation",
                    "from": "all",
                    "usage": "validation",
                    "filters": {
                        "source_role": "sample",
                        "split": "validation",
                    },
                },
                {
                    "id": "closed_test",
                    "from": "all",
                    "usage": "test",
                    "filters": {"source_role": "sample", "split": "test"},
                },
                {
                    "id": "gt_closed",
                    "from": "all",
                    "usage": "prototype",
                    "filters": {"source_role": "prototype"},
                },
                {
                    "id": "open_test",
                    "from": "all",
                    "usage": "unknown_test",
                    "filters": {"source_role": "unknown"},
                },
            ],
        },
    )
    return dataset, split


def test_data_split_train_and_extract_are_one_explicit_call_each(tmp_path, monkeypatch):
    dataset, split = _data_manifests(tmp_path)
    split_dir = tmp_path / "outputs" / "split"
    assert main(["validate-data", "--dataset-manifest", str(dataset)]) == 0
    assert (
        main(
            [
                "build-split",
                "--dataset-manifest",
                str(dataset),
                "--split-manifest",
                str(split),
                "--output",
                str(split_dir),
            ]
        )
        == 0
    )
    assert (split_dir / "views" / "closed_test.csv").is_file()
    split_run = json.loads((split_dir / "run.json").read_text(encoding="utf-8"))
    assert split_run["status"] == "completed"
    assert split_run["command"] == "build-split"
    assert split_run["arguments"]["output"] == str(split_dir)
    assert split_run["workflow_sha256"]
    assert {"metadata.json", "records.csv"} <= {
        artifact["path"] for artifact in split_run["artifacts"]
    }
    assert "run.json" not in {
        artifact["path"] for artifact in split_run["artifacts"]
    }

    training_manifest = _yaml(
        tmp_path / "manifests" / "training.yaml",
        {
            "schema_version": 1,
            "kind": "training",
            "id": "one-model",
            "inputs": {"train": "train", "validation": "validation"},
            "model": {"architecture": "resnet18", "pretrained": False},
            "objective": {"type": "ce"},
            "budget": {"epochs": 1},
            "loader": {"batch_size": 2, "num_workers": 0},
        },
    )
    training_calls = []

    def fake_train(namespace):
        training_calls.append(namespace)
        output = Path(namespace.output_dir)
        output.mkdir(parents=True)
        checkpoint_dir = output / "checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "best.pt").write_bytes(b"checkpoint")
        return output

    monkeypatch.setattr("implantsj.training.train", fake_train)
    train_output = tmp_path / "outputs" / "train"
    assert (
        main(
            [
                "train",
                "--dataset-manifest",
                str(dataset),
                "--split-dir",
                str(split_dir),
                "--experiment-manifest",
                str(training_manifest),
                "--seed",
                "42",
                "--output",
                str(train_output),
            ]
        )
        == 0
    )
    assert len(training_calls) == 1
    assert training_calls[0].seed == 42

    model_manifest = _yaml(
        tmp_path / "manifests" / "model.yaml",
        {
            "schema_version": 1,
            "kind": "model",
            "id": "one-model",
            "architecture": "resnet18",
            "adapter": "timm_pretrained",
            "pretrained": False,
        },
    )
    extraction_calls = []

    def fake_extract_embeddings(**kwargs):
        extraction_calls.append(kwargs)
        frame = pd.read_csv(kwargs["manifest_path"])
        from implantsj.evaluation import EmbeddingTable

        features = np.arange(len(frame) * 3, dtype=np.float32).reshape(len(frame), 3)
        return EmbeddingTable(
            features=features,
            labels=frame["label"].astype(str).to_numpy(),
            paths=frame["path"].astype(str).to_numpy(),
            splits=frame["split"].astype(str).to_numpy(),
            sets=frame["set"].astype(str).to_numpy(),
            metadata={},
        ).save(kwargs["output_path"])

    monkeypatch.setattr(
        "implantsj.evaluation.extract_embeddings", fake_extract_embeddings
    )
    embedding_output = tmp_path / "outputs" / "embedding"
    assert (
        main(
            [
                "extract-embeddings",
                "--dataset-manifest",
                str(dataset),
                "--split-dir",
                str(split_dir),
                "--model-manifest",
                str(model_manifest),
                "--view",
                "closed_test",
                "--embedding-output",
                "encoder",
                "--output",
                str(embedding_output),
            ]
        )
        == 0
    )
    assert len(extraction_calls) == 1
    assert (embedding_output / "embeddings.npz").is_file()


def _embedding_dir(
    path: Path,
    features: np.ndarray,
    labels: list[str],
    sets: list[str],
    view_hash: str,
) -> Path:
    from implantsj.evaluation import EmbeddingTable

    path.mkdir(parents=True)
    count = len(labels)
    EmbeddingTable(
        features=np.asarray(features, dtype=np.float32),
        labels=np.asarray(labels, dtype=str),
        paths=np.asarray([f"/image/{index}.png" for index in range(count)], dtype=str),
        splits=np.asarray(["test"] * count, dtype=str),
        sets=np.asarray(sets, dtype=str),
        metadata={"view_sha256": view_hash},
    ).save(path / "embeddings.npz")
    return path


def test_prototype_evaluation_visualization_probe_and_comparison_are_scalar(tmp_path):
    support = _embedding_dir(
        tmp_path / "support",
        np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [1.1, -0.1],
                [0.0, 1.0],
                [0.1, 0.9],
                [-0.1, 1.1],
            ]
        ),
        ["A", "A", "A", "B", "B", "B"],
        ["closed"] * 6,
        "support-view",
    )
    query = _embedding_dir(
        tmp_path / "query",
        np.asarray([[1.0, 0.05], [0.95, -0.05], [0.05, 1.0], [-0.05, 0.95]]),
        ["A", "A", "B", "B"],
        ["closed"] * 4,
        "query-view",
    )
    prototype = tmp_path / "prototype"
    assert (
        main(
            [
                "build-prototype",
                "--embedding-dir",
                str(support),
                "--prototype-type",
                "mean",
                "--distance-geometry",
                "euclidean",
                "--output",
                str(prototype),
            ]
        )
        == 0
    )
    evaluations = []
    for name in ("left", "right"):
        output = tmp_path / name
        assert (
            main(
                [
                    "evaluate-prototype",
                    "--query-embeddings",
                    str(query),
                    "--prototype-dir",
                    str(prototype),
                    "--distance",
                    "euclidean",
                    "--query-group",
                    "closed",
                    "--candidate-space",
                    "closed27",
                    "--output",
                    str(output),
                ]
            )
            == 0
        )
        evaluations.append(output)
    comparison = tmp_path / "comparison"
    assert (
        main(
            [
                "compare-results",
                "--left",
                str(evaluations[0]),
                "--right",
                str(evaluations[1]),
                "--comparison-id",
                "one",
                "--comparison-label",
                "left_vs_right",
                "--output",
                str(comparison),
            ]
        )
        == 0
    )
    assert (
        json.loads((comparison / "comparison.json").read_text())["right_minus_left"][
            "accuracy"
        ]
        == 0.0
    )

    visualization = tmp_path / "visualization"
    assert (
        main(
            [
                "visualize-tsne",
                "--embeddings",
                str(support),
                "--geometry",
                "cosine",
                "--seed",
                "42",
                "--perplexity",
                "2",
                "--iterations",
                "250",
                "--output",
                str(visualization),
            ]
        )
        == 0
    )
    assert (visualization / "tsne.png").is_file()

    probe = tmp_path / "probe"
    assert (
        main(
            [
                "linear-probe",
                "--train-embeddings",
                str(support),
                "--test-embeddings",
                str(query),
                "--candidate-space",
                "closed27",
                "--seed",
                "42",
                "--output",
                str(probe),
            ]
        )
        == 0
    )
    assert json.loads((probe / "metrics.json").read_text())["accuracy"] == 1.0
    assert (
        main(
            [
                "build-prototype",
                "--embedding-dir",
                str(support),
                "--prototype-type",
                "ss",
                "--samples-per-label",
                "4",
                "--support-seed",
                "42",
                "--distance-geometry",
                "euclidean",
                "--output",
                str(tmp_path / "invalid-ss"),
            ]
        )
        == 4
    )


def test_suite_and_multi_axis_values_are_not_accepted(tmp_path):
    suite = _yaml(
        tmp_path / "suite.yaml",
        {
            "schema_version": 1,
            "kind": "suite",
            "id": "forbidden",
            "experiments": ["a.yaml"],
            "output_root": "out",
        },
    )
    with pytest.raises(ManifestError):
        ManifestResolver().resolve(suite)
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(
            [
                "evaluate-prototype",
                "--query-embeddings",
                "query",
                "--prototype-dir",
                "prototype",
                "--distance",
                "euclidean,cosine",
                "--query-group",
                "closed",
                "--candidate-space",
                "closed27",
                "--output",
                "out",
            ]
        )
    assert error.value.code == 2
