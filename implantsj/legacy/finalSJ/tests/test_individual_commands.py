from __future__ import annotations
import csv

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
    assert "run.json" not in {artifact["path"] for artifact in split_run["artifacts"]}

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
    metadata: dict | None = None,
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
        metadata={"view_sha256": view_hash, **(metadata or {})},
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


def _canonical_protocol(monkeypatch) -> tuple[Path, object]:
    from implantsj.protocol import load_protocol

    values = {
        "BASE_CHECKPOINT": "/tmp/base.pt",
        "BASE_CHECKPOINT_SHA256": "0" * 64,
        "SEED42_ASSIGNMENT_CSV_SHA256": "1" * 64,
        "SEED42_ASSIGNMENT_SHA256": "2" * 64,
        "SEED43_ASSIGNMENT_CSV_SHA256": "3" * 64,
        "SEED43_ASSIGNMENT_SHA256": "4" * 64,
        "SEED44_ASSIGNMENT_CSV_SHA256": "5" * 64,
        "SEED44_ASSIGNMENT_SHA256": "6" * 64,
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    path = Path("manifests/protocols/metric_learning_v1.yaml").resolve()
    return path, load_protocol(path)


def _write_canonical_run(directory, protocol, command, role, seed=42):
    from implantsj.protocol import artifact_inventory_hash

    (directory / "run.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "command": command,
                "role": role,
                "protocol_fingerprint": protocol.fingerprint,
                "seed": seed,
                "artifact_sha256": artifact_inventory_hash(directory),
            }
        ),
        encoding="utf-8",
    )


def _canonical_metrics_frame(key, value):
    return pd.DataFrame(
        {
            "seed": [key.seed] * 3,
            "model": [key.model] * 3,
            "query": [key.query] * 3,
            "prototype": [key.prototype] * 3,
            "distance": [key.distance] * 3,
            "candidate_space": ["full42"] * 3,
            "slice": ["all", "closed", "open"],
            "candidate_classes": [42] * 3,
            "samples": [42, 27, 15],
            "accuracy": [value] * 3,
            "macro_f1": [value] * 3,
            "weighted_f1": [value] * 3,
        }
    )


def test_canonical_evaluation_emits_all_closed_open_metrics(tmp_path, monkeypatch):
    protocol_path, protocol = _canonical_protocol(monkeypatch)
    labels = list(protocol.raw["labels"]["ordered"])
    sets = ["closed"] * 27 + ["open"] * 15
    features = np.eye(42, dtype=np.float32)
    query = _embedding_dir(
        tmp_path / "canonical-query",
        features,
        labels,
        sets,
        "validation-view",
        {
            "protocol_fingerprint": protocol.fingerprint,
            "model_role": "Base",
            "view_role": "validation",
            "lineage": {"test": "lineage"},
        },
    )
    prototype = tmp_path / "canonical-prototype"
    prototype.mkdir()
    np.savez_compressed(
        prototype / "prototypes.npz",
        vectors=features,
        labels=np.asarray(labels, dtype=str),
        sets=np.asarray(sets, dtype=str),
    )
    (prototype / "metadata.json").write_text(
        json.dumps(
            {
                "protocol_fingerprint": protocol.fingerprint,
                "model_role": "Base",
                "prototype_type": "mean",
                "distance_geometry": "euclidean",
                "lineage": {"test": "lineage"},
            }
        ),
        encoding="utf-8",
    )
    for directory, command, role in (
        (query, "extract-embeddings", "embedding"),
        (prototype, "build-prototype", "prototype"),
    ):
        _write_canonical_run(directory, protocol, command, role)

    output = tmp_path / "canonical-evaluation"
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
                "all",
                "--candidate-space",
                "full42",
                "--protocol-manifest",
                str(protocol_path),
                "--seed",
                "42",
                "--model-role",
                "Base",
                "--query-view",
                "validation",
                "--prototype-type",
                "mean",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    metrics = pd.read_csv(output / "metrics.csv")
    assert metrics["slice"].tolist() == ["all", "closed", "open"]
    assert metrics["samples"].tolist() == [42, 27, 15]
    assert metrics["accuracy"].tolist() == [1.0, 1.0, 1.0]
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["role"] == "evaluation"
    assert run["protocol_fingerprint"] == protocol.fingerprint

    test_output = tmp_path / "canonical-test-without-freeze"
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
                "all",
                "--candidate-space",
                "full42",
                "--protocol-manifest",
                str(protocol_path),
                "--seed",
                "42",
                "--model-role",
                "Base",
                "--query-view",
                "test",
                "--prototype-type",
                "mean",
                "--output",
                str(test_output),
            ]
        )
        == 10
    )
    assert not test_output.exists()


def test_canonical_comparison_emits_three_slice_rows(tmp_path, monkeypatch):
    from implantsj.protocol import protocol_lineage

    protocol_path, protocol = _canonical_protocol(monkeypatch)
    inputs = []
    for model, accuracy in (("Base", 0.5), ("ClosedFT", 0.75)):
        directory = tmp_path / model
        directory.mkdir()
        pd.DataFrame(
            {
                "slice": ["all", "closed", "open"],
                "accuracy": [accuracy] * 3,
                "macro_f1": [accuracy] * 3,
                "weighted_f1": [accuracy] * 3,
            }
        ).to_csv(directory / "metrics.csv", index=False)
        (directory / "metadata.json").write_text(
            json.dumps(
                {
                    "protocol_fingerprint": protocol.fingerprint,
                    "seed": 42,
                    "query": "validation",
                    "prototype_type": "mean",
                    "distance": "euclidean",
                    "candidate_space": "full42",
                    "model_role": model,
                    "freeze_commitment_sha256": None,
                    "lineage": protocol_lineage(
                        protocol,
                        model=model,
                        checkpoint_sha256=("a" if model == "Base" else "b") * 64,
                        treatment=protocol.raw["model_roles"][model],
                    ),
                }
            ),
            encoding="utf-8",
        )
        _write_canonical_run(directory, protocol, "evaluate-prototype", "evaluation")
        inputs.append(directory)

    output = tmp_path / "comparison"
    assert (
        main(
            [
                "compare-results",
                "--left",
                str(inputs[0]),
                "--right",
                str(inputs[1]),
                "--comparison-id",
                "1a",
                "--comparison-label",
                "Base_vs_ClosedFT",
                "--protocol-manifest",
                str(protocol_path),
                "--seed",
                "42",
                "--left-model",
                "Base",
                "--right-model",
                "ClosedFT",
                "--query-view",
                "validation",
                "--prototype-type",
                "mean",
                "--distance",
                "euclidean",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    deltas = pd.read_csv(output / "deltas.csv")
    assert deltas["slice"].tolist() == ["all", "closed", "open"]
    assert deltas["accuracy_delta"].tolist() == [0.25, 0.25, 0.25]


def test_validation_freeze_gates_exact_planned_test_cell(tmp_path, monkeypatch):
    from implantsj.protocol import (
        COMPARISON_FIELDS,
        EVALUATION_FIELDS,
        PRETEST_FIELDS,
        expected_comparison_keys,
        expected_evaluation_keys,
    )

    protocol_path, protocol = _canonical_protocol(monkeypatch)
    validation_root = tmp_path / "validation-cells"
    test_root = tmp_path / "test-cells"
    rows = []
    planned_output = None
    for key in expected_evaluation_keys(42):
        cell_name = f"{key.model}-{key.prototype}-{key.distance}"
        if key.query == "validation":
            cell = validation_root / cell_name
            cell.mkdir(parents=True)
            _canonical_metrics_frame(key, 1.0).to_csv(cell / "metrics.csv", index=False)
            _write_canonical_run(cell, protocol, "evaluate-prototype", "evaluation")
            run = json.loads((cell / "run.json").read_text(encoding="utf-8"))
            record_type, status = "completed_validation", "completed"
            artifact_sha256 = run["artifact_sha256"]
        else:
            cell = test_root / cell_name
            record_type, status, artifact_sha256 = "planned_test", "planned", ""
            if (
                key.model,
                key.prototype,
                key.distance,
            ) == ("Base", "mean", "euclidean"):
                planned_output = cell
        rows.append(
            {
                "record_type": record_type,
                "seed": 42,
                "model": key.model,
                "query": key.query,
                "prototype": key.prototype,
                "distance": key.distance,
                "candidate_space": "full42",
                "output_path": str(cell),
                "artifact_sha256": artifact_sha256,
                "status": status,
                "protocol_fingerprint": protocol.fingerprint,
            }
        )

    ledger = tmp_path / "pretest.csv"
    with ledger.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PRETEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    freeze_dir = tmp_path / "freeze"
    assert (
        main(
            [
                "freeze-validation",
                "--protocol-manifest",
                str(protocol_path),
                "--seed",
                "42",
                "--pretest-ledger",
                str(ledger),
                "--output",
                str(freeze_dir),
            ]
        )
        == 0
    )
    freeze = json.loads(
        (freeze_dir / "validation_freeze.json").read_text(encoding="utf-8")
    )
    assert len(pd.read_csv(freeze_dir / "frozen_validation_results.csv")) == 90
    assert freeze["frozen_results_sha256"]
    assert freeze["created_at"]

    labels = list(protocol.raw["labels"]["ordered"])
    sets = ["closed"] * 27 + ["open"] * 15
    features = np.eye(42, dtype=np.float32)
    query = _embedding_dir(
        tmp_path / "test-query",
        features,
        labels,
        sets,
        "test-view",
        {
            "protocol_fingerprint": protocol.fingerprint,
            "model_role": "Base",
            "view_role": "test",
            "lineage": {"test": "lineage"},
        },
    )
    _write_canonical_run(query, protocol, "extract-embeddings", "embedding")
    prototype = tmp_path / "test-prototype"
    prototype.mkdir()
    np.savez_compressed(
        prototype / "prototypes.npz",
        vectors=features,
        labels=np.asarray(labels, dtype=str),
        sets=np.asarray(sets, dtype=str),
    )
    (prototype / "metadata.json").write_text(
        json.dumps(
            {
                "protocol_fingerprint": protocol.fingerprint,
                "model_role": "Base",
                "prototype_type": "mean",
                "distance_geometry": "euclidean",
                "lineage": {"test": "lineage"},
            }
        ),
        encoding="utf-8",
    )
    _write_canonical_run(prototype, protocol, "build-prototype", "prototype")

    assert planned_output is not None
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
                "all",
                "--candidate-space",
                "full42",
                "--protocol-manifest",
                str(protocol_path),
                "--seed",
                "42",
                "--model-role",
                "Base",
                "--query-view",
                "test",
                "--prototype-type",
                "mean",
                "--validation-freeze",
                str(freeze_dir),
                "--output",
                str(planned_output),
            ]
        )
        == 0
    )
    metadata = json.loads(
        (planned_output / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["freeze_commitment_sha256"] == freeze["freeze_commitment_sha256"]
    assert metadata["validation_freeze_sha256"]
    evaluation_rows = []
    evaluation_hashes = {}
    evaluation_paths = {}
    for key in expected_evaluation_keys(42):
        cell = (
            validation_root if key.query == "validation" else test_root
        ) / f"{key.model}-{key.prototype}-{key.distance}"
        if not cell.exists():
            cell.mkdir(parents=True)
            _canonical_metrics_frame(key, 0.5).to_csv(cell / "metrics.csv", index=False)
            _write_canonical_run(cell, protocol, "evaluate-prototype", "evaluation")
        cell_run = json.loads((cell / "run.json").read_text(encoding="utf-8"))
        evaluation_hashes[key] = cell_run["artifact_sha256"]
        evaluation_paths[key] = cell
        evaluation_rows.append(
            {
                "seed": 42,
                "model": key.model,
                "query": key.query,
                "prototype": key.prototype,
                "distance": key.distance,
                "candidate_space": "full42",
                "evaluation_path": str(cell),
                "evaluation_sha256": cell_run["artifact_sha256"],
                "status": "completed",
                "protocol_fingerprint": protocol.fingerprint,
                "freeze_commitment_sha256": (
                    freeze["freeze_commitment_sha256"] if key.query == "test" else ""
                ),
            }
        )
    evaluation_ledger = tmp_path / "evaluations.csv"
    with evaluation_ledger.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVALUATION_FIELDS)
        writer.writeheader()
        writer.writerows(evaluation_rows)

    comparison_rows = []
    comparison_root = tmp_path / "comparison-cells"
    for key in expected_comparison_keys(protocol, 42):
        cell = comparison_root / (
            f"{key.experiment_id}-{key.left_model}-{key.right_model}-"
            f"{key.query}-{key.prototype}-{key.distance}"
        )
        cell.mkdir(parents=True)
        left_key = next(
            item
            for item in expected_evaluation_keys(42)
            if (
                item.model,
                item.query,
                item.prototype,
                item.distance,
            )
            == (
                key.left_model,
                key.query,
                key.prototype,
                key.distance,
            )
        )
        right_key = next(
            item
            for item in expected_evaluation_keys(42)
            if (
                item.model,
                item.query,
                item.prototype,
                item.distance,
            )
            == (
                key.right_model,
                key.query,
                key.prototype,
                key.distance,
            )
        )
        left_metrics = pd.read_csv(
            evaluation_paths[left_key] / "metrics.csv"
        ).set_index("slice")
        right_metrics = pd.read_csv(
            evaluation_paths[right_key] / "metrics.csv"
        ).set_index("slice")
        delta_rows = []
        for slice_name in ("all", "closed", "open"):
            row = {
                "seed": 42,
                "experiment_id": key.experiment_id,
                "left_model": key.left_model,
                "right_model": key.right_model,
                "query": key.query,
                "prototype": key.prototype,
                "distance": key.distance,
                "candidate_space": "full42",
                "slice": slice_name,
            }
            for metric in ("accuracy", "macro_f1", "weighted_f1"):
                left_value = float(left_metrics.loc[slice_name, metric])
                right_value = float(right_metrics.loc[slice_name, metric])
                row[f"left_{metric}"] = left_value
                row[f"right_{metric}"] = right_value
                row[f"{metric}_delta"] = right_value - left_value
            delta_rows.append(row)
        pd.DataFrame(delta_rows).to_csv(cell / "deltas.csv", index=False)
        _write_canonical_run(cell, protocol, "compare-results", "comparison")
        cell_run = json.loads((cell / "run.json").read_text(encoding="utf-8"))
        comparison_rows.append(
            {
                "seed": 42,
                "experiment_id": key.experiment_id,
                "left_model": key.left_model,
                "right_model": key.right_model,
                "query": key.query,
                "prototype": key.prototype,
                "distance": key.distance,
                "candidate_space": "full42",
                "left_evaluation_sha256": evaluation_hashes[left_key],
                "right_evaluation_sha256": evaluation_hashes[right_key],
                "comparison_path": str(cell),
                "comparison_sha256": cell_run["artifact_sha256"],
                "status": "completed",
                "protocol_fingerprint": protocol.fingerprint,
                "freeze_commitment_sha256": (
                    freeze["freeze_commitment_sha256"] if key.query == "test" else ""
                ),
            }
        )
    comparison_ledger = tmp_path / "comparisons.csv"
    with comparison_ledger.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(comparison_rows)

    audit_output = tmp_path / "audit"
    assert (
        main(
            [
                "audit-study",
                "--protocol-manifest",
                str(protocol_path),
                "--seed",
                "42",
                "--validation-freeze",
                str(freeze_dir),
                "--evaluation-ledger",
                str(evaluation_ledger),
                "--comparison-ledger",
                str(comparison_ledger),
                "--output",
                str(audit_output),
            ]
        )
        == 0
    )
    assert len(pd.read_csv(audit_output / "canonical_results.csv")) == 180
    assert len(pd.read_csv(audit_output / "comparison_results.csv")) == 216
    test_row_index = next(
        index for index, row in enumerate(evaluation_rows) if row["query"] == "test"
    )
    test_cell = Path(evaluation_rows[test_row_index]["evaluation_path"])
    original_metrics = pd.read_csv(test_cell / "metrics.csv")
    original_evaluation_sha = evaluation_rows[test_row_index]["evaluation_sha256"]
    invalid_metrics = original_metrics.copy()
    invalid_metrics.loc[0, "samples"] += 1
    invalid_metrics.to_csv(test_cell / "metrics.csv", index=False)
    _write_canonical_run(test_cell, protocol, "evaluate-prototype", "evaluation")
    invalid_run = json.loads((test_cell / "run.json").read_text(encoding="utf-8"))
    invalid_evaluation_sha = invalid_run["artifact_sha256"]
    evaluation_rows[test_row_index]["evaluation_sha256"] = invalid_evaluation_sha
    for row in comparison_rows:
        if row["left_evaluation_sha256"] == original_evaluation_sha:
            row["left_evaluation_sha256"] = invalid_evaluation_sha
        if row["right_evaluation_sha256"] == original_evaluation_sha:
            row["right_evaluation_sha256"] = invalid_evaluation_sha
    with evaluation_ledger.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVALUATION_FIELDS)
        writer.writeheader()
        writer.writerows(evaluation_rows)
    with comparison_ledger.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(comparison_rows)

    invalid_samples_output = tmp_path / "invalid-samples-audit"
    assert (
        main(
            [
                "audit-study",
                "--protocol-manifest",
                str(protocol_path),
                "--seed",
                "42",
                "--validation-freeze",
                str(freeze_dir),
                "--evaluation-ledger",
                str(evaluation_ledger),
                "--comparison-ledger",
                str(comparison_ledger),
                "--output",
                str(invalid_samples_output),
            ]
        )
        == 10
    )
    assert not invalid_samples_output.exists()

    original_metrics.to_csv(test_cell / "metrics.csv", index=False)
    _write_canonical_run(test_cell, protocol, "evaluate-prototype", "evaluation")
    restored_run = json.loads((test_cell / "run.json").read_text(encoding="utf-8"))
    restored_evaluation_sha = restored_run["artifact_sha256"]
    evaluation_rows[test_row_index]["evaluation_sha256"] = restored_evaluation_sha
    for row in comparison_rows:
        if row["left_evaluation_sha256"] == invalid_evaluation_sha:
            row["left_evaluation_sha256"] = restored_evaluation_sha
        if row["right_evaluation_sha256"] == invalid_evaluation_sha:
            row["right_evaluation_sha256"] = restored_evaluation_sha
    with evaluation_ledger.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVALUATION_FIELDS)
        writer.writeheader()
        writer.writerows(evaluation_rows)
    with comparison_ledger.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(comparison_rows)
    tampered_cell = Path(comparison_rows[0]["comparison_path"])
    tampered = pd.read_csv(tampered_cell / "deltas.csv")
    tampered.loc[0, "accuracy_delta"] += 0.25
    tampered.to_csv(tampered_cell / "deltas.csv", index=False)
    _write_canonical_run(tampered_cell, protocol, "compare-results", "comparison")
    tampered_run = json.loads((tampered_cell / "run.json").read_text(encoding="utf-8"))
    comparison_rows[0]["comparison_sha256"] = tampered_run["artifact_sha256"]
    with comparison_ledger.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(comparison_rows)

    rejected_output = tmp_path / "rejected-audit"
    assert (
        main(
            [
                "audit-study",
                "--protocol-manifest",
                str(protocol_path),
                "--seed",
                "42",
                "--validation-freeze",
                str(freeze_dir),
                "--evaluation-ledger",
                str(evaluation_ledger),
                "--comparison-ledger",
                str(comparison_ledger),
                "--output",
                str(rejected_output),
            ]
        )
        == 10
    )
    assert not rejected_output.exists()


@pytest.mark.parametrize(
    "command",
    ["freeze-validation", "audit-study", "summarize-uncertainty"],
)
def test_canonical_scalar_commands_are_registered(command):
    assert command in build_parser()._subparsers._group_actions[0].choices


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
