from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from implantsj.workflow.engine import (  # noqa: E402
    inspect_checkpoint,
    resume_experiment,
    run_experiment,
    validate_experiment,
)
from implantsj.workflow.materialize import (  # noqa: E402
    canonical_isolation_audit,
    canonical_assignment_sha256,
    canonical_materialized_metadata,
    canonical_table_identity,
    materialize_dataset,
    validate_canonical_isolation,
    validate_canonical_table,
    validate_common_gt_identity,
    validate_distinct_assignments,
    validate_exact_label_coverage,
    validate_same_universe,
)
from implantsj.workflow.schema import (  # noqa: E402
    DataError,
    ManifestError,
    ManifestResolver,
    TaskConfigError,
)


def image(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(
        0, 256, size=(32, 32, 3), dtype=np.uint8
    )
    Image.fromarray(pixels).save(path)


def dump(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return path


def loader(*, train: bool) -> dict:
    return {
        "batch_size": 4,
        "num_workers": 0,
        "shuffle": train,
        "sampler": "none" if train else "sequential",
        "drop_last": False,
        "pin_memory": False,
        "persistent_workers": False,
    }


def transform_manifest() -> dict:
    return {
        "schema_version": 1,
        "kind": "transform",
        "id": "tiny-transform",
        "pipelines": {
            "train": [
                {"name": "resize", "params": {"size": [32, 32]}},
                {"name": "to_tensor"},
            ],
            "evaluation": [
                {"name": "resize", "params": {"size": [32, 32]}},
                {"name": "to_tensor"},
            ],
        },
    }


def dataset_manifest(
    sample_csv: Path,
    prototype_csv: Path,
    unknown_csv: Path,
    *,
    sample_count: int = 1,
) -> dict:
    return {
        "schema_version": 1,
        "kind": "dataset",
        "id": "tiny-data",
        "duplicate_policy": "error",
        "sources": [
            {
                "id": "samples",
                "type": "csv",
                "role": "sample",
                "path": str(sample_csv),
                "columns": {
                    "image_uri": "image",
                    "label": "class",
                    "split": "partition",
                },
            },
            {
                "id": "prototypes",
                "type": "csv",
                "role": "prototype",
                "path": str(prototype_csv),
                "columns": {"image_uri": "image", "label": "class"},
            },
            {
                "id": "unknown",
                "type": "csv",
                "role": "unknown",
                "path": str(unknown_csv),
                "columns": {"image_uri": "image", "label": "class"},
            },
        ],
        "labels": {
            "mode": "derived",
            "reference_view": "train",
            "aliases": {"Class-A": "Class A"},
        },
        "split": {
            "strategy": "predefined",
            "mapping": {"train": "train", "validation": "validation", "test": "test"},
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
                "filters": {"source_role": "sample", "split": "validation"},
            },
            {
                "id": "test",
                "from": "all",
                "usage": "test",
                "filters": {"source_role": "sample", "split": "test"},
            },
            {
                "id": "prototype",
                "from": "all",
                "usage": "prototype",
                "filters": {"source_role": "prototype"},
            },
            {
                "id": "prototype_n",
                "from": "prototype",
                "usage": "prototype",
                "sampling": {
                    "type": "per_label",
                    "count": sample_count,
                    "seed": 42,
                },
            },
            {
                "id": "unknown_test",
                "from": "all",
                "usage": "unknown_test",
                "filters": {"source_role": "unknown"},
            },
        ],
    }


def make_tables(root: Path) -> tuple[Path, Path, Path]:
    samples = []
    prototypes = []
    unknown = []
    seed = 0
    for label in ("Class A", "Class B"):
        for split, count in (("train", 4), ("validation", 2), ("test", 2)):
            for index in range(count):
                path = root / "images" / split / label / f"{index}.png"
                image(path, seed)
                seed += 1
                samples.append(
                    {
                        "image": str(path),
                        "class": " Class-A "
                        if label == "Class A" and index == 0
                        else label,
                        "partition": split,
                    }
                )
        for index in range(2):
            path = root / "images" / "prototype" / label / f"{index}.png"
            image(path, seed)
            seed += 1
            prototypes.append({"image": str(path), "class": label})
    for label in ("Open A", "Open B"):
        for index in range(2):
            path = root / "images" / "unknown" / label / f"{index}.png"
            image(path, seed)
            seed += 1
            unknown.append({"image": str(path), "class": label})
    sample_csv = root / "samples.csv"
    prototype_csv = root / "prototypes.csv"
    unknown_csv = root / "unknown.csv"
    pd.DataFrame(samples).to_csv(sample_csv, index=False)
    pd.DataFrame(prototypes).to_csv(prototype_csv, index=False)
    pd.DataFrame(unknown).to_csv(unknown_csv, index=False)
    return sample_csv, prototype_csv, unknown_csv


def experiment_manifest(
    dataset_path: Path, transform_path: Path, output: Path, *, stochastic_eval=False
) -> dict:
    if stochastic_eval:
        transforms = yaml.safe_load(transform_path.read_text())
        transforms["pipelines"]["evaluation"].insert(
            0, {"name": "random_horizontal_flip", "params": {"p": 0.5}}
        )
        dump(transform_path, transforms)
    return {
        "schema_version": 1,
        "kind": "experiment",
        "id": "tiny-experiment",
        "datasets": {"data": str(dataset_path)},
        "transform": str(transform_path),
        "seed": 42,
        "determinism": "strict",
        "runtime": {"device": "cpu", "precision": "float32"},
        "output_root": str(output),
        "artifact_policy": {
            "save_predictions": True,
            "save_embeddings": True,
        },
        "stages": [
            {
                "id": "train",
                "task": "classification_training",
                "inputs": {"train": "data.train", "validation": "data.validation"},
                "transforms": {"train": "train", "validation": "evaluation"},
                "loaders": {
                    "train": loader(train=True),
                    "validation": loader(train=False),
                },
                "model": {
                    "identifier": "resnet18",
                    "parameters": {},
                    "num_classes": "auto",
                },
                "initialization": {"type": "random", "strict": True},
                "objective": {"type": "cross_entropy", "label_smoothing": 0.0},
                "optimizer": {
                    "type": "sgd",
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "momentum": 0.0,
                },
                "scheduler": {"type": "none"},
                "budget": {"epochs": 1, "gradient_accumulation": 1},
                "checkpoint": {
                    "selection_metric": "validation_macro_f1",
                    "mode": "max",
                    "save_format": "pt",
                },
            },
            {
                "id": "evaluate",
                "task": "classification_evaluation",
                "inputs": {
                    "test": "data.test",
                    "checkpoint": "train.best_checkpoint",
                },
                "transforms": {"test": "evaluation"},
                "loaders": {"test": loader(train=False)},
            },
            {
                "id": "embed_prototype",
                "task": "embedding_extraction",
                "inputs": {
                    "data": "data.prototype_n",
                    "checkpoint": "train.best_checkpoint",
                },
                "transforms": {"data": "evaluation"},
                "loaders": {"data": loader(train=False)},
            },
            {
                "id": "embed_test",
                "task": "embedding_extraction",
                "inputs": {
                    "data": "data.test",
                    "checkpoint": "train.best_checkpoint",
                },
                "transforms": {"data": "evaluation"},
                "loaders": {"data": loader(train=False)},
            },
            {
                "id": "embed_validation",
                "task": "embedding_extraction",
                "inputs": {
                    "data": "data.validation",
                    "checkpoint": "train.best_checkpoint",
                },
                "transforms": {"data": "evaluation"},
                "loaders": {"data": loader(train=False)},
            },
            {
                "id": "embed_unknown",
                "task": "embedding_extraction",
                "inputs": {
                    "data": "data.unknown_test",
                    "checkpoint": "train.best_checkpoint",
                },
                "transforms": {"data": "evaluation"},
                "loaders": {"data": loader(train=False)},
            },
            {
                "id": "prototype_test",
                "task": "prototype_evaluation",
                "inputs": {
                    "prototype": "embed_prototype.embeddings",
                    "query": "embed_test.embeddings",
                },
                "prototype": {
                    "reducer": "mean",
                    "distance": "cosine",
                    "normalize": True,
                },
            },
            {
                "id": "retrieval_test",
                "task": "retrieval_evaluation",
                "inputs": {
                    "query": "embed_test.embeddings",
                    "gallery": "embed_prototype.embeddings",
                },
                "retrieval": {
                    "k": [1],
                    "distance": "cosine",
                    "normalize": True,
                },
            },
            {
                "id": "open_set_test",
                "task": "open_set_evaluation",
                "inputs": {
                    "calibration": "embed_validation.embeddings",
                    "known": "embed_test.embeddings",
                    "unknown": "embed_unknown.embeddings",
                    "prototypes": "prototype_test.prototypes",
                },
                "open_set": {
                    "score": "max_cosine",
                    "threshold_method": "known_tpr",
                    "target_tpr": 0.95,
                },
            },
            {
                "id": "visualize_test",
                "task": "embedding_visualization",
                "inputs": {"test": "embed_test.embeddings"},
                "visualization": {
                    "algorithm": "tsne",
                    "seed": 42,
                    "max_points": 0,
                    "perplexity": 2,
                    "iterations": 250,
                },
            },
        ],
    }


def build_fixture(root: Path, **kwargs):
    sample_csv, prototype_csv, unknown_csv = make_tables(root)
    dataset = dump(
        root / "manifests" / "dataset.yaml",
        dataset_manifest(sample_csv, prototype_csv, unknown_csv, **kwargs),
    )
    transform = dump(root / "manifests" / "transform.yaml", transform_manifest())
    experiment = dump(
        root / "manifests" / "experiment.yaml",
        experiment_manifest(dataset, transform, root / "outputs"),
    )
    return dataset, transform, experiment


def test_resolver_preserves_parent_path_origin_and_scalar_override(tmp_path):
    _, _, parent = build_fixture(tmp_path / "parent")
    child_dir = tmp_path / "child"
    child = dump(
        child_dir / "child.yaml",
        {
            "schema_version": 1,
            "kind": "experiment",
            "id": "child-experiment",
            "extends": str(parent),
        },
    )
    resolved = ManifestResolver().resolve(child, ["seed=99"])
    assert resolved.data["seed"] == 99
    assert resolved.data["output_root"] == str(
        (tmp_path / "parent" / "outputs").resolve()
    )
    assert resolved.data["datasets"]["data"]["kind"] == "dataset"
    assert resolved.overrides[0]["field"] == "seed"


def test_unknown_manifest_field_is_rejected(tmp_path):
    dataset, _, _ = build_fixture(tmp_path)
    raw = yaml.safe_load(dataset.read_text())
    raw["mystery"] = True
    dump(dataset, raw)
    with pytest.raises(ManifestError, match="unknown fields"):
        ManifestResolver().resolve(dataset)


def test_materialization_is_order_independent_and_sampling_is_exact(tmp_path):
    dataset_path, _, _ = build_fixture(tmp_path, sample_count=2)
    first_config = ManifestResolver().resolve(dataset_path).data
    first = materialize_dataset(first_config, 42)
    sample_csv = Path(first_config["sources"][0]["path"])
    frame = pd.read_csv(sample_csv).sample(frac=1, random_state=7)
    frame.to_csv(sample_csv, index=False)
    second_config = ManifestResolver().resolve(dataset_path).data
    second = materialize_dataset(second_config, 42)
    assert first.hashes["dataset"] == second.hashes["dataset"]
    assert first.hashes["view:prototype_n"] == second.hashes["view:prototype_n"]
    assert len(first.views["prototype_n"]) == 4
    assert first.label_map == {"Class A": 0, "Class B": 1}


def canonical_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "sample-a",
                "image_uri": "a.png",
                "label": "Closed",
                "group_id": "group-a",
                "split": "train",
                "content_hash": "content-a",
                "source_role": "sample",
            },
            {
                "sample_id": "sample-b",
                "image_uri": "b.png",
                "label": "Open",
                "group_id": "group-b",
                "split": "validation",
                "content_hash": "content-b",
                "source_role": "sample",
            },
        ]
    )


def test_canonical_assignment_identity_and_table_validation():
    rows = canonical_rows()
    assert canonical_assignment_sha256(rows) == canonical_assignment_sha256(
        rows.iloc[::-1]
    )
    assert canonical_table_identity(rows) == canonical_table_identity(rows.iloc[::-1])

    changed = rows.copy()
    changed.loc[0, "split"] = "test"
    assert canonical_assignment_sha256(rows) != canonical_assignment_sha256(changed)

    duplicate = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataError, match="duplicate sample_id assignments"):
        canonical_assignment_sha256(duplicate)
    with pytest.raises(DataError, match="duplicate sample_id"):
        validate_canonical_table(duplicate, role="sample", require_split=True)

    for field in ("sample_id", "group_id"):
        missing = rows.copy()
        missing.loc[0, field] = None
        with pytest.raises(DataError, match="value is null"):
            validate_canonical_table(missing, role="sample", require_split=True)

    invalid = rows.copy()
    invalid.loc[0, "split"] = "holdout"
    with pytest.raises(DataError, match="invalid predefined splits"):
        validate_canonical_table(invalid, role="sample", require_split=True)
    with pytest.raises(DataError, match="source_role"):
        validate_canonical_table(rows, role="prototype", require_split=False)
    with pytest.raises(DataError, match="duplicate content"):
        validate_distinct_assignments({"seed-42": rows, "seed-43": rows.iloc[::-1]})


def test_canonical_universe_gt_coverage_and_isolation():
    rows = canonical_rows()
    altered = rows.copy()
    altered.loc[0, "image_uri"] = "other.png"
    with pytest.raises(DataError, match="universe differs"):
        validate_same_universe(rows, altered)

    gt = rows.drop(columns=["split"]).copy()
    gt["source_role"] = "prototype"
    assert validate_common_gt_identity({"one": gt, "two": gt.iloc[::-1]})
    changed_gt = gt.copy()
    changed_gt.loc[0, "content_hash"] = "changed"
    with pytest.raises(DataError, match="content differs"):
        validate_common_gt_identity({"one": gt, "two": changed_gt})

    complete = pd.DataFrame({"label": ["Closed", "Open", "Closed", "Open"]})
    validate_exact_label_coverage(
        {"train": complete},
        closed_labels=["Closed"],
        open_labels=["Open"],
        minimums={"train": 2},
    )
    with pytest.raises(DataError, match="fewer than 3"):
        validate_exact_label_coverage(
            {"train": complete},
            closed_labels=["Closed"],
            open_labels=["Open"],
            minimums={"train": 3},
        )
    with pytest.raises(DataError, match="labels must exactly match"):
        validate_exact_label_coverage(
            {"train": complete.iloc[:1]},
            closed_labels=["Closed"],
            open_labels=["Open"],
            minimums={"train": 1},
        )

    train = rows.iloc[[0]].copy()
    validation = rows.iloc[[1]].copy()
    test = rows.iloc[[1]].copy()
    test["sample_id"] = "sample-c"
    test["group_id"] = "group-c"
    test["content_hash"] = "content-c"
    gt_isolated = rows.iloc[[1]].copy()
    gt_isolated["sample_id"] = "gt-a"
    gt_isolated["group_id"] = "gt-group"
    gt_isolated["content_hash"] = "gt-content"
    validate_canonical_isolation(
        {
            "train": train,
            "validation": validation,
            "test": test,
            "gt": gt_isolated,
        }
    )
    audit = canonical_isolation_audit(
        {
            "train": train,
            "validation": validation,
            "test": test,
            "gt": gt_isolated,
        }
    )
    assert audit["pairwise_disjoint"] == {
        "sample_id": True,
        "content_hash": True,
        "group_id": True,
    }
    for field in ("sample_id", "content_hash", "group_id"):
        overlapping = gt_isolated.copy()
        overlapping.loc[overlapping.index[0], field] = train.iloc[0][field]
        with pytest.raises(DataError, match=f"{field} overlap"):
            validate_canonical_isolation(
                {
                    "train": train,
                    "validation": validation,
                    "test": test,
                    "gt": overlapping,
                }
            )


def test_canonical_metadata_is_opt_in_for_legacy_materialization(tmp_path):
    dataset_path, _, _ = build_fixture(tmp_path)
    config = ManifestResolver().resolve(dataset_path).data
    materialized = materialize_dataset(config, 42)
    assert materialized.summary["dataset_id"] == "tiny-data"
    with pytest.raises(DataError, match="group_id"):
        canonical_materialized_metadata(materialized)


def test_per_label_shortage_fails_instead_of_silently_shrinking(tmp_path):
    dataset_path, _, _ = build_fixture(tmp_path, sample_count=3)
    config = ManifestResolver().resolve(dataset_path).data
    with pytest.raises(DataError, match="needs 3"):
        materialize_dataset(config, 42)


def test_stochastic_evaluation_transform_is_rejected_before_run(tmp_path):
    dataset, transform, _ = build_fixture(tmp_path)
    experiment = dump(
        tmp_path / "manifests" / "stochastic.yaml",
        experiment_manifest(
            dataset, transform, tmp_path / "outputs", stochastic_eval=True
        ),
    )
    resolved = ManifestResolver().resolve(experiment)
    with pytest.raises(TaskConfigError, match="stochastic transform"):
        validate_experiment(resolved)


def test_supcon_pretrain_then_finetune_keeps_projection_when_declared(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    dataset, transform, _ = build_fixture(tmp_path)
    common_model = {
        "identifier": "resnet18",
        "parameters": {},
        "num_classes": "auto",
    }
    experiment = dump(
        tmp_path / "manifests" / "supcon-transfer.yaml",
        {
            "schema_version": 1,
            "kind": "experiment",
            "id": "supcon-transfer",
            "datasets": {"data": str(dataset)},
            "transform": str(transform),
            "seed": 42,
            "determinism": "strict",
            "runtime": {"device": "cpu", "precision": "float32"},
            "output_root": str(tmp_path / "outputs"),
            "stages": [
                {
                    "id": "pretrain",
                    "task": "supcon_pretraining",
                    "inputs": {
                        "train": "data.train",
                        "validation": "data.validation",
                    },
                    "transforms": {
                        "train": "train",
                        "validation": "evaluation",
                    },
                    "loaders": {
                        "train": loader(train=True),
                        "validation": loader(train=False),
                    },
                    "model": common_model,
                    "initialization": {"type": "random", "strict": True},
                    "objective": {
                        "type": "supervised_contrastive",
                        "views": 2,
                        "temperature": 0.07,
                        "projection_head": True,
                        "projection_dim": 8,
                        "projection_hidden_dim": 16,
                    },
                    "optimizer": {
                        "type": "sgd",
                        "learning_rate": 0.001,
                        "weight_decay": 0.0,
                        "momentum": 0.0,
                    },
                    "scheduler": {"type": "none"},
                    "budget": {"epochs": 1, "gradient_accumulation": 1},
                    "checkpoint": {
                        "selection_metric": "validation_loss",
                        "mode": "min",
                        "save_format": "pt",
                    },
                },
                {
                    "id": "finetune",
                    "task": "classification_finetuning",
                    "depends_on": ["pretrain"],
                    "inputs": {
                        "train": "data.train",
                        "validation": "data.validation",
                    },
                    "transforms": {
                        "train": "train",
                        "validation": "evaluation",
                    },
                    "loaders": {
                        "train": loader(train=True),
                        "validation": loader(train=False),
                    },
                    "model": common_model,
                    "initialization": {
                        "type": "stage",
                        "source_stage": "pretrain",
                        "strict": False,
                        "freeze_encoder": False,
                        "reset_optimizer": True,
                        "projection_head": "keep",
                        "projection_dim": 8,
                        "projection_hidden_dim": 16,
                    },
                    "objective": {
                        "type": "cross_entropy",
                        "label_smoothing": 0.0,
                    },
                    "optimizer": {
                        "type": "sgd",
                        "learning_rate": 0.001,
                        "weight_decay": 0.0,
                        "momentum": 0.0,
                    },
                    "scheduler": {"type": "none"},
                    "budget": {"epochs": 1, "gradient_accumulation": 1},
                    "checkpoint": {
                        "selection_metric": "validation_macro_f1",
                        "mode": "max",
                        "save_format": "pt",
                    },
                },
                {
                    "id": "joint",
                    "task": "joint_supcon_training",
                    "inputs": {
                        "train": "data.train",
                        "validation": "data.validation",
                    },
                    "transforms": {
                        "train": "train",
                        "validation": "evaluation",
                    },
                    "loaders": {
                        "train": loader(train=True),
                        "validation": loader(train=False),
                    },
                    "model": common_model,
                    "initialization": {"type": "random", "strict": True},
                    "objective": {
                        "type": "joint_supcon",
                        "views": 2,
                        "temperature": 0.07,
                        "projection_head": True,
                        "projection_dim": 8,
                        "projection_hidden_dim": 16,
                        "classification_weight": 1.0,
                        "contrastive_weight": 0.1,
                        "classification_views": "all",
                        "classification_reduction": "mean",
                    },
                    "optimizer": {
                        "type": "sgd",
                        "learning_rate": 0.001,
                        "weight_decay": 0.0,
                        "momentum": 0.0,
                    },
                    "scheduler": {"type": "none"},
                    "budget": {"epochs": 1, "gradient_accumulation": 1},
                    "checkpoint": {
                        "selection_metric": "validation_macro_f1",
                        "mode": "max",
                        "save_format": "pt",
                    },
                },
                {
                    "id": "finetune_eval",
                    "task": "classification_evaluation",
                    "inputs": {
                        "test": "data.test",
                        "checkpoint": "finetune.best_checkpoint",
                    },
                    "transforms": {"test": "evaluation"},
                    "loaders": {"test": loader(train=False)},
                },
            ],
        },
    )
    run_dir = run_experiment(ManifestResolver().resolve(experiment))
    import torch

    checkpoint = torch.load(
        run_dir / "stages" / "finetune" / "checkpoints" / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["model_descriptor"]["kind"] == "finetune_with_projection"
    assert any(key.startswith("projection.") for key in checkpoint["model_state_dict"])
    joint_checkpoint = torch.load(
        run_dir / "stages" / "joint" / "checkpoints" / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert joint_checkpoint["model_descriptor"]["kind"] == "joint_supcon"
    assert (run_dir / "stages" / "finetune_eval" / "metrics.json").is_file()


def test_one_epoch_end_to_end_writes_standard_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    _, _, experiment = build_fixture(tmp_path)
    run_dir = run_experiment(ManifestResolver().resolve(experiment))
    assert json.loads((run_dir / "status.json").read_text())["status"] == "completed"
    for name in (
        "resolved_manifest.yaml",
        "source_manifests.json",
        "run_metadata.json",
        "environment.json",
        "git.json",
        "dataset_summary.json",
        "label_map.json",
        "hashes.json",
        "stage_artifacts.json",
    ):
        assert (run_dir / name).is_file()
    assert (run_dir / "views" / "data.train.parquet").is_file()
    finalized = yaml.safe_load((run_dir / "resolved_manifest.yaml").read_text())
    assert finalized["materialized"]["datasets"]["data"]["num_classes"] == 2
    assert finalized["stages"][0]["model"]["num_classes"] == 2
    assert (run_dir / "stages" / "train" / "checkpoints" / "best.pt").is_file()
    assert (run_dir / "stages" / "evaluate" / "predictions.parquet").is_file()
    metrics = json.loads(
        (run_dir / "stages" / "prototype_test" / "metrics.json").read_text()
    )
    assert {"accuracy", "macro_f1", "weighted_f1"} <= set(metrics)
    assert (run_dir / "stages" / "retrieval_test" / "metrics.json").is_file()
    open_metrics = json.loads(
        (run_dir / "stages" / "open_set_test" / "metrics.json").read_text()
    )
    assert {"auroc", "aupr", "fpr@95tpr", "oscr"} <= set(open_metrics)
    assert (run_dir / "stages" / "visualize_test" / "coordinates.csv").is_file()
    checkpoint = run_dir / "stages" / "train" / "checkpoints" / "best.pt"
    inspection = inspect_checkpoint(checkpoint)
    assert inspection["metadata"]["checkpoint_schema_version"] == 1
    status = json.loads((run_dir / "status.json").read_text())
    status["status"] = "interrupted"
    (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
    artifacts = json.loads((run_dir / "stage_artifacts.json").read_text())
    artifacts.pop("visualize_test")
    (run_dir / "stage_artifacts.json").write_text(
        json.dumps(artifacts), encoding="utf-8"
    )
    assert resume_experiment(run_dir) == run_dir
    assert json.loads((run_dir / "status.json").read_text())["status"] == "completed"

    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    legacy_checkpoint = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": payload["model_state_dict"]}, legacy_checkpoint)
    label_map_path = tmp_path / "legacy_labels.json"
    label_map_path.write_text(json.dumps(payload["label_map"]), encoding="utf-8")
    catalog = dump(
        tmp_path / "manifests" / "catalog.yaml",
        {
            "schema_version": 1,
            "kind": "checkpoint_catalog",
            "id": "tiny-catalog",
            "checkpoints": {
                "legacy": {
                    "path": str(legacy_checkpoint),
                    "model": "resnet18",
                    "adapter": "model_state_dict",
                    "label_map": str(label_map_path),
                    "num_classes": 2,
                    "training_state": "complete",
                    "completed_epoch": 1,
                    "planned_epoch": 1,
                    "validation_status": "unverified_preprocessing",
                }
            },
        },
    )
    legacy_experiment = dump(
        tmp_path / "manifests" / "legacy-evaluation.yaml",
        {
            "schema_version": 1,
            "kind": "experiment",
            "id": "legacy-evaluation",
            "datasets": {"data": str(tmp_path / "manifests" / "dataset.yaml")},
            "transform": str(tmp_path / "manifests" / "transform.yaml"),
            "checkpoint_catalog": str(catalog),
            "seed": 42,
            "determinism": "strict",
            "runtime": {"device": "cpu", "precision": "float32"},
            "output_root": str(tmp_path / "outputs"),
            "stages": [
                {
                    "id": "legacy_test",
                    "task": "classification_evaluation",
                    "inputs": {"test": "data.test"},
                    "transforms": {"test": "evaluation"},
                    "loaders": {"test": loader(train=False)},
                    "initialization": {
                        "type": "catalog",
                        "checkpoint": "legacy",
                        "strict": True,
                    },
                }
            ],
        },
    )
    legacy_run = run_experiment(ManifestResolver().resolve(legacy_experiment))
    legacy_metrics = json.loads(
        (legacy_run / "stages" / "legacy_test" / "metrics.json").read_text()
    )
    assert legacy_metrics["preprocessing_status"] == "unverified_preprocessing"
    assert legacy_metrics["official_comparison_eligible"] is False
