"""Explicit one-command/one-condition experiment operations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from .materialize import MaterializedDataset, materialize_dataset
from .schema import DataError, ManifestError, WorkflowError


ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
COMMON_FIELDS = {"schema_version", "kind", "id"}
DATASET_FIELDS = COMMON_FIELDS | {
    "duplicate_policy",
    "sources",
    "labels",
    "label_spaces",
}
SOURCE_FIELDS = {
    "id",
    "type",
    "role",
    "path",
    "image_root",
    "columns",
    "extensions",
}
SPLIT_FIELDS = COMMON_FIELDS | {
    "seed",
    "split",
    "views",
    "label_reference_view",
}
TRAINING_FIELDS = COMMON_FIELDS | {
    "inputs",
    "model",
    "scope",
    "objective",
    "optimizer",
    "scheduler",
    "budget",
    "loader",
    "checkpoint",
    "device",
}
MODEL_FIELDS = COMMON_FIELDS | {
    "architecture",
    "adapter",
    "pretrained",
    "embedding_source",
    "input_size",
    "projection_dim",
    "projection_hidden_dim",
    "checkpoint",
    "checkpoint_format",
    "num_classes",
}
LABEL_FIELDS = {"mode", "values", "aliases"}
LABEL_SPACE_FIELDS = {"closed", "open"}
TRAINING_NESTED_FIELDS = {
    "inputs": {"train", "validation"},
    "model": {
        "architecture",
        "pretrained",
        "input_size",
        "projection_dim",
        "projection_hidden_dim",
    },
    "objective": {
        "type",
        "label_smoothing",
        "temperature",
        "margin",
    },
    "optimizer": {"type", "learning_rate", "weight_decay", "momentum"},
    "scheduler": {"type", "step_size", "gamma"},
    "budget": {"epochs"},
    "loader": {"batch_size", "num_workers", "samples_per_class"},
    "checkpoint": {"input_format", "save_format"},
}


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _substitute_env(value: Any, path: Path) -> Any:
    if isinstance(value, dict):
        return {key: _substitute_env(item, path) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute_env(item, path) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ManifestError(f"{path}: environment variable {name!r} is not set")
        return os.environ[name]

    result = ENV_PATTERN.sub(replace, value)
    if "${" in result:
        raise ManifestError(f"{path}: unsupported environment expression")
    return result


def _load_manifest(
    value: str | Path, kind: str, allowed_fields: set[str]
) -> tuple[dict[str, Any], Path]:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ManifestError(f"manifest does not exist: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ManifestError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError(f"{path}: manifest root must be an object")
    data = _substitute_env(data, path)
    if data.get("schema_version") != 1 or data.get("kind") != kind:
        raise ManifestError(f"{path}: expected schema_version=1 and kind={kind}")
    if not isinstance(data.get("id"), str) or not data["id"]:
        raise ManifestError(f"{path}: id must be a non-empty string")
    unknown = sorted(set(data) - allowed_fields)
    if unknown:
        raise ManifestError(f"{path}: unknown fields: {unknown}")
    return data, path


def _resolve_path(value: str | Path, origin: Path, *, must_exist: bool = True) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = origin.parent / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise ManifestError(f"{origin}: path does not exist: {path}")
    return path


def _dataset_manifest(value: str | Path) -> tuple[dict[str, Any], Path]:
    data, path = _load_manifest(value, "dataset", DATASET_FIELDS)
    sources = data.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ManifestError(f"{path}: sources must be a non-empty array")
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ManifestError(f"{path}: sources.{index} must be an object")
        unknown = sorted(set(source) - SOURCE_FIELDS)
        if unknown:
            raise ManifestError(f"{path}: sources.{index} unknown fields: {unknown}")
        missing = sorted({"id", "type", "role", "path"} - set(source))
        if missing:
            raise ManifestError(f"{path}: sources.{index} missing fields: {missing}")
        if source["type"] not in {"csv", "parquet", "folder_label"}:
            raise ManifestError(f"{path}: sources.{index}.type is invalid")
        if source["role"] not in {"sample", "prototype", "unknown"}:
            raise ManifestError(f"{path}: sources.{index}.role is invalid")
        source["path"] = str(_resolve_path(source["path"], path))
        if source.get("image_root"):
            source["image_root"] = str(_resolve_path(source["image_root"], path))
    labels = data.setdefault("labels", {"mode": "derived", "aliases": {}})
    if not isinstance(labels, dict):
        raise ManifestError(f"{path}: labels must be an object")
    unknown = sorted(set(labels) - LABEL_FIELDS)
    if unknown:
        raise ManifestError(f"{path}: labels unknown fields: {unknown}")
    if labels.get("mode", "derived") not in {"derived", "explicit"}:
        raise ManifestError(f"{path}: labels.mode must be derived or explicit")
    labels.setdefault("mode", "derived")
    labels.setdefault("aliases", {})
    spaces = data.get("label_spaces")
    if spaces is not None:
        if not isinstance(spaces, dict):
            raise ManifestError(f"{path}: label_spaces must be an object")
        unknown = sorted(set(spaces) - LABEL_SPACE_FIELDS)
        if unknown:
            raise ManifestError(f"{path}: label_spaces unknown fields: {unknown}")
    return data, path


def _split_manifest(value: str | Path) -> tuple[dict[str, Any], Path]:
    data, path = _load_manifest(value, "split", SPLIT_FIELDS)
    if not isinstance(data.get("split"), dict):
        raise ManifestError(f"{path}: split must be an object")
    if not isinstance(data.get("views"), list) or not data["views"]:
        raise ManifestError(f"{path}: views must be a non-empty array")
    return data, path


def _materialize(
    dataset: dict[str, Any], split: dict[str, Any] | None = None
) -> MaterializedDataset:
    config = {
        key: value
        for key, value in dataset.items()
        if key in {"id", "duplicate_policy", "sources", "labels"}
    }
    config["labels"] = dict(config["labels"])
    if split is None:
        config["split"] = {
            "strategy": "stratified_hash",
            "ratios": {"train": 1.0},
            "seed": 0,
        }
        config["views"] = [
            {"id": "all", "from": "all", "usage": "analysis", "filters": {}}
        ]
        config["labels"]["reference_view"] = "all"
        seed = 0
    else:
        config["split"] = split["split"]
        config["views"] = split["views"]
        config["labels"]["reference_view"] = split.get("label_reference_view", "train")
        seed = int(split.get("seed", config["split"].get("seed", 0)))
    return materialize_dataset(config, seed)


def _label_spaces(
    dataset: dict[str, Any], materialized: MaterializedDataset
) -> dict[str, list[str]]:
    labels = sorted(set(materialized.records["label"].astype(str)))
    declared = dataset.get("label_spaces")
    if declared is not None:
        if not isinstance(declared, dict):
            raise DataError("dataset label_spaces must be an object")
        if not all(
            isinstance(declared.get(name, []), list) for name in ("closed", "open")
        ):
            raise DataError("label_spaces.closed/open must be arrays")
        closed = [str(item) for item in declared.get("closed", [])]
        opened = [str(item) for item in declared.get("open", [])]
        if set(closed) & set(opened) or set(closed) | set(opened) != set(labels):
            raise DataError(
                "label_spaces.closed/open must be disjoint and cover every label"
            )
    else:
        open_labels = set(
            materialized.records.loc[
                materialized.records["source_role"].eq("unknown"), "label"
            ].astype(str)
        )
        non_open = set(labels) - open_labels
        if not non_open:
            raise DataError("could not derive a non-empty closed label space")
        closed, opened = sorted(non_open), sorted(open_labels)
    return {"all": closed + opened, "closed": closed, "open": opened}


def _with_export_columns(
    frame: pd.DataFrame, spaces: dict[str, list[str]]
) -> pd.DataFrame:
    label_set = {
        **{label: "closed" for label in spaces["closed"]},
        **{label: "open" for label in spaces["open"]},
    }
    result = frame.copy()
    result.insert(0, "path", result["image_uri"])
    result["set"] = result["label"].map(label_set)
    if result["set"].isna().any():
        raise DataError("a materialized row has no closed/open label assignment")
    return result


def _require_new_output(value: str | Path) -> Path:
    output = Path(value).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _finalize_output(output: Path, args: argparse.Namespace) -> None:
    output = output.resolve()
    if not output.is_dir():
        raise FileNotFoundError(f"command output directory does not exist: {output}")
    artifacts = [
        {
            "path": str(path.relative_to(output)),
            "bytes": path.stat().st_size,
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "run.json"
    ]
    _write_json(
        output / "run.json",
        {
            "schema_version": 1,
            "status": "completed",
            "command": args.command,
            "arguments": {
                key: value
                for key, value in vars(args).items()
                if key != "command"
            },
            "output": str(output),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "workflow_sha256": _sha256_file(__file__),
            "artifacts": artifacts,
        },
    )


def validate_data(args: argparse.Namespace) -> dict[str, Any]:
    dataset, path = _dataset_manifest(args.dataset_manifest)
    materialized = _materialize(dataset)
    spaces = _label_spaces(dataset, materialized)
    return {
        "dataset_manifest": str(path),
        "dataset_manifest_sha256": _sha256_file(path),
        "dataset_id": dataset["id"],
        "samples": len(materialized.records),
        "labels": {key: len(value) for key, value in spaces.items()},
        "sources": materialized.summary["sources"],
        "dataset_hash": materialized.hashes["dataset"],
    }


def build_split(args: argparse.Namespace) -> Path:
    dataset, dataset_path = _dataset_manifest(args.dataset_manifest)
    split, split_path = _split_manifest(args.split_manifest)
    materialized = _materialize(dataset, split)
    spaces = _label_spaces(dataset, materialized)
    output = _require_new_output(args.output)
    output.mkdir()
    views_dir = output / "views"
    views_dir.mkdir()
    _with_export_columns(materialized.records, spaces).to_csv(
        output / "records.csv", index=False
    )
    view_paths: dict[str, str] = {}
    for view_id, frame in materialized.views.items():
        path = views_dir / f"{view_id}.csv"
        _with_export_columns(frame, spaces).to_csv(path, index=False)
        view_paths[view_id] = str(path.relative_to(output))
    _write_json(output / "labels.json", spaces)
    metadata = {
        "dataset_id": dataset["id"],
        "split_id": split["id"],
        "dataset_manifest": str(dataset_path),
        "dataset_manifest_sha256": _sha256_file(dataset_path),
        "split_manifest": str(split_path),
        "split_manifest_sha256": _sha256_file(split_path),
        "seed": int(split.get("seed", 0)),
        "views": view_paths,
        "hashes": materialized.hashes,
        "summary": materialized.summary,
    }
    _write_json(output / "metadata.json", metadata)
    _write_json(
        output / "sha256.json",
        {
            "records": _sha256_file(output / "records.csv"),
            **{
                f"view:{view_id}": _sha256_file(output / relative)
                for view_id, relative in view_paths.items()
            },
        },
    )
    return output


def _split_metadata(
    dataset_manifest: str | Path, split_dir: str | Path
) -> tuple[dict, Path]:
    dataset_path = Path(dataset_manifest).expanduser().resolve()
    split_path = Path(split_dir).expanduser().resolve()
    metadata_path = split_path / "metadata.json"
    if not metadata_path.is_file():
        raise DataError(f"split directory has no metadata.json: {split_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("dataset_manifest_sha256") != _sha256_file(dataset_path):
        raise DataError("dataset manifest differs from the one used to build the split")
    return metadata, split_path


def _view_path(metadata: dict, split_dir: Path, view: str) -> Path:
    relative = metadata.get("views", {}).get(view)
    if not relative:
        available = ", ".join(sorted(metadata.get("views", {})))
        raise DataError(f"unknown split view {view!r}; available: {available}")
    path = (split_dir / relative).resolve()
    if not path.is_file():
        raise DataError(f"split view is missing: {path}")
    return path


def _training_manifest(value: str | Path) -> tuple[dict[str, Any], Path]:
    data, path = _load_manifest(value, "training", TRAINING_FIELDS)
    for field in ("model", "objective"):
        if not isinstance(data.get(field), dict):
            raise ManifestError(f"{path}: {field} must be an object")
    for field, allowed in TRAINING_NESTED_FIELDS.items():
        value = data.get(field)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ManifestError(f"{path}: {field} must be an object")
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ManifestError(f"{path}: {field} unknown fields: {unknown}")
    if data.get("scope", "all") not in {"closed", "all"}:
        raise ManifestError(f"{path}: scope must be closed or all")
    if data["objective"].get("type", "ce") not in {"ce", "supcon", "triplet"}:
        raise ManifestError(f"{path}: objective.type must be ce, supcon, or triplet")
    return data, path


def train(args: argparse.Namespace) -> Path:
    metadata, split_dir = _split_metadata(args.dataset_manifest, args.split_dir)
    config, config_path = _training_manifest(args.experiment_manifest)
    inputs = config.get("inputs", {"train": "train", "validation": "validation"})
    train_path = _view_path(metadata, split_dir, inputs.get("train", "train"))
    valid_path = _view_path(metadata, split_dir, inputs.get("validation", "validation"))
    model = config["model"]
    objective = config["objective"]
    optimizer = config.get("optimizer", {})
    scheduler = config.get("scheduler", {})
    budget = config.get("budget", {})
    loader = config.get("loader", {})
    checkpoint = config.get("checkpoint", {})
    architecture = model.get("architecture")
    if not isinstance(architecture, str) or not architecture:
        raise ManifestError(f"{config_path}: model.architecture is required")
    init_checkpoint = (
        Path(args.init_checkpoint).expanduser().resolve()
        if args.init_checkpoint
        else None
    )
    if init_checkpoint and not init_checkpoint.is_file():
        raise FileNotFoundError(init_checkpoint)
    output = _require_new_output(args.output)
    namespace = SimpleNamespace(
        experiment_name=config["id"],
        data="manifest",
        model=architecture,
        pretrained=bool(model.get("pretrained", init_checkpoint is None)),
        checkpoint=str(init_checkpoint) if init_checkpoint else None,
        checkpoint_format=checkpoint.get("input_format", "auto"),
        train_data=str(train_path),
        valid_data=str(valid_path),
        test_data=None,
        output_dir=str(output),
        scope=config.get("scope", "all"),
        loss=objective.get("type", "ce"),
        optimizer=optimizer.get("type", "adamw"),
        scheduler=scheduler.get("type", "cosine"),
        save_format=checkpoint.get("save_format", "ckpt"),
        epochs=int(budget.get("epochs", 10)),
        batch_size=int(loader.get("batch_size", 32)),
        samples_per_class=int(loader.get("samples_per_class", 2)),
        learning_rate=float(optimizer.get("learning_rate", 5e-5)),
        weight_decay=float(optimizer.get("weight_decay", 0.05)),
        momentum=float(optimizer.get("momentum", 0.9)),
        scheduler_step_size=int(scheduler.get("step_size", 10)),
        scheduler_gamma=float(scheduler.get("gamma", 0.1)),
        label_smoothing=float(objective.get("label_smoothing", 0.0)),
        temperature=float(objective.get("temperature", 0.07)),
        triplet_margin=float(objective.get("margin", 0.2)),
        input_size=int(model.get("input_size", 224)),
        projection_dim=int(model.get("projection_dim", 128)),
        projection_hidden_dim=int(model.get("projection_hidden_dim", 512)),
        num_workers=int(loader.get("num_workers", 4)),
        seed=int(args.seed),
        device=config.get("device", "auto"),
    )
    from ..training import train as train_one

    result = Path(train_one(namespace)).resolve()
    _write_json(
        result / "command.json",
        {
            "command": "train",
            "dataset_manifest": str(Path(args.dataset_manifest).resolve()),
            "dataset_manifest_sha256": _sha256_file(args.dataset_manifest),
            "split_dir": str(split_dir),
            "split_manifest_sha256": metadata["split_manifest_sha256"],
            "experiment_manifest": str(config_path),
            "experiment_manifest_sha256": _sha256_file(config_path),
            "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
            "init_checkpoint_sha256": (
                _sha256_file(init_checkpoint) if init_checkpoint else None
            ),
            "seed": int(args.seed),
        },
    )
    return result


def _model_manifest(value: str | Path) -> tuple[dict[str, Any], Path]:
    data, path = _load_manifest(value, "model", MODEL_FIELDS)
    if not isinstance(data.get("architecture"), str) or not data["architecture"]:
        raise ManifestError(f"{path}: architecture is required")
    if data.get("adapter", "timm_pretrained") not in {
        "timm_pretrained",
        "classifier_checkpoint",
        "supcon_checkpoint",
    }:
        raise ManifestError(f"{path}: adapter is invalid")
    if data.get("checkpoint_format", "auto") not in {
        "auto",
        "ckpt",
        "safetensors",
    }:
        raise ManifestError(f"{path}: checkpoint_format is invalid")
    if data.get("checkpoint"):
        data["checkpoint"] = str(_resolve_path(data["checkpoint"], path))
    return data, path


def _extract_embeddings_to(
    args: argparse.Namespace, output: Path
) -> tuple[Path, dict[str, Any]]:
    metadata, split_dir = _split_metadata(args.dataset_manifest, args.split_dir)
    view_path = _view_path(metadata, split_dir, args.view)
    model, model_path = _model_manifest(args.model_manifest)
    checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else Path(model["checkpoint"]).resolve()
        if model.get("checkpoint")
        else None
    )
    if checkpoint and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output.mkdir(parents=True, exist_ok=False)
    embedding_path = output / "embeddings.npz"
    from ..evaluation import extract_embeddings

    extract_embeddings(
        model_name=model["id"],
        model=model["architecture"],
        adapter=model.get("adapter", "timm_pretrained"),
        pretrained=bool(model.get("pretrained", checkpoint is None)),
        embedding_source=(
            "backbone" if args.embedding_output == "encoder" else "projection"
        ),
        input_size=int(model.get("input_size", 224)),
        projection_dim=int(model.get("projection_dim", 128)),
        projection_hidden_dim=int(model.get("projection_hidden_dim", 512)),
        num_classes=model.get("num_classes"),
        manifest_path=view_path,
        output_path=embedding_path,
        checkpoint_path=checkpoint,
        checkpoint_format=model.get("checkpoint_format", "auto"),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=args.device,
    )
    embedding_metadata_path = embedding_path.with_suffix(".json")
    embedding_metadata = json.loads(embedding_metadata_path.read_text(encoding="utf-8"))
    command = {
        "command": "extract-embeddings",
        "dataset_manifest": str(Path(args.dataset_manifest).expanduser().resolve()),
        "dataset_manifest_sha256": _sha256_file(args.dataset_manifest),
        "split_dir": str(split_dir),
        "split_manifest_sha256": metadata["split_manifest_sha256"],
        "view": args.view,
        "view_sha256": _sha256_file(view_path),
        "model_manifest": str(model_path),
        "model_manifest_sha256": _sha256_file(model_path),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": _sha256_file(checkpoint) if checkpoint else None,
        "embedding_output": args.embedding_output,
    }
    embedding_metadata.update(command)
    _write_json(embedding_metadata_path, embedding_metadata)
    _write_json(output / "command.json", command)
    return embedding_path, embedding_metadata


def extract_embeddings(args: argparse.Namespace) -> Path:
    output = _require_new_output(args.output)
    _extract_embeddings_to(args, output)
    return output


def _embedding_file(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "embeddings.npz"
    if not path.is_file():
        raise FileNotFoundError(f"embedding file does not exist: {path}")
    return path


def _embedding_table(value: str | Path):
    from ..evaluation import EmbeddingTable

    return EmbeddingTable.load(_embedding_file(value))


def _embedding_metadata(value: str | Path) -> dict[str, Any]:
    path = _embedding_file(value).with_suffix(".json")
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _label_sets(table) -> tuple[list[str], list[str]]:
    by_label: dict[str, set[str]] = {}
    for label, group in zip(table.labels.astype(str), table.sets.astype(str)):
        by_label.setdefault(label, set()).add(group)
    invalid = {
        label: sorted(groups)
        for label, groups in by_label.items()
        if len(groups) != 1 or next(iter(groups)) not in {"closed", "open"}
    }
    if invalid:
        raise DataError(f"embedding labels need one closed/open set each: {invalid}")
    labels = sorted(by_label)
    return labels, [next(iter(by_label[label])) for label in labels]


def build_prototype(args: argparse.Namespace) -> Path:
    embedding_path = _embedding_file(args.embedding_dir)
    table = _embedding_table(embedding_path)
    labels, sets = _label_sets(table)
    if args.prototype_type != "ss" and args.samples_per_label is not None:
        raise ValueError("--samples-per-label is valid only for prototype-type=ss")
    vectors: list[np.ndarray] = []
    memberships: list[dict[str, Any]] = []
    for label in labels:
        indices = np.flatnonzero(table.labels == label)
        if args.prototype_type == "ss":
            count = args.samples_per_label or 5
            if len(indices) < count:
                raise DataError(
                    f"label {label!r} has {len(indices)} samples, needs {count}"
                )
            ranked = sorted(
                indices,
                key=lambda index: hashlib.sha256(
                    f"{args.support_seed}\0{label}\0{table.paths[index]}".encode()
                ).hexdigest(),
            )
            indices = np.asarray(ranked[:count])
        selected = table.features[indices].astype(np.float32)
        if args.distance_geometry == "cosine":
            selected = selected / np.maximum(
                np.linalg.norm(selected, axis=1, keepdims=True), 1e-12
            )
        prototype = selected.mean(axis=0)
        if args.distance_geometry == "cosine":
            prototype = prototype / max(float(np.linalg.norm(prototype)), 1e-12)
        vectors.append(prototype.astype(np.float32))
        memberships.extend(
            {
                "label": label,
                "path": str(table.paths[index]),
                "source_index": int(index),
            }
            for index in indices
        )
    output = _require_new_output(args.output)
    output.mkdir()
    prototype_path = output / "prototypes.npz"
    np.savez_compressed(
        prototype_path,
        vectors=np.stack(vectors),
        labels=np.asarray(labels, dtype=str),
        sets=np.asarray(sets, dtype=str),
    )
    pd.DataFrame(memberships).to_csv(output / "membership.csv", index=False)
    source_metadata = _embedding_metadata(embedding_path)
    metadata = {
        "command": "build-prototype",
        "embedding": str(embedding_path),
        "embedding_sha256": _sha256_file(embedding_path),
        "source_view_sha256": source_metadata.get("view_sha256"),
        "prototype_type": args.prototype_type,
        "distance_geometry": args.distance_geometry,
        "samples_per_label": (
            args.samples_per_label or 5 if args.prototype_type == "ss" else None
        ),
        "support_seed": int(args.support_seed),
        "labels": labels,
    }
    _write_json(output / "metadata.json", metadata)
    return output


def _load_prototypes(
    value: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    directory = Path(value).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"prototype directory does not exist: {directory}")
    path = directory / "prototypes.npz"
    if not path.is_file():
        raise FileNotFoundError(f"prototype file does not exist: {path}")
    arrays = np.load(path, allow_pickle=False)
    metadata_path = path.parent / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    return (
        np.asarray(arrays["vectors"], dtype=np.float32),
        np.asarray(arrays["labels"], dtype=str),
        np.asarray(arrays["sets"], dtype=str),
        metadata,
    )


def _predict(
    queries: np.ndarray, prototypes: np.ndarray, labels: np.ndarray, distance: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if queries.shape[1] != prototypes.shape[1]:
        raise DataError("query and prototype dimensions differ")
    if distance == "euclidean":
        matrix = np.linalg.norm(queries[:, None] - prototypes[None, :], axis=2)
        order = np.argsort(matrix, axis=1)
        values = matrix[np.arange(len(matrix)), order[:, 0]]
        margins = (
            matrix[np.arange(len(matrix)), order[:, 1]] - values
            if len(labels) > 1
            else np.full(len(matrix), np.nan)
        )
    else:
        query = queries / np.maximum(
            np.linalg.norm(queries, axis=1, keepdims=True), 1e-12
        )
        prototype = prototypes / np.maximum(
            np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12
        )
        matrix = query @ prototype.T
        order = np.argsort(-matrix, axis=1)
        values = matrix[np.arange(len(matrix)), order[:, 0]]
        margins = (
            values - matrix[np.arange(len(matrix)), order[:, 1]]
            if len(labels) > 1
            else np.full(len(matrix), np.nan)
        )
    return labels[order[:, 0]], values, margins


def evaluate_prototype(args: argparse.Namespace) -> Path:
    query_path = _embedding_file(args.query_embeddings)
    query = _embedding_table(query_path)
    prototypes, labels, sets, prototype_metadata = _load_prototypes(args.prototype_dir)
    geometry = prototype_metadata.get("distance_geometry")
    if geometry and geometry != args.distance:
        raise DataError(
            f"prototype geometry is {geometry!r}, requested distance is {args.distance!r}"
        )
    candidate_mask = (
        sets == "closed"
        if args.candidate_space == "closed27"
        else np.ones(len(labels), dtype=bool)
    )
    query_mask = (
        query.sets.astype(str) == args.query_group
        if args.query_group in {"closed", "open"}
        else np.ones(len(query.labels), dtype=bool)
    )
    candidate_vectors = prototypes[candidate_mask]
    candidate_labels = labels[candidate_mask]
    query_features = query.features[query_mask]
    truth = query.labels.astype(str)[query_mask]
    paths = query.paths.astype(str)[query_mask]
    if not len(query_features) or not len(candidate_labels):
        raise DataError("query group or candidate space is empty")
    outside = sorted(set(truth) - set(candidate_labels))
    if outside:
        raise DataError(f"query labels are outside candidate space: {outside}")
    predictions, values, margins = _predict(
        query_features, candidate_vectors, candidate_labels, args.distance
    )
    from sklearn.metrics import (
        confusion_matrix,
        precision_recall_fscore_support,
    )

    metric_labels = sorted(set(truth))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        predictions,
        labels=metric_labels,
        zero_division=0,
    )
    metrics = {
        "samples": int(len(truth)),
        "candidate_classes": int(len(candidate_labels)),
        "accuracy": float((truth == predictions).mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support) if support.sum() else 0.0),
        "distance": args.distance,
        "query_group": args.query_group,
        "candidate_space": args.candidate_space,
    }
    query_metadata = _embedding_metadata(query_path)
    signature = _sha256_json(
        {
            "query_view_sha256": query_metadata.get("view_sha256"),
            "prototype_source_view_sha256": prototype_metadata.get(
                "source_view_sha256"
            ),
            "candidate_labels": candidate_labels.tolist(),
            "prototype_type": prototype_metadata.get("prototype_type"),
            "samples_per_label": prototype_metadata.get("samples_per_label"),
            "support_seed": prototype_metadata.get("support_seed"),
            "distance_geometry": prototype_metadata.get("distance_geometry"),
            "distance": args.distance,
            "query_group": args.query_group,
            "candidate_space": args.candidate_space,
            "evaluator_version": 1,
        }
    )
    metrics["comparison_signature"] = signature
    output = _require_new_output(args.output)
    output.mkdir()
    _write_json(output / "metrics.json", metrics)
    pd.DataFrame(
        {
            "path": paths,
            "true_label": truth,
            "predicted_label": predictions,
            "correct": truth == predictions,
            "decision_value": values,
            "top2_margin": margins,
        }
    ).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(
        {
            "label": metric_labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    ).to_csv(output / "per_class_metrics.csv", index=False)
    matrix = confusion_matrix(truth, predictions, labels=candidate_labels.tolist())
    pd.DataFrame(matrix, index=candidate_labels, columns=candidate_labels).to_csv(
        output / "confusion_matrix.csv"
    )
    _write_json(
        output / "metadata.json",
        {
            "command": "evaluate-prototype",
            "query_embeddings": str(query_path),
            "query_embeddings_sha256": _sha256_file(query_path),
            "prototype_dir": str(Path(args.prototype_dir).expanduser().resolve()),
            "prototype_sha256": _sha256_file(
                Path(args.prototype_dir).expanduser().resolve() / "prototypes.npz"
            ),
            **{
                key: metrics[key]
                for key in (
                    "distance",
                    "query_group",
                    "candidate_space",
                    "comparison_signature",
                )
            },
        },
    )
    return output


def visualize_tsne(args: argparse.Namespace) -> Path:
    embedding_path = _embedding_file(args.embeddings)
    table = _embedding_table(embedding_path)
    features = table.features.astype(np.float32)
    if args.geometry == "cosine":
        features = features / np.maximum(
            np.linalg.norm(features, axis=1, keepdims=True), 1e-12
        )
    if args.max_points and len(features) > args.max_points:
        order = sorted(
            range(len(features)),
            key=lambda index: hashlib.sha256(
                f"{args.seed}\0{table.paths[index]}".encode()
            ).hexdigest(),
        )[: args.max_points]
        features = features[order]
        labels = table.labels[order]
        paths = table.paths[order]
    else:
        labels, paths = table.labels, table.paths
    if len(features) < 2:
        raise DataError("t-SNE requires at least two embeddings")
    from sklearn.manifold import TSNE

    perplexity = min(float(args.perplexity), float(len(features) - 1))
    coordinates = TSNE(
        n_components=2,
        random_state=int(args.seed),
        perplexity=perplexity,
        max_iter=int(args.iterations),
        init="pca",
        learning_rate="auto",
    ).fit_transform(features)
    output = _require_new_output(args.output)
    output.mkdir()
    frame = pd.DataFrame(
        {
            "path": paths.astype(str),
            "label": labels.astype(str),
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
        }
    )
    frame.to_csv(output / "coordinates.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 8))
    for label, group in frame.groupby("label", sort=True):
        axis.scatter(group.x, group.y, s=10, alpha=0.75, label=label)
    if frame["label"].nunique() <= 20:
        axis.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.set_title(f"t-SNE ({args.geometry})")
    figure.tight_layout()
    figure.savefig(output / "tsne.png", dpi=160)
    plt.close(figure)
    _write_json(
        output / "metadata.json",
        {
            "command": "visualize-tsne",
            "embeddings": str(embedding_path),
            "embeddings_sha256": _sha256_file(embedding_path),
            "geometry": args.geometry,
            "seed": int(args.seed),
            "samples": len(frame),
            "perplexity": perplexity,
            "iterations": int(args.iterations),
        },
    )
    return output


def compare_results(args: argparse.Namespace) -> Path:
    left_dir = Path(args.left).expanduser().resolve()
    right_dir = Path(args.right).expanduser().resolve()
    left = json.loads((left_dir / "metrics.json").read_text(encoding="utf-8"))
    right = json.loads((right_dir / "metrics.json").read_text(encoding="utf-8"))
    if left.get("comparison_signature") != right.get("comparison_signature"):
        raise DataError("results have different comparison signatures")
    deltas = {
        name: float(right[name]) - float(left[name])
        for name in ("accuracy", "macro_f1", "weighted_f1")
    }
    output = _require_new_output(args.output)
    output.mkdir()
    result = {
        "comparison_id": args.comparison_id,
        "comparison_label": args.comparison_label,
        "left": str(left_dir),
        "right": str(right_dir),
        "comparison_signature": left["comparison_signature"],
        "left_metrics": left,
        "right_metrics": right,
        "right_minus_left": deltas,
    }
    _write_json(output / "comparison.json", result)
    (output / "REPORT.md").write_text(
        "\n".join(
            [
                f"# {args.comparison_id}: {args.comparison_label}",
                "",
                "| Metric | Left | Right | Right − Left |",
                "|---|---:|---:|---:|",
                *[
                    f"| {name} | {left[name]:.6f} | {right[name]:.6f} | {delta:+.6f} |"
                    for name, delta in deltas.items()
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output


def _geometry(features: np.ndarray) -> dict[str, Any]:
    norms = np.linalg.norm(features, axis=1)
    normalized = features / np.maximum(norms[:, None], 1e-12)
    count = min(len(features), 1000)
    selected = normalized[:count]
    cosine = selected @ selected.T
    upper = cosine[np.triu_indices(count, k=1)] if count > 1 else np.asarray([])
    return {
        "samples": int(len(features)),
        "dimension": int(features.shape[1]),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "mean_dimension_std": float(features.std(axis=0).mean()),
        "near_zero_dimensions": int((features.std(axis=0) < 1e-8).sum()),
        "sampled_mean_pairwise_cosine": (float(upper.mean()) if len(upper) else None),
    }


def diagnose_geometry(args: argparse.Namespace) -> Path:
    output = _require_new_output(args.output)
    embedding_path, metadata = _extract_embeddings_to(args, output)
    table = _embedding_table(embedding_path)
    _write_json(
        output / "geometry.json",
        {
            "command": "diagnose-geometry",
            "checkpoint_sha256": metadata.get("checkpoint_sha256"),
            "view": args.view,
            **_geometry(table.features),
        },
    )
    return output


def linear_probe(args: argparse.Namespace) -> Path:
    train_path = _embedding_file(args.train_embeddings)
    test_path = _embedding_file(args.test_embeddings)
    train_table = _embedding_table(train_path)
    test_table = _embedding_table(test_path)
    desired = {"closed"} if args.candidate_space == "closed27" else {"closed", "open"}
    train_mask = np.isin(train_table.sets.astype(str), list(desired))
    test_mask = np.isin(test_table.sets.astype(str), list(desired))
    train_x, train_y = train_table.features[train_mask], train_table.labels[train_mask]
    test_x, test_y = test_table.features[test_mask], test_table.labels[test_mask]
    if set(test_y) - set(train_y):
        raise DataError("linear probe test labels are absent from training embeddings")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=int(args.seed)),
    )
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    _, _, f1, support = precision_recall_fscore_support(
        test_y, predictions, labels=sorted(set(train_y)), zero_division=0
    )
    metrics = {
        "samples": int(len(test_y)),
        "classes": int(len(set(train_y))),
        "accuracy": float((test_y == predictions).mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support) if support.sum() else 0.0),
        "candidate_space": args.candidate_space,
        "seed": int(args.seed),
    }
    output = _require_new_output(args.output)
    output.mkdir()
    _write_json(output / "metrics.json", metrics)
    pd.DataFrame(
        {
            "path": test_table.paths[test_mask],
            "true_label": test_y,
            "predicted_label": predictions,
            "correct": test_y == predictions,
        }
    ).to_csv(output / "predictions.csv", index=False)
    return output


def _add_extract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--view", required=True)
    parser.add_argument(
        "--embedding-output", choices=["encoder", "projection"], required=True
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="implant-experiment")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-data")
    validate.add_argument("--dataset-manifest", required=True)

    split = commands.add_parser("build-split")
    split.add_argument("--dataset-manifest", required=True)
    split.add_argument("--split-manifest", required=True)
    split.add_argument("--output", required=True)

    training = commands.add_parser("train")
    training.add_argument("--dataset-manifest", required=True)
    training.add_argument("--split-dir", required=True)
    training.add_argument("--experiment-manifest", required=True)
    training.add_argument("--init-checkpoint")
    training.add_argument("--seed", type=int, required=True)
    training.add_argument("--output", required=True)

    embedding = commands.add_parser("extract-embeddings")
    _add_extract_arguments(embedding)

    prototype = commands.add_parser("build-prototype")
    prototype.add_argument("--embedding-dir", required=True)
    prototype.add_argument(
        "--prototype-type", choices=["gt", "mean", "ss"], required=True
    )
    prototype.add_argument("--samples-per-label", type=int)
    prototype.add_argument("--support-seed", type=int, default=42)
    prototype.add_argument(
        "--distance-geometry",
        choices=["euclidean", "cosine"],
        required=True,
    )
    prototype.add_argument("--output", required=True)

    evaluation = commands.add_parser("evaluate-prototype")
    evaluation.add_argument("--query-embeddings", required=True)
    evaluation.add_argument("--prototype-dir", required=True)
    evaluation.add_argument(
        "--distance", choices=["euclidean", "cosine"], required=True
    )
    evaluation.add_argument(
        "--query-group", choices=["closed", "open", "all"], required=True
    )
    evaluation.add_argument(
        "--candidate-space", choices=["closed27", "full42"], required=True
    )
    evaluation.add_argument("--output", required=True)

    tsne = commands.add_parser("visualize-tsne")
    tsne.add_argument("--embeddings", required=True)
    tsne.add_argument("--geometry", choices=["euclidean", "cosine"], required=True)
    tsne.add_argument("--seed", type=int, required=True)
    tsne.add_argument("--max-points", type=int, default=5000)
    tsne.add_argument("--perplexity", type=float, default=30.0)
    tsne.add_argument("--iterations", type=int, default=1000)
    tsne.add_argument("--output", required=True)

    compare = commands.add_parser("compare-results")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--comparison-id", required=True)
    compare.add_argument("--comparison-label", required=True)
    compare.add_argument("--output", required=True)

    diagnose = commands.add_parser("diagnose-geometry")
    _add_extract_arguments(diagnose)

    probe = commands.add_parser("linear-probe")
    probe.add_argument("--train-embeddings", required=True)
    probe.add_argument("--test-embeddings", required=True)
    probe.add_argument(
        "--candidate-space", choices=["closed27", "full42"], required=True
    )
    probe.add_argument("--seed", type=int, required=True)
    probe.add_argument("--output", required=True)
    return parser


HANDLERS = {
    "validate-data": validate_data,
    "build-split": build_split,
    "train": train,
    "extract-embeddings": extract_embeddings,
    "build-prototype": build_prototype,
    "evaluate-prototype": evaluate_prototype,
    "visualize-tsne": visualize_tsne,
    "compare-results": compare_results,
    "diagnose-geometry": diagnose_geometry,
    "linear-probe": linear_probe,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = HANDLERS[args.command](args)
        if isinstance(result, dict):
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(result)
            _finalize_output(Path(result), args)
        return 0
    except KeyboardInterrupt:
        return 130
    except WorkflowError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 10


if __name__ == "__main__":
    raise SystemExit(main())
