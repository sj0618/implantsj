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
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from ..protocol import (
    MODELS as PROTOCOL_MODELS,
    SLICES as PROTOCOL_SLICES,
    ProtocolError,
    artifact_inventory_hash,
    audit_study as validate_study_audit,
    freeze_validation as build_validation_freeze,
    protocol_lineage,
    load_protocol,
    paired_uncertainty,
    sha256_file as protocol_sha256_file,
    validate_freeze_commitment,
    validate_lineage_compatibility,
    validate_upstream_run,
)
from .materialize import (
    MaterializedDataset,
    canonical_isolation_audit,
    canonical_assignment_sha256,
    canonical_materialized_metadata,
    canonical_table_identity,
    materialize_dataset,
    validate_canonical_source,
    validate_distinct_assignments,
    validate_exact_label_coverage,
    validate_same_universe,
)
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


def _protocol_from_args(args: argparse.Namespace):
    value = getattr(args, "protocol_manifest", None)
    return load_protocol(value) if value else None


def _protocol_base_checkpoint(protocol, protocol_manifest: str | Path) -> Path:
    value = Path(protocol.raw["base_checkpoint"]["path"]).expanduser()
    if not value.is_absolute():
        value = Path(protocol_manifest).expanduser().resolve().parent / value
    return value.resolve()


def _read_freeze(path: str | Path) -> dict[str, Any]:
    directory = Path(path).expanduser().resolve()
    freeze_path = (
        directory / "validation_freeze.json" if directory.is_dir() else directory
    )
    try:
        return json.loads(freeze_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid validation freeze: {freeze_path}") from exc


def _validated_freeze(
    value: str | Path, protocol, seed: int
) -> tuple[dict[str, Any], Path]:
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "validation_freeze.json"
    freeze = _read_freeze(path)
    validate_upstream_run(
        path.parent,
        command="freeze-validation",
        role="validation-freeze",
        protocol=protocol,
        seed=seed,
    )
    validate_freeze_commitment(freeze, protocol, seed)
    frozen_results = path.parent / "frozen_validation_results.csv"
    if freeze.get("frozen_results_sha256") != _sha256_file(
        frozen_results
    ) or not freeze.get("created_at"):
        raise ProtocolError("validation freeze receipt is incomplete or altered")
    return freeze, path


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
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "command": args.command,
        "arguments": {
            key: value
            for key, value in vars(args).items()
            if key != "command" and not key.startswith("_")
        },
        "output": str(output),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "workflow_sha256": _sha256_file(__file__),
        "artifact_sha256": artifact_inventory_hash(output),
        "artifacts": artifacts,
    }
    protocol = _protocol_from_args(args)
    if protocol is not None:
        payload.update(
            {
                "protocol_fingerprint": protocol.fingerprint,
                "role": getattr(args, "_output_role", args.command),
            }
        )
        if getattr(args, "seed", None) is not None:
            payload["seed"] = int(args.seed)
    _write_json(output / "run.json", payload)


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


def _validate_protocol_sources(
    protocol, protocol_manifest: str | Path
) -> dict[str, str]:
    protocol_dir = Path(protocol_manifest).expanduser().resolve().parent
    assignments: dict[str, pd.DataFrame] = {}
    samples: dict[str, pd.DataFrame] = {}
    gt_identities: dict[str, str] = {}
    gt_file_hashes: dict[str, str] = {}

    for seed, binding in sorted(protocol.seed_bindings.items()):
        expected_dataset = (protocol_dir / binding.dataset_manifest).resolve()
        expected_split = (protocol_dir / binding.split_manifest).resolve()
        dataset, dataset_path = _dataset_manifest(expected_dataset)
        split, split_path = _split_manifest(expected_split)
        if dataset_path != expected_dataset or split_path != expected_split:
            raise ProtocolError(f"seed {seed} manifest binding changed")
        if (
            binding.dataset_sha256
            and _sha256_file(dataset_path) != binding.dataset_sha256
        ):
            raise ProtocolError(f"seed {seed} dataset manifest hash mismatch")
        if binding.split_sha256 and _sha256_file(split_path) != binding.split_sha256:
            raise ProtocolError(f"seed {seed} split manifest hash mismatch")
        if dataset["id"] != f"implant-metric42-seed{seed}-v1":
            raise ProtocolError(f"seed {seed} dataset id mismatch")
        if split["id"] != f"metric42-seed{seed}" or int(split["seed"]) != seed:
            raise ProtocolError(f"seed {seed} split id mismatch")

        sample_sources = [
            source for source in dataset["sources"] if source["role"] == "sample"
        ]
        gt_sources = [
            source for source in dataset["sources"] if source["role"] == "prototype"
        ]
        if len(sample_sources) != 1 or len(gt_sources) != 1:
            raise ProtocolError(
                f"seed {seed} requires exactly one sample and one GT source"
            )
        sample_source, gt_source = sample_sources[0], gt_sources[0]
        if _sha256_file(sample_source["path"]) != binding.assignment_csv_sha256:
            raise ProtocolError(f"seed {seed} assignment CSV hash mismatch")

        sample = validate_canonical_source(sample_source)
        gt = validate_canonical_source(gt_source)
        normalized = canonical_assignment_sha256(
            sample, context=f"seed {seed} assignment"
        )
        if normalized != binding.normalized_assignment_sha256:
            raise ProtocolError(f"seed {seed} normalized assignment hash mismatch")
        assignments[str(seed)] = sample
        samples[str(seed)] = sample
        gt_identities[str(seed)] = canonical_table_identity(
            gt,
            context=f"seed {seed} GT",
            fields=("sample_id", "image_uri", "label", "group_id"),
        )
        gt_file_hashes[str(seed)] = _sha256_file(gt_source["path"])

    validate_distinct_assignments(assignments)
    reference_seed = str(min(protocol.seed_bindings))
    for seed, sample in samples.items():
        if seed != reference_seed:
            validate_same_universe(
                samples[reference_seed],
                sample,
                reference_name=f"seed {reference_seed} samples",
                candidate_name=f"seed {seed} samples",
            )
    if len(set(gt_identities.values())) != 1 or len(set(gt_file_hashes.values())) != 1:
        raise ProtocolError("canonical GT source differs across seed bindings")
    return {
        "sample_universe_sha256": canonical_table_identity(
            samples[reference_seed], context=f"seed {reference_seed} samples"
        ),
        "gt_identity_sha256": gt_identities[reference_seed],
        "gt_file_sha256": gt_file_hashes[reference_seed],
    }


def build_split(args: argparse.Namespace) -> Path:
    dataset, dataset_path = _dataset_manifest(args.dataset_manifest)
    split, split_path = _split_manifest(args.split_manifest)
    materialized = _materialize(dataset, split)
    spaces = _label_spaces(dataset, materialized)
    protocol = _protocol_from_args(args)
    canonical_metadata = None
    isolation_audit = None
    if protocol is not None:
        source_audit = _validate_protocol_sources(protocol, args.protocol_manifest)
        seed = int(split.get("seed", -1))
        binding = protocol.seed_binding(seed)
        protocol_dir = Path(args.protocol_manifest).expanduser().resolve().parent
        expected_dataset = (protocol_dir / binding.dataset_manifest).resolve()
        expected_split = (protocol_dir / binding.split_manifest).resolve()
        if dataset_path != expected_dataset or split_path != expected_split:
            raise ProtocolError(
                "dataset/split manifest does not match protocol seed binding"
            )
        sample_sources = [
            source for source in dataset["sources"] if source["role"] == "sample"
        ]
        if len(sample_sources) != 1:
            raise ProtocolError(
                "canonical dataset requires one sample assignment source"
            )
        if _sha256_file(sample_sources[0]["path"]) != binding.assignment_csv_sha256:
            raise ProtocolError("assignment CSV hash does not match protocol")
        labels = protocol.raw["labels"]["ordered"]
        validate_exact_label_coverage(
            materialized.views,
            closed_labels=labels[:27],
            open_labels=labels[27:],
            minimums={"train": 1, "validation": 1, "test": 1, "gt_all": 1},
        )
        isolation_audit = canonical_isolation_audit(
            materialized.views, gt_view="gt_all"
        )
        canonical_metadata = canonical_materialized_metadata(materialized)
        canonical_metadata.update(source_audit)
        if (
            canonical_metadata["assignment_sha256"]
            != binding.normalized_assignment_sha256
        ):
            raise ProtocolError("normalized assignment hash does not match protocol")
        args.seed = seed
        args._output_role = "split"
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
    if isolation_audit is not None:
        isolation_path = output / "isolation_audit.json"
        _write_json(isolation_path, isolation_audit)
        canonical_metadata["isolation_audit_sha256"] = _sha256_file(isolation_path)
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
    if canonical_metadata is not None:
        metadata["canonical"] = canonical_metadata
        metadata["protocol_fingerprint"] = protocol.fingerprint
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


def _validate_protocol_split(
    split_dir: Path, metadata: dict[str, Any], protocol, seed: int
) -> None:
    run_path = split_dir / "run.json"
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("canonical split has no valid run.json") from exc
    if (
        run.get("status") != "completed"
        or run.get("command") != "build-split"
        or run.get("role") != "split"
        or run.get("protocol_fingerprint") != protocol.fingerprint
        or int(run.get("seed", -1)) != seed
    ):
        raise ProtocolError("canonical split completion/lineage mismatch")
    binding = protocol.seed_binding(seed)
    canonical = metadata.get("canonical", {})
    if canonical.get("assignment_sha256") != binding.normalized_assignment_sha256:
        raise ProtocolError("canonical split assignment mismatch")
    hashes = json.loads((split_dir / "sha256.json").read_text(encoding="utf-8"))
    for name, expected in hashes.items():
        relative = (
            "records.csv"
            if name == "records"
            else metadata["views"][name.split(":", 1)[1]]
        )
        if _sha256_file(split_dir / relative) != expected:
            raise ProtocolError(f"canonical split artifact changed: {name}")


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
    protocol = _protocol_from_args(args)
    canonical_training = None
    if protocol is not None:
        if not args.model_role or args.model_role == "Base":
            raise ProtocolError("canonical training requires a trained model role")
        role = protocol.raw["model_roles"].get(args.model_role)
        if not isinstance(role, dict) or not role.get("training_applied"):
            raise ProtocolError("model role is not a canonical training treatment")
        expected_objective = role.get("objective")
        actual_objective = objective.get("type", "ce")
        if expected_objective != actual_objective:
            raise ProtocolError("training objective does not match model role")
        if role.get("scope") != config.get("scope", "all"):
            raise ProtocolError("training scope does not match model role")
        if architecture != protocol.raw["architecture"]:
            raise ProtocolError("training architecture does not match protocol")
        if init_checkpoint is None:
            raise ProtocolError(
                "canonical training requires the pinned Base checkpoint"
            )
        if init_checkpoint != _protocol_base_checkpoint(
            protocol, args.protocol_manifest
        ):
            raise ProtocolError("Base checkpoint path does not match protocol")
        base_sha = _sha256_file(init_checkpoint)
        if base_sha != protocol.raw["base_checkpoint"]["sha256"]:
            raise ProtocolError("Base checkpoint hash does not match protocol")
        _validate_protocol_split(split_dir, metadata, protocol, int(args.seed))
        binding = protocol.seed_binding(int(args.seed))
        ordered_labels = sorted(protocol.raw["labels"]["ordered"])
        if role["scope"] == "closed":
            ordered_labels = sorted(protocol.raw["labels"]["ordered"][:27])
        canonical_training = {
            "model_role": args.model_role,
            "protocol_fingerprint": protocol.fingerprint,
            "base_checkpoint_sha256": base_sha,
            "assignment_sha256": binding.normalized_assignment_sha256,
            "objective": (
                "pure_supcon" if actual_objective == "supcon" else "cross_entropy"
            ),
            "ordered_labels": ordered_labels,
            "architecture": architecture,
            "epochs": int(budget.get("epochs", 10)),
            "lineage": protocol_lineage(
                protocol,
                model=args.model_role,
                checkpoint_sha256=base_sha,
                treatment=role,
            ),
        }
        args._output_role = "training"
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
        canonical_metadata=canonical_training,
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
    protocol = _protocol_from_args(args)
    lineage = None
    canonical_checkpoint = None
    if protocol is not None:
        if args.seed is None:
            raise ProtocolError("canonical embedding requires --seed")
        if not args.model_role or args.model_role not in PROTOCOL_MODELS:
            raise ProtocolError("canonical embedding requires a model role")
        if args.embedding_output != "encoder":
            raise ProtocolError("canonical embeddings must use encoder output")
        _validate_protocol_split(split_dir, metadata, protocol, int(args.seed))
        if checkpoint is None:
            raise ProtocolError("canonical embedding requires a checkpoint")
        checkpoint_sha = _sha256_file(checkpoint)
        if args.model_role == "Base":
            if checkpoint != _protocol_base_checkpoint(
                protocol, args.protocol_manifest
            ):
                raise ProtocolError(
                    "Base embedding checkpoint path does not match protocol"
                )
            if checkpoint_sha != protocol.raw["base_checkpoint"]["sha256"]:
                raise ProtocolError("Base embedding checkpoint does not match protocol")
        else:
            if checkpoint.name != "last.pt":
                raise ProtocolError("canonical trained embedding requires last.pt")
            validate_upstream_run(
                checkpoint.parent.parent,
                command="train",
                role="training",
                protocol=protocol,
                seed=int(args.seed),
            )
        lineage = protocol_lineage(
            protocol,
            model=args.model_role,
            checkpoint_sha256=checkpoint_sha,
            treatment=protocol.raw["model_roles"][args.model_role],
        )
        if args.model_role != "Base":
            treatment = protocol.raw["model_roles"][args.model_role]
            canonical_checkpoint = {
                "model_role": args.model_role,
                "architecture": protocol.raw["architecture"],
                "base_checkpoint_sha256": protocol.raw["base_checkpoint"]["sha256"],
                "scope": treatment["scope"],
                "objective": (
                    "pure_supcon"
                    if treatment["objective"] == "supcon"
                    else "cross_entropy"
                ),
                "seed": int(args.seed),
                "epochs": int(protocol.raw["budget"]["epochs"]),
                "protocol_fingerprint": protocol.fingerprint,
                "assignment_sha256": protocol.seed_binding(
                    int(args.seed)
                ).normalized_assignment_sha256,
            }
        args._output_role = "embedding"
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
        canonical_checkpoint=canonical_checkpoint,
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
    if protocol is not None:
        command.update(
            {
                "protocol_fingerprint": protocol.fingerprint,
                "seed": int(args.seed),
                "model_role": args.model_role,
                "view_role": metadata.get("canonical", {})
                .get("views", {})
                .get(args.view, {})
                .get("usage"),
                "lineage": lineage,
            }
        )
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
    source_metadata = _embedding_metadata(embedding_path)
    protocol = _protocol_from_args(args)
    if protocol is not None:
        if args.seed is None:
            raise ProtocolError("canonical prototype requires --seed")
        validate_upstream_run(
            embedding_path.parent,
            command="extract-embeddings",
            role="embedding",
            protocol=protocol,
            seed=int(args.seed),
        )
        if source_metadata.get("protocol_fingerprint") != protocol.fingerprint:
            raise ProtocolError("prototype embedding protocol mismatch")
        expected_role = "prototype" if args.prototype_type == "gt" else "train"
        if source_metadata.get("view_role") != expected_role:
            raise ProtocolError(
                f"{args.prototype_type} prototype requires {expected_role} embedding"
            )
        ordered = list(protocol.raw["labels"]["ordered"])
        if set(labels) != set(ordered):
            raise ProtocolError("canonical prototype must cover ordered full42 labels")
        set_by_label = dict(zip(labels, sets))
        labels = ordered
        sets = [set_by_label[label] for label in labels]
        if args.prototype_type == "ss" and int(args.support_seed) != int(args.seed):
            raise ProtocolError("canonical SS support seed must equal paired seed")
        args._output_role = "prototype"
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
    if protocol is not None:
        metadata.update(
            {
                "protocol_fingerprint": protocol.fingerprint,
                "seed": int(args.seed),
                "model_role": source_metadata.get("model_role"),
                "lineage": source_metadata.get("lineage"),
            }
        )
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


def _metric_summary(truth: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import precision_recall_fscore_support

    labels = sorted(set(truth.astype(str)))
    _, _, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=labels, zero_division=0
    )
    return {
        "samples": int(len(truth)),
        "accuracy": float((truth == predictions).mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support) if support.sum() else 0.0),
    }


def _evaluate_canonical(args: argparse.Namespace, protocol) -> Path:
    if (
        args.seed is None
        or args.prototype_type not in {"gt", "mean", "ss"}
        or args.query_group != "all"
        or args.candidate_space != "full42"
        or args.model_role not in PROTOCOL_MODELS
        or args.query_view not in {"validation", "test"}
    ):
        raise ProtocolError(
            "canonical evaluation requires all/full42, model role, and validation/test view"
        )
    output = _require_new_output(args.output)
    freeze = None
    freeze_path = None
    if args.query_view == "test":
        if not args.validation_freeze:
            raise ProtocolError("canonical test evaluation requires validation freeze")
        freeze, freeze_path = _validated_freeze(
            args.validation_freeze, protocol, int(args.seed)
        )
        matches = [
            row
            for row in freeze["test_plan"]
            if int(row["seed"]) == int(args.seed)
            and row["model"] == args.model_role
            and row["query"] == "test"
            and row["prototype"] == args.prototype_type
            and row["distance"] == args.distance
            and row["candidate_space"] == "full42"
        ]
        if len(matches) != 1 or Path(matches[0]["output_path"]).resolve() != output:
            raise ProtocolError("test output is not the exact frozen planned cell")
        for row in freeze["validation"]:
            artifact = Path(row["output_path"])
            actual = (
                protocol_sha256_file(artifact)
                if artifact.is_file()
                else artifact_inventory_hash(artifact)
            )
            if actual != row["artifact_sha256"]:
                raise ProtocolError("frozen validation artifact changed")
    elif args.validation_freeze:
        raise ProtocolError("validation evaluation must not consume a freeze")

    query_path = _embedding_file(args.query_embeddings)
    query_metadata = _embedding_metadata(query_path)
    prototypes, labels, sets, prototype_metadata = _load_prototypes(args.prototype_dir)
    validate_upstream_run(
        query_path.parent,
        command="extract-embeddings",
        role="embedding",
        protocol=protocol,
        seed=int(args.seed),
    )
    validate_upstream_run(
        Path(args.prototype_dir).expanduser().resolve(),
        command="build-prototype",
        role="prototype",
        protocol=protocol,
        seed=int(args.seed),
    )
    if (
        query_metadata.get("protocol_fingerprint") != protocol.fingerprint
        or prototype_metadata.get("protocol_fingerprint") != protocol.fingerprint
        or query_metadata.get("model_role") != args.model_role
        or prototype_metadata.get("model_role") != args.model_role
    ):
        raise ProtocolError("canonical evaluation input lineage mismatch")
    if query_metadata.get("view_role") != args.query_view:
        raise ProtocolError("query embedding role does not match evaluation view")
    if prototype_metadata.get("prototype_type") != args.prototype_type:
        raise ProtocolError("prototype type does not match evaluation key")
    if prototype_metadata.get("distance_geometry") != args.distance:
        raise ProtocolError("prototype geometry does not match evaluation distance")
    if labels.tolist() != protocol.raw["labels"]["ordered"]:
        raise ProtocolError("canonical evaluation requires ordered full42 prototypes")

    query = _embedding_table(query_path)
    truth = query.labels.astype(str)
    true_sets = query.sets.astype(str)
    if set(truth) - set(labels):
        raise ProtocolError("query labels are outside canonical full42 labels")
    predictions, values, margins = _predict(
        query.features, prototypes, labels, args.distance
    )
    rows = []
    for slice_name in PROTOCOL_SLICES:
        mask = (
            np.ones(len(truth), dtype=bool)
            if slice_name == "all"
            else true_sets == slice_name
        )
        if not mask.any():
            raise ProtocolError(f"canonical evaluation slice is empty: {slice_name}")
        rows.append(
            {
                "seed": int(args.seed),
                "model": args.model_role,
                "query": args.query_view,
                "prototype": args.prototype_type,
                "distance": args.distance,
                "candidate_space": "full42",
                "slice": slice_name,
                "candidate_classes": 42,
                **_metric_summary(truth[mask], predictions[mask]),
            }
        )

    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    ordered = protocol.raw["labels"]["ordered"]
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=ordered, zero_division=0
    )
    output.mkdir()
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    _write_json(output / "metrics.json", rows[0])
    pd.DataFrame(
        {
            "path": query.paths.astype(str),
            "true_label": truth,
            "true_set": true_sets,
            "predicted_label": predictions,
            "correct": truth == predictions,
            "decision_value": values,
            "top2_margin": margins,
        }
    ).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(
        {
            "label": ordered,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    ).to_csv(output / "per_class_metrics.csv", index=False)
    matrix = confusion_matrix(truth, predictions, labels=ordered)
    pd.DataFrame(matrix, index=ordered, columns=ordered).to_csv(
        output / "confusion_matrix.csv"
    )
    metadata = {
        "command": "evaluate-prototype",
        "protocol_fingerprint": protocol.fingerprint,
        "seed": int(args.seed),
        "model_role": args.model_role,
        "query": args.query_view,
        "prototype_type": args.prototype_type,
        "distance": args.distance,
        "candidate_space": "full42",
        "query_embeddings": str(query_path),
        "query_embeddings_sha256": _sha256_file(query_path),
        "prototype_dir": str(Path(args.prototype_dir).expanduser().resolve()),
        "prototype_sha256": _sha256_file(
            Path(args.prototype_dir).expanduser().resolve() / "prototypes.npz"
        ),
        "lineage": query_metadata.get("lineage"),
        "comparison_signature": _sha256_json(
            {
                "protocol": protocol.fingerprint,
                "seed": int(args.seed),
                "query": args.query_view,
                "prototype": args.prototype_type,
                "distance": args.distance,
                "candidate_space": "full42",
            }
        ),
        "freeze_commitment_sha256": (
            freeze["freeze_commitment_sha256"] if freeze else None
        ),
        "validation_freeze": str(freeze_path) if freeze_path else None,
        "validation_freeze_sha256": (
            _sha256_file(freeze_path) if freeze_path else None
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output / "metadata.json", metadata)
    args._output_role = "evaluation"
    return output


def evaluate_prototype(args: argparse.Namespace) -> Path:
    protocol = _protocol_from_args(args)
    if protocol is not None:
        return _evaluate_canonical(args, protocol)
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


def _compare_canonical(args: argparse.Namespace, protocol) -> Path:
    if (
        args.seed is None
        or args.left_model not in PROTOCOL_MODELS
        or args.right_model not in PROTOCOL_MODELS
        or args.query_view not in {"validation", "test"}
        or args.prototype_type not in {"gt", "mean", "ss"}
        or args.distance not in {"euclidean", "cosine"}
    ):
        raise ProtocolError("canonical comparison requires every scalar study key")
    left_dir = Path(args.left).expanduser().resolve()
    right_dir = Path(args.right).expanduser().resolve()
    validate_upstream_run(
        left_dir,
        command="evaluate-prototype",
        role="evaluation",
        protocol=protocol,
        seed=int(args.seed),
    )
    validate_upstream_run(
        right_dir,
        command="evaluate-prototype",
        role="evaluation",
        protocol=protocol,
        seed=int(args.seed),
    )
    left_meta = json.loads((left_dir / "metadata.json").read_text(encoding="utf-8"))
    right_meta = json.loads((right_dir / "metadata.json").read_text(encoding="utf-8"))
    for metadata in (left_meta, right_meta):
        if (
            metadata.get("protocol_fingerprint") != protocol.fingerprint
            or int(metadata.get("seed", -1)) != int(args.seed)
            or metadata.get("query") != args.query_view
            or metadata.get("prototype_type") != args.prototype_type
            or metadata.get("distance") != args.distance
            or metadata.get("candidate_space") != "full42"
        ):
            raise ProtocolError("comparison evaluation cell mismatch")
    validate_lineage_compatibility(
        left_meta.get("lineage", {}),
        right_meta.get("lineage", {}),
        protocol,
        experiment_id=args.comparison_id,
        distance=args.distance,
    )
    if (
        left_meta.get("model_role") != args.left_model
        or right_meta.get("model_role") != args.right_model
    ):
        raise ProtocolError("comparison model order mismatch")
    left = pd.read_csv(left_dir / "metrics.csv").sort_values("slice")
    right = pd.read_csv(right_dir / "metrics.csv").sort_values("slice")
    if list(left["slice"]) != list(right["slice"]) or set(left["slice"]) != set(
        PROTOCOL_SLICES
    ):
        raise ProtocolError("comparison requires matching all/closed/open slices")
    rows = []
    for left_row, right_row in zip(left.to_dict("records"), right.to_dict("records")):
        row = {
            "seed": int(args.seed),
            "experiment_id": args.comparison_id,
            "left_model": args.left_model,
            "right_model": args.right_model,
            "query": args.query_view,
            "prototype": args.prototype_type,
            "distance": args.distance,
            "candidate_space": "full42",
            "slice": left_row["slice"],
        }
        for metric in ("accuracy", "macro_f1", "weighted_f1"):
            row[f"left_{metric}"] = float(left_row[metric])
            row[f"right_{metric}"] = float(right_row[metric])
            row[f"{metric}_delta"] = float(right_row[metric]) - float(left_row[metric])
        rows.append(row)
    left_freeze = left_meta.get("freeze_commitment_sha256")
    right_freeze = right_meta.get("freeze_commitment_sha256")
    if left_freeze != right_freeze:
        raise ProtocolError("comparison freeze commitment mismatch")
    output = _require_new_output(args.output)
    output.mkdir()
    pd.DataFrame(rows).to_csv(output / "deltas.csv", index=False)
    result = {
        "comparison_id": args.comparison_id,
        "comparison_label": args.comparison_label,
        "left": str(left_dir),
        "right": str(right_dir),
        "left_model": args.left_model,
        "right_model": args.right_model,
        "seed": int(args.seed),
        "query": args.query_view,
        "prototype": args.prototype_type,
        "distance": args.distance,
        "candidate_space": "full42",
        "protocol_fingerprint": protocol.fingerprint,
        "left_evaluation_sha256": artifact_inventory_hash(left_dir),
        "right_evaluation_sha256": artifact_inventory_hash(right_dir),
        "freeze_commitment_sha256": left_freeze,
        "rows": len(rows),
    }
    _write_json(output / "comparison.json", result)
    args._output_role = "comparison"
    return output


def compare_results(args: argparse.Namespace) -> Path:
    protocol = _protocol_from_args(args)
    if protocol is not None:
        return _compare_canonical(args, protocol)
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


def freeze_validation_command(args: argparse.Namespace) -> Path:
    protocol = _protocol_from_args(args)
    if protocol is None:
        raise ProtocolError("freeze-validation requires --protocol-manifest")
    output = _require_new_output(args.output)
    freeze = build_validation_freeze(
        args.pretest_ledger, protocol, int(args.seed), verify_artifacts=True
    )
    frames = []
    for row in freeze["validation"]:
        frame = pd.read_csv(Path(row["output_path"]) / "metrics.csv")
        if len(frame) != 3 or set(frame["slice"]) != set(PROTOCOL_SLICES):
            raise ProtocolError("validation artifact must contain three metric slices")
        frames.append(frame)
    frozen_results = pd.concat(frames, ignore_index=True)
    if len(frozen_results) != 90:
        raise ProtocolError("validation freeze must contain exactly 90 metric rows")
    output.mkdir()
    frozen_path = output / "frozen_validation_results.csv"
    frozen_results.to_csv(frozen_path, index=False)
    freeze["frozen_results_sha256"] = _sha256_file(frozen_path)
    freeze["created_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(output / "validation_freeze.json", freeze)
    args._output_role = "validation-freeze"
    return output


_METRICS = ("accuracy", "macro_f1", "weighted_f1")


def _validated_evaluation_frame(
    path: Path, ledger_row: Mapping[str, Any]
) -> pd.DataFrame:
    frame = pd.read_csv(path / "metrics.csv")
    required = {
        "seed",
        "model",
        "query",
        "prototype",
        "distance",
        "candidate_space",
        "slice",
        "candidate_classes",
        "samples",
        *_METRICS,
    }
    if set(frame.columns) != required or frame["slice"].tolist() != list(
        PROTOCOL_SLICES
    ):
        raise ProtocolError("evaluation metrics schema or slices are invalid")
    expected = {
        "seed": int(ledger_row["seed"]),
        "model": ledger_row["model"],
        "query": ledger_row["query"],
        "prototype": ledger_row["prototype"],
        "distance": ledger_row["distance"],
        "candidate_space": "full42",
    }
    for field, value in expected.items():
        if not (frame[field] == value).all():
            raise ProtocolError(f"evaluation metrics key mismatch: {field}")
    candidate_classes = pd.to_numeric(frame["candidate_classes"], errors="coerce")
    sample_counts = pd.to_numeric(frame["samples"], errors="coerce")
    metric_values = frame.loc[:, _METRICS].apply(pd.to_numeric, errors="coerce")
    if (
        not np.isfinite(candidate_classes).all()
        or not (candidate_classes == 42).all()
        or not np.isfinite(sample_counts).all()
        or not (sample_counts > 0).all()
        or not (sample_counts == np.floor(sample_counts)).all()
        or not np.isfinite(metric_values.to_numpy()).all()
        or not ((metric_values >= 0.0) & (metric_values <= 1.0)).all().all()
    ):
        raise ProtocolError("evaluation metrics class/sample/metric values are invalid")
    samples_by_slice = dict(zip(frame["slice"], sample_counts, strict=True))
    if samples_by_slice["all"] != (
        samples_by_slice["closed"] + samples_by_slice["open"]
    ):
        raise ProtocolError(
            "evaluation metrics all samples must equal closed plus open samples"
        )
    return frame


def _validated_comparison_frame(
    path: Path,
    ledger_row: Mapping[str, Any],
    evaluation_frames: Mapping[tuple[str, str, str, str], pd.DataFrame],
) -> pd.DataFrame:
    frame = pd.read_csv(path / "deltas.csv")
    key_fields = {
        "seed",
        "experiment_id",
        "left_model",
        "right_model",
        "query",
        "prototype",
        "distance",
        "candidate_space",
        "slice",
    }
    metric_fields = {
        f"{side}_{metric}" for side in ("left", "right") for metric in _METRICS
    } | {f"{metric}_delta" for metric in _METRICS}
    if set(frame.columns) != key_fields | metric_fields or frame[
        "slice"
    ].tolist() != list(PROTOCOL_SLICES):
        raise ProtocolError("comparison delta schema or slices are invalid")
    expected = {
        "seed": int(ledger_row["seed"]),
        "experiment_id": ledger_row["experiment_id"],
        "left_model": ledger_row["left_model"],
        "right_model": ledger_row["right_model"],
        "query": ledger_row["query"],
        "prototype": ledger_row["prototype"],
        "distance": ledger_row["distance"],
        "candidate_space": "full42",
    }
    for field, value in expected.items():
        if not (frame[field] == value).all():
            raise ProtocolError(f"comparison delta key mismatch: {field}")
    left = evaluation_frames[
        (
            ledger_row["left_model"],
            ledger_row["query"],
            ledger_row["prototype"],
            ledger_row["distance"],
        )
    ].set_index("slice")
    right = evaluation_frames[
        (
            ledger_row["right_model"],
            ledger_row["query"],
            ledger_row["prototype"],
            ledger_row["distance"],
        )
    ].set_index("slice")
    for row in frame.to_dict("records"):
        slice_name = row["slice"]
        for metric in _METRICS:
            left_value = float(left.loc[slice_name, metric])
            right_value = float(right.loc[slice_name, metric])
            if (
                not np.isclose(float(row[f"left_{metric}"]), left_value)
                or not np.isclose(float(row[f"right_{metric}"]), right_value)
                or not np.isclose(
                    float(row[f"{metric}_delta"]), right_value - left_value
                )
            ):
                raise ProtocolError(
                    f"comparison delta value mismatch: {slice_name}/{metric}"
                )
    return frame


def audit_study_command(args: argparse.Namespace) -> Path:
    protocol = _protocol_from_args(args)
    if protocol is None:
        raise ProtocolError("audit-study requires --protocol-manifest")
    freeze, _ = _validated_freeze(args.validation_freeze, protocol, int(args.seed))
    summary = validate_study_audit(
        args.evaluation_ledger,
        args.comparison_ledger,
        protocol,
        int(args.seed),
        freeze,
    )
    evaluations = pd.read_csv(args.evaluation_ledger)
    comparisons = pd.read_csv(args.comparison_ledger)
    result_frames = []
    evaluation_frames = {}
    for row in evaluations.to_dict("records"):
        frame = _validated_evaluation_frame(Path(row["evaluation_path"]), row)
        result_frames.append(frame)
        evaluation_frames[
            (
                row["model"],
                row["query"],
                row["prototype"],
                row["distance"],
            )
        ] = frame
    delta_frames = []
    for row in comparisons.to_dict("records"):
        delta_frames.append(
            _validated_comparison_frame(
                Path(row["comparison_path"]), row, evaluation_frames
            )
        )
    results = pd.concat(result_frames, ignore_index=True)
    deltas = pd.concat(delta_frames, ignore_index=True)
    if len(results) != 180 or len(deltas) != 216:
        raise ProtocolError("final audit row totals are not 180/216")
    output = _require_new_output(args.output)
    output.mkdir()
    results.to_csv(output / "canonical_results.csv", index=False)
    deltas.to_csv(output / "comparison_results.csv", index=False)
    _write_json(output / "coverage_audit.json", summary)
    args._output_role = "final-audit"
    return output


def summarize_uncertainty_command(args: argparse.Namespace) -> Path:
    protocol = _protocol_from_args(args)
    if protocol is None:
        raise ProtocolError("summarize-uncertainty requires --protocol-manifest")
    if len(args.audit_dir) != 3:
        raise ProtocolError("uncertainty summary requires exactly three audits")
    result_frames = []
    delta_frames = []
    assignments = set()
    audit_inputs = []
    for directory_value in args.audit_dir:
        directory = Path(directory_value).expanduser().resolve()
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if (
            run.get("status") != "completed"
            or run.get("command") != "audit-study"
            or run.get("protocol_fingerprint") != protocol.fingerprint
        ):
            raise ProtocolError("uncertainty input is not a completed canonical audit")
        seed = int(run["seed"])
        validate_upstream_run(
            directory,
            command="audit-study",
            role="final-audit",
            protocol=protocol,
            seed=seed,
        )
        audit_inputs.append(
            {
                "directory": str(directory),
                "seed": seed,
                "artifact_sha256": run["artifact_sha256"],
                "canonical_results_sha256": _sha256_file(
                    directory / "canonical_results.csv"
                ),
                "comparison_results_sha256": _sha256_file(
                    directory / "comparison_results.csv"
                ),
            }
        )
        assignments.add(protocol.seed_binding(seed).normalized_assignment_sha256)
        result_frames.append(pd.read_csv(directory / "canonical_results.csv"))
        delta_frames.append(pd.read_csv(directory / "comparison_results.csv"))
    if len(assignments) != 3:
        raise ProtocolError("uncertainty inputs must use three distinct assignments")
    results = pd.concat(result_frames, ignore_index=True)
    deltas = pd.concat(delta_frames, ignore_index=True)
    result_keys = [
        "model",
        "query",
        "prototype",
        "distance",
        "candidate_space",
        "slice",
    ]
    result_rows = []
    for key, group in results.groupby(result_keys, sort=True):
        for metric in ("accuracy", "macro_f1", "weighted_f1"):
            stats = paired_uncertainty(group[metric].astype(float).tolist())
            result_rows.append(
                {**dict(zip(result_keys, key)), "metric": metric, **stats}
            )
    delta_keys = [
        "experiment_id",
        "left_model",
        "right_model",
        "query",
        "prototype",
        "distance",
        "candidate_space",
        "slice",
    ]
    delta_rows = []
    for key, group in deltas.groupby(delta_keys, sort=True):
        for metric in ("accuracy", "macro_f1", "weighted_f1"):
            stats = paired_uncertainty(group[f"{metric}_delta"].astype(float).tolist())
            delta_rows.append({**dict(zip(delta_keys, key)), "metric": metric, **stats})
    output = _require_new_output(args.output)
    output.mkdir()
    results.to_csv(output / "per_seed_results.csv", index=False)
    deltas.to_csv(output / "per_seed_comparison_results.csv", index=False)
    pd.DataFrame(result_rows).to_csv(output / "metric_uncertainty.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(
        output / "paired_delta_uncertainty.csv", index=False
    )
    _write_json(
        output / "summary.json",
        {
            "protocol_fingerprint": protocol.fingerprint,
            "audit_inputs": audit_inputs,
            "n": 3,
            "df": 2,
            "warning": "n=3; confidence intervals may be wide",
        },
    )
    args._output_role = "uncertainty"
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
    parser.add_argument("--protocol-manifest")
    parser.add_argument("--model-role", choices=PROTOCOL_MODELS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="implant-experiment")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-data")
    validate.add_argument("--dataset-manifest", required=True)

    split = commands.add_parser("build-split")
    split.add_argument("--dataset-manifest", required=True)
    split.add_argument("--split-manifest", required=True)
    split.add_argument("--protocol-manifest")
    split.add_argument("--output", required=True)

    training = commands.add_parser("train")
    training.add_argument("--dataset-manifest", required=True)
    training.add_argument("--split-dir", required=True)
    training.add_argument("--experiment-manifest", required=True)
    training.add_argument("--init-checkpoint")
    training.add_argument("--seed", type=int, required=True)
    training.add_argument("--protocol-manifest")
    training.add_argument("--model-role", choices=PROTOCOL_MODELS)
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
    prototype.add_argument("--protocol-manifest")
    prototype.add_argument("--seed", type=int)
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
    evaluation.add_argument("--protocol-manifest")
    evaluation.add_argument("--seed", type=int)
    evaluation.add_argument("--model-role", choices=PROTOCOL_MODELS)
    evaluation.add_argument("--query-view", choices=["validation", "test"])
    evaluation.add_argument("--prototype-type", choices=["gt", "mean", "ss"])
    evaluation.add_argument("--validation-freeze")
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
    compare.add_argument("--protocol-manifest")
    compare.add_argument("--seed", type=int)
    compare.add_argument("--left-model", choices=PROTOCOL_MODELS)
    compare.add_argument("--right-model", choices=PROTOCOL_MODELS)
    compare.add_argument("--query-view", choices=["validation", "test"])
    compare.add_argument("--prototype-type", choices=["gt", "mean", "ss"])
    compare.add_argument("--distance", choices=["euclidean", "cosine"])
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

    freeze = commands.add_parser("freeze-validation")
    freeze.add_argument("--protocol-manifest", required=True)
    freeze.add_argument("--seed", type=int, required=True)
    freeze.add_argument("--pretest-ledger", required=True)
    freeze.add_argument("--output", required=True)

    audit = commands.add_parser("audit-study")
    audit.add_argument("--protocol-manifest", required=True)
    audit.add_argument("--seed", type=int, required=True)
    audit.add_argument("--validation-freeze", required=True)
    audit.add_argument("--evaluation-ledger", required=True)
    audit.add_argument("--comparison-ledger", required=True)
    audit.add_argument("--output", required=True)

    uncertainty = commands.add_parser("summarize-uncertainty")
    uncertainty.add_argument("--protocol-manifest", required=True)
    uncertainty.add_argument("--audit-dir", action="append", required=True)
    uncertainty.add_argument("--output", required=True)
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
    "freeze-validation": freeze_validation_command,
    "audit-study": audit_study_command,
    "summarize-uncertainty": summarize_uncertainty_command,
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
