"""Strict manifest loading, inheritance, references, and CLI overrides."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


SCHEMA_VERSION = 1
KINDS = {"dataset", "transform", "experiment", "checkpoint_catalog"}
ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
IMMUTABLE_OVERRIDE_ROOTS = {"schema_version", "kind", "id"}
PATH_KEYS = {
    "path",
    "image_root",
    "output_root",
    "manifest",
    "label_map",
    "checkpoint",
}


class WorkflowError(Exception):
    exit_code = 10


class CliError(WorkflowError):
    exit_code = 2


class ManifestError(WorkflowError):
    exit_code = 3


class DataError(WorkflowError):
    exit_code = 4


class CheckpointError(WorkflowError):
    exit_code = 5


class DependencyError(WorkflowError):
    exit_code = 6


class TaskConfigError(WorkflowError):
    exit_code = 7


@dataclass(frozen=True)
class ResolvedManifest:
    data: dict[str, Any]
    root_path: Path
    source_manifests: tuple[dict[str, str], ...]
    overrides: tuple[dict[str, Any], ...]
    manifest_hash: str


COMMON_FIELDS = {"schema_version", "kind", "id"}
ROOT_FIELDS = {
    "dataset": COMMON_FIELDS
    | {"sources", "duplicate_policy", "labels", "split", "views"},
    "transform": COMMON_FIELDS | {"pipelines"},
    "experiment": COMMON_FIELDS
    | {
        "extends",
        "datasets",
        "transform",
        "checkpoint_catalog",
        "stages",
        "seed",
        "determinism",
        "runtime",
        "output_root",
        "artifact_policy",
    },
    "checkpoint_catalog": COMMON_FIELDS | {"checkpoints"},
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
VIEW_FIELDS = {"id", "from", "usage", "filters", "sampling"}
SAMPLING_FIELDS = {"type", "count", "seed"}
LABEL_FIELDS = {"mode", "reference_view", "values", "aliases"}
SPLIT_FIELDS = {
    "strategy",
    "mapping",
    "ratios",
    "seed",
    "validation_folds",
    "test_folds",
    "remaining",
}
STAGE_FIELDS = {
    "id",
    "task",
    "inputs",
    "transforms",
    "loaders",
    "model",
    "initialization",
    "objective",
    "optimizer",
    "scheduler",
    "budget",
    "checkpoint",
    "metrics",
    "depends_on",
    "prototype",
    "retrieval",
    "open_set",
    "visualization",
}
LOADER_FIELDS = {
    "batch_size",
    "num_workers",
    "shuffle",
    "sampler",
    "drop_last",
    "pin_memory",
    "persistent_workers",
}
TRANSFORM_STEP_FIELDS = {"name", "params"}
CHECKPOINT_FIELDS = {
    "path",
    "model",
    "adapter",
    "label_map",
    "num_classes",
    "training_state",
    "completed_epoch",
    "planned_epoch",
    "transform",
    "dataset",
    "historical_metrics",
    "validation_status",
}
MODEL_FIELDS = {"identifier", "parameters", "num_classes"}
INITIALIZATION_FIELDS = {
    "type",
    "checkpoint",
    "checkpoint_format",
    "strict",
    "source_stage",
    "freeze_encoder",
    "projection_head",
    "projection_dim",
    "projection_hidden_dim",
    "reset_optimizer",
}
OBJECTIVE_FIELDS = {
    "type",
    "views",
    "classification_weight",
    "contrastive_weight",
    "classification_views",
    "classification_reduction",
    "temperature",
    "projection_head",
    "projection_dim",
    "projection_hidden_dim",
    "label_smoothing",
    "margin",
}
OPTIMIZER_FIELDS = {
    "type",
    "learning_rate",
    "weight_decay",
    "momentum",
}
SCHEDULER_FIELDS = {"type", "step_size", "gamma", "eta_min"}
BUDGET_FIELDS = {"epochs", "gradient_accumulation"}
CHECKPOINT_POLICY_FIELDS = {"selection_metric", "mode", "save_format"}
RUNTIME_FIELDS = {"device", "precision"}
ARTIFACT_POLICY_FIELDS = {"save_predictions", "save_embeddings"}
PROTOTYPE_FIELDS = {"reducer", "distance", "normalize"}
RETRIEVAL_FIELDS = {"k", "distance", "normalize"}
OPEN_SET_FIELDS = {"score", "threshold_method", "target_tpr"}
VISUALIZATION_FIELDS = {
    "algorithm",
    "seed",
    "max_points",
    "perplexity",
    "iterations",
    "n_neighbors",
    "min_dist",
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _error(path: Path, field: str, message: str) -> ManifestError:
    return ManifestError(f"{path}: {field}: {message}")


def _reject_unknown(
    value: dict[str, Any], allowed: set[str], path: Path, field: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise _error(path, field, f"unknown fields: {unknown}")


def _require(
    value: dict[str, Any], fields: Iterable[str], path: Path, field: str
) -> None:
    missing = sorted(name for name in fields if name not in value)
    if missing:
        raise _error(path, field, f"missing required fields: {missing}")


def _validate_common(data: dict[str, Any], path: Path) -> str:
    if not isinstance(data, dict):
        raise _error(path, "$", "root must be an object")
    _require(data, COMMON_FIELDS, path, "$")
    if data["schema_version"] != SCHEMA_VERSION:
        raise _error(
            path,
            "schema_version",
            f"supported version is {SCHEMA_VERSION}, got {data['schema_version']!r}",
        )
    kind = data["kind"]
    if kind not in KINDS:
        raise _error(path, "kind", f"must be one of {sorted(KINDS)}")
    if not isinstance(data["id"], str) or not data["id"].strip():
        raise _error(path, "id", "must be a non-empty string")
    _reject_unknown(data, ROOT_FIELDS[kind], path, "$")
    return kind


def _validate_dataset(data: dict[str, Any], path: Path) -> None:
    _require(data, {"sources", "labels", "split", "views"}, path, "$")
    if not isinstance(data["sources"], list) or not data["sources"]:
        raise _error(path, "sources", "must be a non-empty array")
    source_ids: set[str] = set()
    for index, source in enumerate(data["sources"]):
        field = f"sources.{index}"
        if not isinstance(source, dict):
            raise _error(path, field, "must be an object")
        _reject_unknown(source, SOURCE_FIELDS, path, field)
        _require(source, {"id", "type", "role", "path"}, path, field)
        if source["id"] in source_ids:
            raise _error(path, f"{field}.id", "must be unique")
        source_ids.add(source["id"])
        if source["type"] not in {"csv", "parquet", "folder_label"}:
            raise _error(path, f"{field}.type", "must be csv, parquet, or folder_label")
        if source["role"] not in {"sample", "prototype", "unknown"}:
            raise _error(path, f"{field}.role", "invalid source role")
        if source["type"] != "folder_label":
            if not isinstance(source.get("columns"), dict):
                raise _error(path, f"{field}.columns", "is required for table sources")
            required = {"image_uri", "label"}
            if not required <= set(source["columns"]):
                raise _error(path, f"{field}.columns", "must map image_uri and label")
            allowed = {
                "image_uri",
                "label",
                "sample_id",
                "split",
                "fold",
                "group_id",
                "source_metadata",
            }
            unknown = sorted(set(source["columns"]) - allowed)
            if unknown:
                raise _error(path, f"{field}.columns", f"unknown mappings: {unknown}")
    labels = data["labels"]
    if not isinstance(labels, dict):
        raise _error(path, "labels", "must be an object")
    _reject_unknown(labels, LABEL_FIELDS, path, "labels")
    _require(labels, {"mode", "reference_view"}, path, "labels")
    if labels["mode"] not in {"explicit", "derived"}:
        raise _error(path, "labels.mode", "must be explicit or derived")
    if labels["mode"] == "explicit" and not isinstance(labels.get("values"), list):
        raise _error(path, "labels.values", "is required for explicit mode")
    if "aliases" in labels and not isinstance(labels["aliases"], dict):
        raise _error(path, "labels.aliases", "must be an object")
    split = data["split"]
    if not isinstance(split, dict):
        raise _error(path, "split", "must be an object")
    _reject_unknown(split, SPLIT_FIELDS, path, "split")
    _require(split, {"strategy"}, path, "split")
    if split["strategy"] not in {"predefined", "fold_column", "stratified_hash"}:
        raise _error(path, "split.strategy", "invalid split strategy")
    views = data["views"]
    if not isinstance(views, list) or not views:
        raise _error(path, "views", "must be a non-empty array")
    view_ids: set[str] = set()
    for index, view in enumerate(views):
        field = f"views.{index}"
        if not isinstance(view, dict):
            raise _error(path, field, "must be an object")
        _reject_unknown(view, VIEW_FIELDS, path, field)
        _require(view, {"id", "from", "usage"}, path, field)
        if view["id"] in view_ids:
            raise _error(path, f"{field}.id", "must be unique")
        view_ids.add(view["id"])
        if view["usage"] not in {
            "train",
            "validation",
            "test",
            "prototype",
            "unknown_test",
            "analysis",
        }:
            raise _error(path, f"{field}.usage", "invalid usage")
        if "filters" in view and not isinstance(view["filters"], dict):
            raise _error(path, f"{field}.filters", "must be an object")
        if "sampling" in view:
            sampling = view["sampling"]
            if not isinstance(sampling, dict):
                raise _error(path, f"{field}.sampling", "must be an object")
            _reject_unknown(sampling, SAMPLING_FIELDS, path, f"{field}.sampling")
            if sampling.get("type", "all") not in {"all", "per_label"}:
                raise _error(
                    path,
                    f"{field}.sampling.type",
                    "must be all or per_label",
                )


def _validate_transform(data: dict[str, Any], path: Path) -> None:
    _require(data, {"pipelines"}, path, "$")
    if not isinstance(data["pipelines"], dict) or not data["pipelines"]:
        raise _error(path, "pipelines", "must be a non-empty object")
    for pipeline_id, steps in data["pipelines"].items():
        if not isinstance(steps, list) or not steps:
            raise _error(path, f"pipelines.{pipeline_id}", "must be a non-empty array")
        for index, step in enumerate(steps):
            field = f"pipelines.{pipeline_id}.{index}"
            if not isinstance(step, dict):
                raise _error(path, field, "must be an object")
            _reject_unknown(step, TRANSFORM_STEP_FIELDS, path, field)
            _require(step, {"name"}, path, field)
            if "params" in step and not isinstance(step["params"], dict):
                raise _error(path, f"{field}.params", "must be an object")


def _validate_stage(stage: dict[str, Any], path: Path, index: int) -> None:
    field = f"stages.{index}"
    _reject_unknown(stage, STAGE_FIELDS, path, field)
    _require(stage, {"id", "task", "inputs"}, path, field)
    if not isinstance(stage["inputs"], dict) or not stage["inputs"]:
        raise _error(path, f"{field}.inputs", "must be a non-empty object")
    for role, loader in stage.get("loaders", {}).items():
        loader_field = f"{field}.loaders.{role}"
        if not isinstance(loader, dict):
            raise _error(path, loader_field, "must be an object")
        _reject_unknown(loader, LOADER_FIELDS, path, loader_field)
        _require(loader, LOADER_FIELDS, path, loader_field)
    nested = (
        ("model", MODEL_FIELDS),
        ("initialization", INITIALIZATION_FIELDS),
        ("objective", OBJECTIVE_FIELDS),
        ("optimizer", OPTIMIZER_FIELDS),
        ("scheduler", SCHEDULER_FIELDS),
        ("budget", BUDGET_FIELDS),
        ("checkpoint", CHECKPOINT_POLICY_FIELDS),
        ("prototype", PROTOTYPE_FIELDS),
        ("retrieval", RETRIEVAL_FIELDS),
        ("open_set", OPEN_SET_FIELDS),
        ("visualization", VISUALIZATION_FIELDS),
    )
    for name, allowed in nested:
        if name in stage:
            if not isinstance(stage[name], dict):
                raise _error(path, f"{field}.{name}", "must be an object")
            _reject_unknown(stage[name], allowed, path, f"{field}.{name}")


def _validate_experiment(data: dict[str, Any], path: Path) -> None:
    _require(
        data,
        {
            "datasets",
            "transform",
            "stages",
            "seed",
            "determinism",
            "runtime",
            "output_root",
        },
        path,
        "$",
    )
    if not isinstance(data["datasets"], dict) or not data["datasets"]:
        raise _error(path, "datasets", "must be a non-empty object")
    if not isinstance(data["stages"], list) or not data["stages"]:
        raise _error(path, "stages", "must be a non-empty array")
    stage_ids: set[str] = set()
    for index, stage in enumerate(data["stages"]):
        if not isinstance(stage, dict):
            raise _error(path, f"stages.{index}", "must be an object")
        _validate_stage(stage, path, index)
        if stage["id"] in stage_ids:
            raise _error(path, f"stages.{index}.id", "must be unique")
        stage_ids.add(stage["id"])
    if not isinstance(data["seed"], int):
        raise _error(path, "seed", "must be an integer")
    if data["determinism"] not in {"strict", "best_effort"}:
        raise _error(path, "determinism", "must be strict or best_effort")
    if not isinstance(data["runtime"], dict):
        raise _error(path, "runtime", "must be an object")
    _reject_unknown(data["runtime"], RUNTIME_FIELDS, path, "runtime")
    _require(data["runtime"], RUNTIME_FIELDS, path, "runtime")
    if "artifact_policy" in data:
        _reject_unknown(
            data["artifact_policy"], ARTIFACT_POLICY_FIELDS, path, "artifact_policy"
        )


def _validate_checkpoint_catalog(data: dict[str, Any], path: Path) -> None:
    _require(data, {"checkpoints"}, path, "$")
    if not isinstance(data["checkpoints"], dict):
        raise _error(path, "checkpoints", "must be an object")
    for checkpoint_id, checkpoint in data["checkpoints"].items():
        field = f"checkpoints.{checkpoint_id}"
        if not isinstance(checkpoint, dict):
            raise _error(path, field, "must be an object")
        _reject_unknown(checkpoint, CHECKPOINT_FIELDS, path, field)
        _require(
            checkpoint,
            {
                "path",
                "model",
                "adapter",
                "num_classes",
                "training_state",
                "completed_epoch",
                "planned_epoch",
                "validation_status",
            },
            path,
            field,
        )
        if checkpoint["validation_status"] not in {
            "verified",
            "unverified_preprocessing",
            "unavailable",
        }:
            raise _error(
                path,
                f"{field}.validation_status",
                "must be verified, unverified_preprocessing, or unavailable",
            )
        if checkpoint["adapter"] not in {
            "model_state_dict",
            "state_dict",
            "raw_state_dict",
            "classifier_checkpoint",
            "supcon_checkpoint",
        }:
            raise _error(
                path,
                f"{field}.adapter",
                "unknown state dictionary adapter",
            )
        if checkpoint["validation_status"] != "unavailable" and not checkpoint.get(
            "label_map"
        ):
            raise _error(
                path,
                f"{field}.label_map",
                "is required for an available legacy checkpoint",
            )
        if checkpoint["validation_status"] == "verified" and not checkpoint.get(
            "transform"
        ):
            raise _error(
                path,
                f"{field}.transform",
                "is required for a verified legacy checkpoint",
            )


def validate_manifest(data: dict[str, Any], path: Path) -> None:
    kind = _validate_common(data, path)
    {
        "dataset": _validate_dataset,
        "transform": _validate_transform,
        "experiment": _validate_experiment,
        "checkpoint_catalog": _validate_checkpoint_catalog,
    }[kind](data, path)


def _substitute_env(value: Any, path: Path, field: str = "$") -> Any:
    if isinstance(value, dict):
        return {
            key: _substitute_env(item, path, f"{field}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _substitute_env(item, path, f"{field}.{index}")
            for index, item in enumerate(value)
        ]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise _error(path, field, f"environment variable {name!r} is not set")
        return os.environ[name]

    replaced = ENV_PATTERN.sub(replace, value)
    if "${" in replaced:
        raise _error(path, field, "unsupported environment expression")
    return replaced


def _path(value: Any, origin: Path, field: str, *, must_exist: bool) -> str:
    if not isinstance(value, str) or not value:
        raise _error(origin, field, "path must be a non-empty string")
    candidate = Path(value)
    if candidate.parts and candidate.parts[0] == "~":
        raise _error(origin, field, "'~' expansion is not supported")
    if not candidate.is_absolute():
        candidate = origin.parent / candidate
    candidate = candidate.resolve(strict=False)
    if must_exist and not candidate.exists():
        raise _error(origin, field, f"path does not exist: {candidate}")
    return str(candidate.resolve(strict=must_exist))


def _resolve_declared_paths(data: dict[str, Any], origin: Path) -> dict[str, Any]:
    data = copy.deepcopy(data)
    kind = data.get("kind")
    if kind == "dataset":
        for index, source in enumerate(data.get("sources", [])):
            source["path"] = _path(
                source["path"], origin, f"sources.{index}.path", must_exist=True
            )
            if "image_root" in source:
                source["image_root"] = _path(
                    source["image_root"],
                    origin,
                    f"sources.{index}.image_root",
                    must_exist=True,
                )
    elif kind == "experiment":
        if "extends" in data:
            data["extends"] = _path(data["extends"], origin, "extends", must_exist=True)
        for alias, reference in data.get("datasets", {}).items():
            data["datasets"][alias] = _path(
                reference, origin, f"datasets.{alias}", must_exist=True
            )
        if "transform" in data:
            data["transform"] = _path(
                data["transform"], origin, "transform", must_exist=True
            )
        if data.get("checkpoint_catalog"):
            data["checkpoint_catalog"] = _path(
                data["checkpoint_catalog"],
                origin,
                "checkpoint_catalog",
                must_exist=True,
            )
        if "output_root" in data:
            data["output_root"] = _path(
                data["output_root"], origin, "output_root", must_exist=False
            )
        for index, stage in enumerate(data.get("stages", [])):
            initialization = stage.get("initialization", {})
            if (
                initialization.get("checkpoint")
                and initialization.get("type") == "path"
            ):
                initialization["checkpoint"] = _path(
                    initialization["checkpoint"],
                    origin,
                    f"stages.{index}.initialization.checkpoint",
                    must_exist=True,
                )
    elif kind == "checkpoint_catalog":
        for checkpoint_id, checkpoint in data.get("checkpoints", {}).items():
            checkpoint["path"] = _path(
                checkpoint["path"],
                origin,
                f"checkpoints.{checkpoint_id}.path",
                must_exist=False,
            )
            if checkpoint.get("label_map"):
                checkpoint["label_map"] = _path(
                    checkpoint["label_map"],
                    origin,
                    f"checkpoints.{checkpoint_id}.label_map",
                    must_exist=True,
                )
    return data


def deep_merge(parent: Any, child: Any) -> Any:
    if isinstance(parent, dict) and isinstance(child, dict):
        merged = copy.deepcopy(parent)
        for key, value in child.items():
            merged[key] = (
                deep_merge(merged[key], value)
                if key in merged
                else copy.deepcopy(value)
            )
        return merged
    return copy.deepcopy(child)


def _parse_override(value: str) -> Any:
    parsed = yaml.safe_load(value)
    if isinstance(parsed, (dict, list)):
        raise CliError("CLI override values must be scalar")
    return parsed


def _lookup(container: Any, part: str, full_path: str) -> tuple[Any, str | int]:
    if isinstance(container, list):
        try:
            index = int(part)
        except ValueError as exc:
            raise CliError(f"override {full_path!r}: expected array index") from exc
        if index < 0 or index >= len(container):
            raise CliError(f"override {full_path!r}: array index is out of range")
        return container, index
    if not isinstance(container, dict) or part not in container:
        raise CliError(f"override {full_path!r}: field does not exist")
    return container, part


def apply_overrides(
    data: dict[str, Any], values: Iterable[str], cwd: Path
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    resolved = copy.deepcopy(data)
    applied: list[dict[str, Any]] = []
    for expression in values:
        if "=" not in expression:
            raise CliError(f"override must be FIELD=VALUE, got {expression!r}")
        field, raw = expression.split("=", 1)
        parts = field.split(".")
        if not field or parts[0] in IMMUTABLE_OVERRIDE_ROOTS or parts[-1] == "id":
            raise CliError(f"override of {field!r} is forbidden")
        current: Any = resolved
        for part in parts[:-1]:
            current, key = _lookup(current, part, field)
            current = current[key]
        parent, key = _lookup(current, parts[-1], field)
        old = parent[key]
        if isinstance(old, (dict, list)) or old is None:
            raise CliError(
                f"override {field!r}: target must be an existing scalar leaf"
            )
        new = _parse_override(raw)
        if isinstance(old, bool) and not isinstance(new, bool):
            raise CliError(f"override {field!r}: expected boolean")
        if (
            isinstance(old, int)
            and not isinstance(old, bool)
            and not isinstance(new, int)
        ):
            raise CliError(f"override {field!r}: expected integer")
        if isinstance(old, float) and not isinstance(new, (int, float)):
            raise CliError(f"override {field!r}: expected number")
        if parts[-1] in PATH_KEYS and isinstance(new, str):
            path = Path(new)
            if not path.is_absolute():
                path = cwd / path
            new = str(path.resolve(strict=False))
        parent[key] = new
        applied.append({"field": field, "old": old, "new": new})
    return resolved, tuple(applied)


def _meaningful_manifest(data: dict[str, Any]) -> dict[str, Any]:
    excluded = {"output_root", "run_id", "run_time", "verbosity", "gpu_ordinal"}

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: clean(item) for key, item in value.items() if key not in excluded
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return clean(data)


def resolved_manifest_hash(data: dict[str, Any]) -> str:
    return sha256_json(_meaningful_manifest(data))


class ManifestResolver:
    def __init__(self) -> None:
        self._sources: dict[Path, dict[str, str]] = {}

    def _read(self, path: Path) -> dict[str, Any]:
        path = path.resolve()
        try:
            documents = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
        except (OSError, yaml.YAMLError) as exc:
            raise ManifestError(f"{path}: cannot read manifest: {exc}") from exc
        documents = [item for item in documents if item is not None]
        if len(documents) != 1:
            raise ManifestError(f"{path}: exactly one root manifest is required")
        data = _substitute_env(documents[0], path)
        if not isinstance(data, dict):
            raise ManifestError(f"{path}: root must be an object")
        _validate_common(data, path)
        data = _resolve_declared_paths(data, path)
        self._sources[path] = {
            "path": str(path),
            "kind": str(data["kind"]),
            "id": str(data["id"]),
            "sha256": sha256_file(path),
        }
        return data

    def _resolve(
        self,
        path: Path,
        expected_kind: str | None,
        stack: tuple[Path, ...],
        overrides: Iterable[str] = (),
        cwd: Path | None = None,
        embed_refs: bool = True,
    ) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
        path = path.resolve()
        if path in stack:
            cycle = " -> ".join(str(item) for item in (*stack, path))
            raise ManifestError(f"manifest reference cycle: {cycle}")
        data = self._read(path)
        if expected_kind and data["kind"] != expected_kind:
            raise ManifestError(
                f"{path}: expected kind {expected_kind!r}, got {data['kind']!r}"
            )
        next_stack = (*stack, path)
        if data["kind"] == "experiment" and data.get("extends"):
            parent, _ = self._resolve(
                Path(data["extends"]), "experiment", next_stack, embed_refs=False
            )
            child = {key: value for key, value in data.items() if key != "extends"}
            data = deep_merge(parent, child)
        applied: tuple[dict[str, Any], ...] = ()
        if overrides:
            data, applied = apply_overrides(data, overrides, cwd or Path.cwd())
        validate_manifest(data, path)
        if not embed_refs:
            return data, applied
        if data["kind"] == "experiment":
            data["datasets"] = {
                alias: self._resolve(Path(reference), "dataset", next_stack)[0]
                for alias, reference in data["datasets"].items()
            }
            data["transform"] = self._resolve(
                Path(data["transform"]), "transform", next_stack
            )[0]
            if data.get("checkpoint_catalog"):
                data["checkpoint_catalog"] = self._resolve(
                    Path(data["checkpoint_catalog"]),
                    "checkpoint_catalog",
                    next_stack,
                )[0]
        return data, applied

    def resolve(
        self, root: str | Path, overrides: Iterable[str] = (), cwd: Path | None = None
    ) -> ResolvedManifest:
        root_path = Path(root).resolve()
        data, applied = self._resolve(
            root_path, None, (), overrides=tuple(overrides), cwd=cwd
        )
        by_id: dict[str, list[str]] = {}
        for source in self._sources.values():
            by_id.setdefault(source["id"], []).append(source["path"])
        duplicates = {
            manifest_id: paths for manifest_id, paths in by_id.items() if len(paths) > 1
        }
        if duplicates:
            raise ManifestError(
                f"manifest ids must be unique in the dependency graph: {duplicates}"
            )
        manifest_hash = resolved_manifest_hash(data)
        return ResolvedManifest(
            data=data,
            root_path=root_path,
            source_manifests=tuple(
                self._sources[path] for path in sorted(self._sources)
            ),
            overrides=applied,
            manifest_hash=manifest_hash,
        )
