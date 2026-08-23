"""Artifact store and sequential task runner for resolved experiments."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from .materialize import (
    MaterializedDataset,
    materialize_dataset,
    validate_bound_inputs,
    write_parquet,
)
from .schema import (
    CheckpointError,
    CliError,
    DataError,
    DependencyError,
    ResolvedManifest,
    ManifestResolver,
    TaskConfigError,
    sha256_file,
    sha256_json,
    resolved_manifest_hash,
)
from .transforms import build_loader, build_pipeline, derived_seed


TASKS = {
    "classification_training",
    "joint_supcon_training",
    "supcon_pretraining",
    "classification_finetuning",
    "classification_evaluation",
    "embedding_extraction",
    "prototype_evaluation",
    "retrieval_evaluation",
    "open_set_evaluation",
    "embedding_visualization",
}
EVALUATION_TASKS = {
    "classification_evaluation",
    "embedding_extraction",
    "prototype_evaluation",
    "retrieval_evaluation",
    "open_set_evaluation",
    "embedding_visualization",
}
TRAINING_TASKS = {
    "classification_training",
    "joint_supcon_training",
    "supcon_pretraining",
    "classification_finetuning",
}
TASK_OUTPUTS = {
    **{
        task: {"best_checkpoint", "last_checkpoint", "metrics"}
        for task in TRAINING_TASKS
    },
    "classification_evaluation": {
        "predictions",
        "metrics",
        "per_class_metrics",
        "confusion_matrix",
    },
    "embedding_extraction": {"embeddings", "embedding_metadata", "metadata"},
    "prototype_evaluation": {
        "prototypes",
        "prototype_membership",
        "predictions",
        "metrics",
    },
    "retrieval_evaluation": {"metrics"},
    "open_set_evaluation": {"predictions", "metrics"},
    "embedding_visualization": {"coordinates", "image", "metadata"},
}


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(
        path, json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n"
    )


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _git_info(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    commit = run("rev-parse", "HEAD")
    return {
        "commit": commit or None,
        "dirty": bool(run("status", "--porcelain")),
        "branch": run("branch", "--show-current") or None,
    }


def _environment(runtime: dict[str, Any], determinism: str) -> dict[str, Any]:
    packages = {}
    for name in (
        "torch",
        "torchvision",
        "timm",
        "numpy",
        "pandas",
        "scikit-learn",
        "Pillow",
        "PyYAML",
        "pyarrow",
        "umap-learn",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "precision": runtime["precision"],
        "determinism": determinism,
    }
    try:
        import torch

        result.update(
            {
                "pytorch": torch.__version__,
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "gpu": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
            }
        )
    except ImportError:
        pass
    return result


@dataclass
class RunArtifacts:
    path: Path
    resolved: ResolvedManifest
    command: str
    git: dict[str, Any]

    @classmethod
    def create(cls, resolved: ResolvedManifest, command: str) -> "RunArtifacts":
        data = resolved.data
        root = Path(data["output_root"])
        run_id = f"{_utc_stamp()}-{resolved.manifest_hash[:8]}"
        path = root / data["id"] / run_id
        path.mkdir(parents=True, exist_ok=False)
        artifact = cls(
            path,
            resolved,
            command,
            _git_info(Path(__file__).resolve().parents[3]),
        )
        (path / "logs").mkdir()
        (path / "views").mkdir()
        artifact.status("created")
        _atomic_text(
            path / "resolved_manifest.yaml",
            yaml.safe_dump(
                data, allow_unicode=True, sort_keys=False, default_flow_style=False
            ),
        )
        _write_json(path / "source_manifests.json", resolved.source_manifests)
        _write_json(
            path / "run_metadata.json",
            {
                "run_id": run_id,
                "command": command,
                "manifest_hash": resolved.manifest_hash,
                "seed": data.get("seed"),
                "seed_policy": (
                    "sha256(base_seed, dataset|stage, operation, epoch, worker|view)"
                    if data.get("seed") is not None
                    else None
                ),
                "overrides": resolved.overrides,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        _write_json(path / "git.json", artifact.git)
        if data["kind"] == "experiment":
            _write_json(
                path / "environment.json",
                _environment(data["runtime"], data["determinism"]),
            )
        return artifact

    def status(
        self,
        state: str,
        *,
        stage: str | None = None,
        error: BaseException | None = None,
        skips: list[str] | None = None,
    ) -> None:
        _write_json(
            self.path / "status.json",
            {
                "status": state,
                "stage": stage,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "error": (
                    {
                        "type": type(error).__name__,
                        "message": str(error),
                        "exit_code": getattr(error, "exit_code", 10),
                    }
                    if error
                    else None
                ),
                "skips": skips or [],
            },
        )


@dataclass
class ExperimentContext:
    resolved: ResolvedManifest
    artifacts: RunArtifacts
    datasets: dict[str, MaterializedDataset]
    stage_artifacts: dict[str, dict[str, Path]] = field(default_factory=dict)
    resuming: bool = False

    @property
    def experiment(self) -> dict[str, Any]:
        return self.resolved.data

    @property
    def seed(self) -> int:
        return int(self.experiment["seed"])

    def dataset_input(
        self, stage: dict[str, Any], role: str
    ) -> tuple[pd.DataFrame, str]:
        try:
            reference = stage["inputs"][role]
            alias, view_id = reference.split(".", 1)
            dataset = self.datasets[alias]
            return dataset.views[view_id], reference
        except (KeyError, ValueError) as exc:
            raise TaskConfigError(
                f"stage {stage['id']}: input role {role!r} must bind a dataset view"
            ) from exc

    def artifact_input(self, stage: dict[str, Any], role: str) -> Path:
        try:
            reference = stage["inputs"][role]
            stage_id, artifact_id = reference.split(".", 1)
            return self.stage_artifacts[stage_id][artifact_id]
        except (KeyError, ValueError) as exc:
            raise TaskConfigError(
                f"stage {stage['id']}: input role {role!r} must bind a previous stage artifact"
            ) from exc

    def label_map_for(self, reference: str) -> dict[str, int]:
        alias = reference.split(".", 1)[0]
        return self.datasets[alias].label_map


def _materialize(resolved: ResolvedManifest) -> dict[str, MaterializedDataset]:
    if resolved.data["kind"] != "experiment":
        raise TaskConfigError("materialization root must be an experiment")
    datasets = {
        alias: materialize_dataset(config, int(resolved.data["seed"]))
        for alias, config in resolved.data["datasets"].items()
    }
    validate_bound_inputs(resolved.data, datasets)
    _validate_stage_graph(resolved.data, datasets)
    return datasets


def _finalize_resolved(
    resolved: ResolvedManifest, datasets: dict[str, MaterializedDataset]
) -> ResolvedManifest:
    data = copy.deepcopy(resolved.data)
    dynamic = {}
    for alias, dataset in datasets.items():
        dynamic[alias] = {
            "num_classes": len(dataset.label_map),
            "label_map": dataset.label_map,
            "sample_count": len(dataset.records),
            "view_counts": {
                view_id: len(frame) for view_id, frame in dataset.views.items()
            },
            "hashes": dataset.hashes,
        }
    data["materialized"] = {"datasets": dynamic}
    for stage in data["stages"]:
        if stage["task"] not in TRAINING_TASKS:
            continue
        alias = stage["inputs"]["train"].split(".", 1)[0]
        configured = stage["model"].get("num_classes", "auto")
        actual = len(datasets[alias].label_map)
        if configured not in {"auto", actual}:
            raise TaskConfigError(
                f"stage {stage['id']}: model.num_classes={configured} != {actual}"
            )
        stage["model"]["num_classes"] = actual
    return ResolvedManifest(
        data=data,
        root_path=resolved.root_path,
        source_manifests=resolved.source_manifests,
        overrides=resolved.overrides,
        manifest_hash=resolved_manifest_hash(data),
    )


def _write_materialized(
    resolved: ResolvedManifest,
    datasets: dict[str, MaterializedDataset],
    artifacts: RunArtifacts,
) -> None:
    summaries: dict[str, Any] = {}
    hashes: dict[str, str] = {"manifest": resolved.manifest_hash}
    label_maps: dict[str, Any] = {}
    duplicates: dict[str, Any] = {}
    for alias, dataset in datasets.items():
        summaries[alias] = dataset.summary
        label_maps[alias] = {
            "labels": [
                {"index": index, "label": label}
                for label, index in dataset.label_map.items()
            ],
            "hash": dataset.hashes["label_map"],
        }
        hashes.update(
            {f"{alias}:{key}": value for key, value in dataset.hashes.items()}
        )
        duplicates[alias] = dataset.duplicates
        for view_id, frame in dataset.views.items():
            write_parquet(
                frame, artifacts.path / "views" / f"{alias}.{view_id}.parquet"
            )
    _write_json(artifacts.path / "dataset_summary.json", summaries)
    _write_json(artifacts.path / "label_map.json", label_maps)
    _write_json(artifacts.path / "hashes.json", hashes)
    if any(duplicates.values()):
        _write_json(artifacts.path / "duplicates.json", duplicates)


def _validate_stage_graph(
    experiment: dict[str, Any], datasets: dict[str, MaterializedDataset]
) -> None:
    completed: set[str] = set()
    declared = {stage["id"]: stage for stage in experiment["stages"]}
    transform_manifest = experiment["transform"]
    for stage in experiment["stages"]:
        stage_id = stage["id"]
        task = stage["task"]
        if task not in TASKS:
            raise TaskConfigError(
                f"stage {stage_id}: unknown task {task!r}; available={sorted(TASKS)}"
            )
        for dependency in stage.get("depends_on", []):
            if dependency not in completed:
                raise TaskConfigError(
                    f"stage {stage_id}: dependency {dependency!r} is not an earlier stage"
                )
        if (
            stage.get("initialization", {}).get("type") == "stage"
            and stage["initialization"].get("source_stage") not in completed
        ):
            raise TaskConfigError(
                f"stage {stage_id}: initialization.source_stage must be earlier"
            )
        for role, reference in stage["inputs"].items():
            owner, _, artifact = reference.partition(".")
            if owner in datasets:
                if not artifact or artifact not in datasets[owner].views:
                    raise TaskConfigError(
                        f"stage {stage_id} input {role}: unknown view {reference!r}"
                    )
            elif owner not in completed:
                raise TaskConfigError(
                    f"stage {stage_id} input {role}: {reference!r} is not a dataset "
                    "view or earlier stage artifact"
                )
            elif artifact not in TASK_OUTPUTS[declared[owner]["task"]]:
                raise TaskConfigError(
                    f"stage {stage_id} input {role}: task {owner!r} does not "
                    f"produce artifact {artifact!r}"
                )
        for role, pipeline_id in stage.get("transforms", {}).items():
            evaluation = role not in {"train"}
            build_pipeline(transform_manifest, pipeline_id, evaluation=evaluation)
        for role, loader_config in stage.get("loaders", {}).items():
            _validate_loader_config(stage_id, role, loader_config)
        if task in TRAINING_TASKS:
            for field in (
                "model",
                "initialization",
                "objective",
                "optimizer",
                "scheduler",
                "budget",
                "checkpoint",
            ):
                if field not in stage:
                    raise TaskConfigError(f"stage {stage_id}: {field} is required")
            for role in ("train", "validation"):
                if role not in stage["inputs"] or role not in stage.get("loaders", {}):
                    raise TaskConfigError(
                        f"stage {stage_id}: {role} input and loader are required"
                    )
                if role not in stage.get("transforms", {}):
                    raise TaskConfigError(
                        f"stage {stage_id}: {role} transform is required"
                    )
            if not stage["model"].get("identifier"):
                raise TaskConfigError(f"stage {stage_id}: model.identifier is required")
            try:
                import timm
            except ImportError as exc:
                raise DependencyError("timm is required for model validation") from exc
            if not timm.is_model(stage["model"]["identifier"]):
                raise TaskConfigError(
                    f"stage {stage_id}: unknown timm model "
                    f"{stage['model']['identifier']!r}"
                )
            if stage["initialization"].get("type") not in {
                "random",
                "pretrained",
                "path",
                "catalog",
                "stage",
            }:
                raise TaskConfigError(f"stage {stage_id}: invalid initialization.type")
            _validate_external_checkpoint(experiment, stage, datasets, "train")
            if stage["optimizer"].get("type") not in {"adamw", "sgd"}:
                raise TaskConfigError(
                    f"stage {stage_id}: optimizer.type must be adamw or sgd"
                )
            if stage["scheduler"].get("type") not in {"none", "cosine", "step"}:
                raise TaskConfigError(
                    f"stage {stage_id}: scheduler.type must be none, cosine, or step"
                )
            if int(stage["budget"].get("epochs", 0)) <= 0:
                raise TaskConfigError(
                    f"stage {stage_id}: budget.epochs must be positive"
                )
            if stage["checkpoint"].get("save_format") != "pt":
                raise TaskConfigError(
                    f"stage {stage_id}: save_format must be pt for resumable checkpoints"
                )
            objective = stage["objective"]
            if task in {"classification_training", "classification_finetuning"}:
                if objective.get("type") != "cross_entropy":
                    raise TaskConfigError(
                        f"stage {stage_id}: objective.type must be cross_entropy"
                    )
                if task == "classification_finetuning":
                    required_policy = {
                        "freeze_encoder",
                        "reset_optimizer",
                        "projection_head",
                    }
                    missing = sorted(required_policy - set(stage["initialization"]))
                    if missing:
                        raise TaskConfigError(
                            f"stage {stage_id}: missing fine-tuning policies {missing}"
                        )
                    if not stage["initialization"]["reset_optimizer"]:
                        raise TaskConfigError(
                            f"stage {stage_id}: optimizer state reuse is not supported; "
                            "set reset_optimizer=true"
                        )
                    projection_policy = stage["initialization"]["projection_head"]
                    if projection_policy not in {"discard", "keep"}:
                        raise TaskConfigError(
                            f"stage {stage_id}: projection_head must be discard or keep"
                        )
                    if (
                        projection_policy == "keep"
                        and "projection_dim" not in stage["initialization"]
                    ):
                        raise TaskConfigError(
                            f"stage {stage_id}: projection_head=keep requires "
                            "initialization.projection_dim"
                        )
            else:
                required = {"views", "temperature", "projection_head"}
                missing = sorted(required - set(objective))
                if missing:
                    raise TaskConfigError(
                        f"stage {stage_id}: missing SupCon objective fields {missing}"
                    )
                if objective["projection_head"] and "projection_dim" not in objective:
                    raise TaskConfigError(
                        f"stage {stage_id}: projection_head=true requires projection_dim"
                    )
                if int(objective["views"]) < 2:
                    raise TaskConfigError(
                        f"stage {stage_id}: SupCon views must be at least 2"
                    )
                expected_type = (
                    "joint_supcon"
                    if task == "joint_supcon_training"
                    else "supervised_contrastive"
                )
                if objective.get("type") != expected_type:
                    raise TaskConfigError(
                        f"stage {stage_id}: objective.type must be {expected_type}"
                    )
                if task == "joint_supcon_training":
                    joint_fields = {
                        "classification_weight",
                        "contrastive_weight",
                        "classification_views",
                        "classification_reduction",
                    }
                    missing = sorted(joint_fields - set(objective))
                    if missing:
                        raise TaskConfigError(
                            f"stage {stage_id}: missing joint objective fields {missing}"
                        )
                    if objective["classification_views"] not in {"first", "all"}:
                        raise TaskConfigError(
                            f"stage {stage_id}: classification_views must be first or all"
                        )
                    if objective["classification_reduction"] not in {"mean", "sum"}:
                        raise TaskConfigError(
                            f"stage {stage_id}: classification_reduction must be mean or sum"
                        )
        elif task == "classification_evaluation":
            role = "test" if "test" in stage["inputs"] else "query"
            _require_image_role(stage, role)
            _require_checkpoint_source(stage)
            _validate_external_checkpoint(experiment, stage, datasets, role)
        elif task == "embedding_extraction":
            roles = [
                role
                for role in ("data", "input", "query", "test")
                if role in stage["inputs"]
            ]
            if len(roles) != 1:
                raise TaskConfigError(
                    f"stage {stage_id}: bind exactly one data/input/query/test role"
                )
            _require_image_role(stage, roles[0])
            _require_checkpoint_source(stage)
            _validate_external_checkpoint(experiment, stage, datasets, roles[0])
        elif task == "prototype_evaluation":
            _require_artifact_role(stage, "prototype")
            _require_artifact_role(stage, "query")
            if "prototype" not in stage:
                raise TaskConfigError(f"stage {stage_id}: prototype config is required")
        elif task == "retrieval_evaluation":
            _require_artifact_role(stage, "query")
            _require_artifact_role(stage, "gallery")
            if "retrieval" not in stage:
                raise TaskConfigError(f"stage {stage_id}: retrieval config is required")
        elif task == "open_set_evaluation":
            for role in ("calibration", "known", "unknown", "prototypes"):
                _require_artifact_role(stage, role)
            if "open_set" not in stage:
                raise TaskConfigError(f"stage {stage_id}: open_set config is required")
        elif task == "embedding_visualization":
            if not stage["inputs"] or "visualization" not in stage:
                raise TaskConfigError(
                    f"stage {stage_id}: embedding inputs and visualization config are required"
                )
            for role in stage["inputs"]:
                _require_artifact_role(stage, role)
        completed.add(stage_id)


def _require_image_role(stage: dict[str, Any], role: str) -> None:
    if (
        role not in stage["inputs"]
        or role not in stage.get("transforms", {})
        or role not in stage.get("loaders", {})
    ):
        raise TaskConfigError(
            f"stage {stage['id']}: {role} input, transform, and loader are required"
        )


def _validate_loader_config(stage_id: str, role: str, config: dict[str, Any]) -> None:
    if int(config["batch_size"]) <= 0 or int(config["num_workers"]) < 0:
        raise TaskConfigError(
            f"stage {stage_id} loader {role}: invalid batch_size or num_workers"
        )
    if config["num_workers"] == 0 and config["persistent_workers"]:
        raise TaskConfigError(
            f"stage {stage_id} loader {role}: persistent_workers needs workers"
        )
    sampler = config["sampler"]
    if role == "train":
        if config["shuffle"] and sampler != "none":
            raise TaskConfigError(
                f"stage {stage_id}: training shuffle and sampler are exclusive"
            )
        if isinstance(sampler, str) and sampler != "none":
            raise TaskConfigError(
                f"stage {stage_id}: unknown training sampler {sampler!r}"
            )
        if isinstance(sampler, dict) and sampler.get("type") != "class_aware":
            raise TaskConfigError(f"stage {stage_id}: sampler.type must be class_aware")
    elif (
        config["shuffle"]
        or config["drop_last"]
        or not isinstance(sampler, str)
        or sampler not in {"none", "sequential"}
    ):
        raise TaskConfigError(
            f"stage {stage_id} loader {role}: evaluation requires shuffle=false, "
            "drop_last=false, and a sequential sampler"
        )


def _require_artifact_role(stage: dict[str, Any], role: str) -> None:
    if role not in stage["inputs"]:
        raise TaskConfigError(f"stage {stage['id']}: input role {role!r} is required")


def _require_checkpoint_source(stage: dict[str, Any]) -> None:
    if "checkpoint" in stage["inputs"]:
        return
    initialization = stage.get("initialization", {})
    if initialization.get("type") not in {"path", "catalog", "stage"}:
        raise TaskConfigError(
            f"stage {stage['id']}: checkpoint input or initialization is required"
        )


def _validate_external_checkpoint(
    experiment: dict[str, Any],
    stage: dict[str, Any],
    datasets: dict[str, MaterializedDataset],
    data_role: str,
) -> None:
    initialization = stage.get("initialization", {})
    kind = initialization.get("type")
    if kind == "path":
        path = Path(initialization["checkpoint"])
        if not path.is_file():
            raise CheckpointError(f"checkpoint does not exist: {path}")
        return
    if kind != "catalog":
        return
    entries = experiment.get("checkpoint_catalog", {}).get("checkpoints", {})
    checkpoint_id = initialization.get("checkpoint")
    if checkpoint_id not in entries:
        raise CheckpointError(f"unknown checkpoint catalog entry {checkpoint_id!r}")
    entry = entries[checkpoint_id]
    if entry["validation_status"] == "unavailable":
        raise CheckpointError(f"checkpoint {checkpoint_id!r} is unavailable")
    path = Path(entry["path"])
    if not path.is_file():
        raise CheckpointError(f"checkpoint does not exist: {path}")
    reference = stage["inputs"][data_role]
    alias = reference.split(".", 1)[0]
    if alias in datasets:
        checkpoint_map = _read_label_map(Path(entry["label_map"]))
        if checkpoint_map != datasets[alias].label_map:
            raise CheckpointError(
                f"checkpoint {checkpoint_id!r} label map differs from {reference}"
            )
    if entry["validation_status"] == "verified":
        pipeline_id = stage["transforms"][data_role]
        pipeline_hash = sha256_json(experiment["transform"]["pipelines"][pipeline_id])
        if entry["transform"] not in {pipeline_id, pipeline_hash}:
            raise CheckpointError(
                f"checkpoint {checkpoint_id!r} preprocessing differs from "
                f"pipeline {pipeline_id!r}"
            )


def validate_experiment(resolved: ResolvedManifest) -> dict[str, Any]:
    datasets = _materialize(resolved)
    finalized = _finalize_resolved(resolved, datasets)
    return {
        "manifest_hash": finalized.manifest_hash,
        "datasets": {alias: item.summary for alias, item in datasets.items()},
        "stages": [stage["id"] for stage in resolved.data["stages"]],
    }


def _materialized_run(
    resolved: ResolvedManifest, *, command: str, final_status: str
) -> Path:
    datasets = _materialize(resolved)
    resolved = _finalize_resolved(resolved, datasets)
    artifacts = RunArtifacts.create(resolved, command)
    try:
        artifacts.status("running")
        _write_materialized(resolved, datasets, artifacts)
        artifacts.status(final_status)
        return artifacts.path
    except BaseException as exc:
        artifacts.status("failed", error=exc)
        raise


def materialize_experiment(resolved: ResolvedManifest) -> Path:
    return _materialized_run(resolved, command="materialize", final_status="completed")


def prepare_experiment(resolved: ResolvedManifest) -> Path:
    return _materialized_run(resolved, command="prepare", final_status="ready")


def _seed_runtime(seed: int, strict: bool) -> None:
    try:
        import torch
    except ImportError as exc:
        raise DependencyError("PyTorch is required for task execution") from exc
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(strict, warn_only=not strict)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _device(value: str):
    import torch

    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise TaskConfigError("CUDA was requested but is unavailable")
    return device


def _autocast(device, precision: str):
    import torch

    if precision == "float32":
        return contextlib.nullcontext()
    if precision not in {"float16", "bfloat16"}:
        raise TaskConfigError("runtime.precision must be float32, float16, or bfloat16")
    dtype = torch.float16 if precision == "float16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _model_components(stage: dict[str, Any], num_classes: int, device):
    try:
        import timm
        import torch
    except ImportError as exc:
        raise DependencyError("timm and PyTorch are required for model tasks") from exc
    model_config = stage["model"]
    configured_classes = model_config.get("num_classes", "auto")
    if configured_classes not in {"auto", num_classes}:
        raise TaskConfigError(
            f"model.num_classes={configured_classes} does not match {num_classes}"
        )
    identifier = model_config["identifier"]
    params = dict(model_config.get("parameters", {}))
    init_type = stage["initialization"]["type"]
    pretrained = init_type == "pretrained"
    task = stage["task"]
    objective = stage["objective"]

    keep_projection = (
        task == "classification_finetuning"
        and stage["initialization"].get("projection_head") == "keep"
    )
    if keep_projection:
        backbone = timm.create_model(
            identifier, pretrained=pretrained, num_classes=0, **params
        )
        feature_dim = int(backbone.num_features)
        projection_dim = int(stage["initialization"]["projection_dim"])
        hidden_dim = int(
            stage["initialization"].get("projection_hidden_dim", feature_dim)
        )

        class FineTuneWithProjection(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.projection = torch.nn.Sequential(
                    torch.nn.Linear(feature_dim, hidden_dim),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(hidden_dim, projection_dim),
                )
                self.classifier = torch.nn.Linear(feature_dim, num_classes)

            def encode(self, images):
                return self.backbone(images)

            def forward(self, images):
                return self.classifier(self.encode(images))

        model = FineTuneWithProjection()
        kind = "finetune_with_projection"
    elif task in {"classification_training", "classification_finetuning"}:
        model = timm.create_model(
            identifier, pretrained=pretrained, num_classes=num_classes, **params
        )
        kind = "classifier"
    else:
        backbone = timm.create_model(
            identifier, pretrained=pretrained, num_classes=0, **params
        )
        feature_dim = int(backbone.num_features)
        use_projection = bool(objective["projection_head"])
        projection_dim = (
            int(objective["projection_dim"]) if use_projection else feature_dim
        )
        hidden_dim = int(objective.get("projection_hidden_dim", feature_dim))

        class ContrastiveModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.projection = (
                    torch.nn.Sequential(
                        torch.nn.Linear(feature_dim, hidden_dim),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Linear(hidden_dim, projection_dim),
                    )
                    if use_projection
                    else torch.nn.Identity()
                )
                self.classifier = (
                    torch.nn.Linear(feature_dim, num_classes)
                    if task == "joint_supcon_training"
                    else None
                )

            def encode(self, images):
                return self.backbone(images)

            def forward(self, images):
                features = self.encode(images)
                projection = torch.nn.functional.normalize(
                    self.projection(features), dim=1
                )
                logits = (
                    self.classifier(features) if self.classifier is not None else None
                )
                return logits, projection

        model = ContrastiveModel()
        kind = "joint_supcon" if task == "joint_supcon_training" else "supcon"
    return model.to(device), {
        "kind": kind,
        "identifier": identifier,
        "parameters": params,
        "num_classes": num_classes,
        "objective": objective,
        "initialization": {
            key: stage["initialization"][key]
            for key in (
                "projection_head",
                "projection_dim",
                "projection_hidden_dim",
            )
            if key in stage["initialization"]
        },
    }


def _state_dict_from_file(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise DependencyError("PyTorch is required for checkpoints") from exc
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise DependencyError(
                "safetensors is required for this checkpoint"
            ) from exc
        return load_file(str(path)), {}
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointError(f"cannot load checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointError(f"checkpoint {path} is not an object")
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(payload.get(key), dict):
            return payload[key], payload
    if payload and all(hasattr(value, "shape") for value in payload.values()):
        return payload, {}
    raise CheckpointError(f"checkpoint {path} has no recognized state dictionary")


def _clean_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    prefixes = ("module.", "model.")
    cleaned = {}
    for key, value in state.items():
        while any(key.startswith(prefix) for prefix in prefixes):
            key = key.split(".", 1)[1]
        cleaned[key] = value
    return cleaned


def _initial_checkpoint(
    stage: dict[str, Any], context: ExperimentContext
) -> tuple[Path | None, dict[str, Any] | None]:
    initialization = stage["initialization"]
    kind = initialization["type"]
    if kind in {"random", "pretrained"}:
        return None, None
    if kind == "path":
        path = Path(initialization["checkpoint"])
        return path, None
    if kind == "catalog":
        catalog = context.experiment.get("checkpoint_catalog", {}).get(
            "checkpoints", {}
        )
        checkpoint_id = initialization["checkpoint"]
        if checkpoint_id not in catalog:
            raise CheckpointError(f"unknown checkpoint catalog entry {checkpoint_id!r}")
        entry = catalog[checkpoint_id]
        if entry["validation_status"] == "unavailable":
            raise CheckpointError(f"checkpoint {checkpoint_id!r} is unavailable")
        return Path(entry["path"]), entry
    if kind == "stage":
        source = initialization["source_stage"]
        try:
            return context.stage_artifacts[source]["best_checkpoint"], None
        except KeyError as exc:
            raise CheckpointError(
                f"stage {stage['id']}: source stage {source!r} has no best checkpoint"
            ) from exc
    raise TaskConfigError(f"stage {stage['id']}: unknown initialization type {kind!r}")


def _load_initial_weights(
    model,
    stage: dict[str, Any],
    context: ExperimentContext,
    label_map: dict[str, int],
) -> dict[str, Any]:
    path, catalog = _initial_checkpoint(stage, context)
    if path is None:
        return {"source": stage["initialization"]["type"]}
    if not path.is_file():
        raise CheckpointError(f"checkpoint does not exist: {path}")
    if catalog:
        if int(catalog["num_classes"]) != len(label_map):
            raise CheckpointError(
                f"checkpoint class count {catalog['num_classes']} != {len(label_map)}"
            )
        if catalog.get("label_map"):
            raw = json.loads(Path(catalog["label_map"]).read_text(encoding="utf-8"))
            labels = raw.get("labels", raw)
            if isinstance(labels, list):
                checkpoint_map = {
                    str(item["label"] if isinstance(item, dict) else item): index
                    for index, item in enumerate(labels)
                }
            else:
                checkpoint_map = {str(key): int(value) for key, value in labels.items()}
            if checkpoint_map != label_map:
                raise CheckpointError("checkpoint label map does not match experiment")
    state, payload = _state_dict_from_file(path)
    strict = bool(stage["initialization"].get("strict", True))
    if not strict and stage["task"] != "classification_finetuning":
        raise CheckpointError(
            "non-strict loading is allowed only for classification_finetuning"
        )
    cleaned = _clean_state_dict(state)
    if not strict and not hasattr(model, "projection"):
        backbone_state = {
            key.removeprefix("backbone."): value
            for key, value in cleaned.items()
            if key.startswith("backbone.")
        }
        if backbone_state:
            cleaned = backbone_state
    try:
        missing, unexpected = model.load_state_dict(cleaned, strict=strict)
    except RuntimeError as exc:
        raise CheckpointError(f"checkpoint {path} is incompatible: {exc}") from exc
    if not strict and len(unexpected) == len(cleaned):
        raise CheckpointError(f"checkpoint {path} did not load any encoder parameters")
    return {
        "source": str(path),
        "sha256": sha256_file(path),
        "strict": strict,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "payload": payload,
        "catalog_status": catalog.get("validation_status") if catalog else None,
    }


def _optimizer(model, config: dict[str, Any]):
    import torch

    kind = config["type"]
    kwargs = {
        "lr": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
    }
    if kind == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if kind == "sgd":
        return torch.optim.SGD(
            model.parameters(), momentum=float(config.get("momentum", 0.0)), **kwargs
        )
    raise TaskConfigError(f"unknown optimizer {kind!r}")


def _scheduler(optimizer, config: dict[str, Any], epochs: int):
    import torch

    kind = config["type"]
    if kind == "none":
        return None
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=float(config.get("eta_min", 0.0))
        )
    if kind == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(config["step_size"]),
            gamma=float(config["gamma"]),
        )
    raise TaskConfigError(f"unknown scheduler {kind!r}")


def _supcon_loss(features, targets, temperature: float):
    import torch

    if features.ndim != 3 or features.shape[1] < 2:
        raise TaskConfigError("SupCon features must be [batch, views>=2, dim]")
    batch, views, _ = features.shape
    flattened = torch.cat([features[:, index] for index in range(views)], dim=0)
    labels = targets.repeat(views)
    logits = flattened @ flattened.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    identity = torch.eye(batch * views, dtype=torch.bool, device=features.device)
    positives = labels[:, None].eq(labels[None, :]) & ~identity
    denominator = torch.exp(logits).masked_fill(identity, 0).sum(dim=1).clamp_min(1e-12)
    log_prob = logits - denominator.log()[:, None]
    counts = positives.sum(dim=1)
    if (counts == 0).any():
        raise TaskConfigError("SupCon batch contains an anchor without a positive")
    return -((log_prob * positives).sum(dim=1) / counts).mean()


def _classification_summary(targets: list[int], predictions: list[int], count: int):
    from sklearn.metrics import accuracy_score, f1_score

    labels = list(range(count))
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(
                targets, predictions, labels=labels, average="macro", zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(
                targets, predictions, labels=labels, average="weighted", zero_division=0
            )
        ),
    }


def _run_epoch(
    model,
    loader,
    stage: dict[str, Any],
    device,
    precision: str,
    optimizer=None,
) -> dict[str, Any]:
    import torch

    training = optimizer is not None
    model.train(training)
    objective = stage["objective"]
    kind = stage["task"]
    total_loss = 0.0
    total = 0
    targets_all: list[int] = []
    predictions_all: list[int] = []
    class_count = 0
    accumulation = int(stage.get("budget", {}).get("gradient_accumulation", 1))
    if accumulation <= 0:
        raise TaskConfigError("gradient_accumulation must be positive")
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, (images, targets, _) in enumerate(loader):
        targets = targets.to(device)
        with torch.set_grad_enabled(training), _autocast(device, precision):
            if kind in {"classification_training", "classification_finetuning"}:
                images = images.to(device)
                logits = model(images)
                loss = torch.nn.functional.cross_entropy(
                    logits,
                    targets,
                    label_smoothing=float(objective.get("label_smoothing", 0.0)),
                )
            else:
                image_views = [image.to(device) for image in images]
                outputs = [model(image) for image in image_views]
                projections = torch.stack([item[1] for item in outputs], dim=1)
                contrastive = _supcon_loss(
                    projections, targets, float(objective["temperature"])
                )
                if kind == "joint_supcon_training":
                    logits_by_view = [item[0] for item in outputs]
                    selected = (
                        logits_by_view[:1]
                        if objective["classification_views"] == "first"
                        else logits_by_view
                    )
                    classification_losses = [
                        torch.nn.functional.cross_entropy(logits, targets)
                        for logits in selected
                    ]
                    classification = torch.stack(classification_losses)
                    classification = (
                        classification.mean()
                        if objective["classification_reduction"] == "mean"
                        else classification.sum()
                    )
                    loss = (
                        float(objective["classification_weight"]) * classification
                        + float(objective["contrastive_weight"]) * contrastive
                    )
                    logits = logits_by_view[0]
                else:
                    loss = contrastive
                    logits = None
            if training:
                (loss / accumulation).backward()
                if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(
                    loader
                ):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        count = len(targets)
        total_loss += float(loss.detach()) * count
        total += count
        if logits is not None:
            class_count = int(logits.shape[1])
            targets_all.extend(targets.detach().cpu().tolist())
            predictions_all.extend(logits.argmax(dim=1).detach().cpu().tolist())
    result = {"loss": total_loss / max(total, 1)}
    if targets_all:
        result.update(
            _classification_summary(targets_all, predictions_all, class_count)
        )
    return result


def _checkpoint_payload(
    context: ExperimentContext,
    stage: dict[str, Any],
    model,
    model_descriptor: dict[str, Any],
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    selection_value: float,
    label_map: dict[str, int],
    train_reference: str,
    validation_reference: str,
    transform_hashes: dict[str, str],
) -> dict[str, Any]:
    import torch

    train_alias, train_view = train_reference.split(".", 1)
    valid_alias, valid_view = validation_reference.split(".", 1)
    return {
        "checkpoint_schema_version": 1,
        "model_identifier": model_descriptor["identifier"],
        "model_descriptor": model_descriptor,
        "model_configuration_hash": sha256_json(model_descriptor),
        "model_state_dict": model.state_dict(),
        "label_map": label_map,
        "label_map_hash": sha256_json(
            [{"index": index, "label": label} for label, index in label_map.items()]
        ),
        "train_view_hash": context.datasets[train_alias].hashes[f"view:{train_view}"],
        "validation_view_hash": context.datasets[valid_alias].hashes[
            f"view:{valid_view}"
        ],
        "transform_hashes": transform_hashes,
        "objective": stage["objective"],
        "optimizer": stage["optimizer"],
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler": stage["scheduler"],
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "global_step": global_step,
        "completed_epoch": epoch,
        "planned_epoch": int(stage["budget"]["epochs"]),
        "selection_metric": stage["checkpoint"]["selection_metric"],
        "selection_value": selection_value,
        "source_code_commit": context.artifacts.git["commit"],
        "runtime_precision": context.experiment["runtime"]["precision"],
        "run_manifest_hash": context.resolved.manifest_hash,
        "sampler_state": {
            "epoch": epoch,
            "seed": derived_seed(context.seed, stage["id"], epoch),
        },
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }


def _save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    import torch

    if path.suffix != ".pt":
        raise TaskConfigError(
            "rich resume checkpoints require checkpoint.save_format=pt"
        )
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _training_stage(
    stage: dict[str, Any], context: ExperimentContext
) -> dict[str, Path]:
    stage_dir = context.artifacts.path / "stages" / stage["id"]
    checkpoint_dir = stage_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_frame, train_reference = context.dataset_input(stage, "train")
    validation_frame, validation_reference = context.dataset_input(stage, "validation")
    label_map = context.label_map_for(train_reference)
    if set(validation_frame["label"]) - set(label_map):
        raise DataError(f"stage {stage['id']}: validation contains unknown labels")
    runtime = context.experiment["runtime"]
    device = _device(runtime["device"])
    _seed_runtime(
        derived_seed(context.seed, stage["id"], "model_initialization"),
        context.experiment["determinism"] == "strict",
    )
    model, descriptor = _model_components(stage, len(label_map), device)
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    resume_payload: dict[str, Any] | None = None
    if context.resuming and last_path.is_file():
        _, resume_payload = _state_dict_from_file(last_path)
        if resume_payload.get("run_manifest_hash") != context.resolved.manifest_hash:
            raise CheckpointError("resume checkpoint manifest hash differs")
        if resume_payload.get("label_map") != label_map:
            raise CheckpointError("resume checkpoint label map differs")
        model.load_state_dict(resume_payload["model_state_dict"], strict=True)
        initialization = {"source": str(last_path), "resume": True}
    else:
        initialization = _load_initial_weights(model, stage, context, label_map)
    if stage["initialization"].get("freeze_encoder"):
        backbone = getattr(model, "backbone", None)
        if backbone is not None:
            for parameter in backbone.parameters():
                parameter.requires_grad = False
        elif hasattr(model, "get_classifier"):
            for parameter in model.parameters():
                parameter.requires_grad = False
            classifier = model.get_classifier()
            if not hasattr(classifier, "parameters"):
                raise TaskConfigError(
                    f"stage {stage['id']}: classifier head cannot be unfrozen"
                )
            for parameter in classifier.parameters():
                parameter.requires_grad = True
        else:
            raise TaskConfigError(
                f"stage {stage['id']}: model has no encoder to freeze"
            )
    optimizer = _optimizer(model, stage["optimizer"])
    epochs = int(stage["budget"]["epochs"])
    if epochs <= 0:
        raise TaskConfigError(f"stage {stage['id']}: epochs must be positive")
    scheduler = _scheduler(optimizer, stage["scheduler"], epochs)
    if resume_payload:
        import torch

        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        if scheduler and resume_payload.get("scheduler_state_dict"):
            scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        rng = resume_payload.get("rng_state", {})
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("cuda"):
                torch.cuda.set_rng_state_all(rng["cuda"])
    train_transform, train_transform_hash = build_pipeline(
        context.experiment["transform"],
        stage["transforms"]["train"],
        evaluation=False,
    )
    eval_transform, eval_transform_hash = build_pipeline(
        context.experiment["transform"],
        stage["transforms"]["validation"],
        evaluation=True,
    )
    if resume_payload:
        train_alias, train_view = train_reference.split(".", 1)
        valid_alias, valid_view = validation_reference.split(".", 1)
        expected = {
            "model_configuration_hash": sha256_json(descriptor),
            "train_view_hash": context.datasets[train_alias].hashes[
                f"view:{train_view}"
            ],
            "validation_view_hash": context.datasets[valid_alias].hashes[
                f"view:{valid_view}"
            ],
            "transform_hashes": {
                "train": train_transform_hash,
                "validation": eval_transform_hash,
            },
            "objective": stage["objective"],
            "runtime_precision": runtime["precision"],
            "checkpoint_schema_version": 1,
        }
        mismatches = [
            key for key, value in expected.items() if resume_payload.get(key) != value
        ]
        if mismatches:
            raise CheckpointError(
                f"resume checkpoint compatibility failed: {mismatches}"
            )
    views = (
        int(stage["objective"]["views"])
        if stage["task"] in {"joint_supcon_training", "supcon_pretraining"}
        else 1
    )
    if views < 2 and stage["task"] in {"joint_supcon_training", "supcon_pretraining"}:
        raise TaskConfigError(f"stage {stage['id']}: SupCon views must be >= 2")
    precision = runtime["precision"]
    metrics_path = stage_dir / "metrics.csv"
    history = (
        pd.read_csv(metrics_path).to_dict("records")
        if resume_payload and metrics_path.is_file()
        else []
    )
    best_value = None
    if resume_payload and best_path.is_file():
        _, best_payload = _state_dict_from_file(best_path)
        best_value = float(best_payload["selection_value"])
    global_step = int(resume_payload.get("global_step", 0)) if resume_payload else 0
    start_epoch = (
        int(resume_payload.get("completed_epoch", 0)) + 1 if resume_payload else 1
    )
    selection_metric = stage["checkpoint"]["selection_metric"]
    mode = stage["checkpoint"]["mode"]
    if mode not in {"min", "max"}:
        raise TaskConfigError(
            f"stage {stage['id']}: checkpoint.mode must be min or max"
        )
    for epoch in range(start_epoch, epochs + 1):
        train_loader = build_loader(
            train_frame,
            label_map,
            train_transform,
            stage["loaders"]["train"],
            base_seed=context.seed,
            stage_id=stage["id"],
            epoch=epoch,
            evaluation=False,
            views=views,
        )
        validation_loader = build_loader(
            validation_frame,
            label_map,
            eval_transform,
            stage["loaders"]["validation"],
            base_seed=context.seed,
            stage_id=stage["id"],
            epoch=epoch,
            evaluation=True,
            views=views,
        )
        train_metrics = _run_epoch(
            model, train_loader, stage, device, precision, optimizer
        )
        validation_metrics = _run_epoch(
            model, validation_loader, stage, device, precision
        )
        accumulation = int(stage["budget"].get("gradient_accumulation", 1))
        global_step += (len(train_loader) + accumulation - 1) // accumulation
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        if selection_metric not in row:
            raise TaskConfigError(
                f"stage {stage['id']}: selection metric {selection_metric!r} "
                f"is unavailable; produced={sorted(row)}"
            )
        selection_value = float(row[selection_metric])
        payload = _checkpoint_payload(
            context,
            stage,
            model,
            descriptor,
            optimizer,
            scheduler,
            epoch,
            global_step,
            selection_value,
            label_map,
            train_reference,
            validation_reference,
            {"train": train_transform_hash, "validation": eval_transform_hash},
        )
        payload["initialization"] = initialization
        _save_checkpoint(payload, last_path)
        improved = best_value is None or (
            selection_value < best_value
            if mode == "min"
            else selection_value > best_value
        )
        if improved:
            best_value = selection_value
            shutil.copy2(last_path, best_path)
        if scheduler:
            scheduler.step()
    pd.DataFrame(history).to_csv(stage_dir / "metrics.csv", index=False)
    _write_json(
        stage_dir / "metadata.json",
        {
            "stage": stage["id"],
            "task": stage["task"],
            "best_value": best_value,
            "selection_metric": selection_metric,
            "initialization": initialization,
        },
    )
    return {
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
        "metrics": stage_dir / "metrics.csv",
    }


def _checkpoint_for_stage(stage: dict[str, Any], context: ExperimentContext) -> Path:
    if "checkpoint" in stage["inputs"]:
        return context.artifact_input(stage, "checkpoint")
    initialization = stage.get("initialization")
    if not initialization:
        raise CheckpointError(f"stage {stage['id']}: checkpoint input is required")
    path, _ = _initial_checkpoint(stage, context)
    if path is None:
        raise CheckpointError(f"stage {stage['id']}: checkpoint input is required")
    return path


def _read_label_map(path: Path) -> dict[str, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw = raw.get("labels", raw)
    if isinstance(raw, list):
        if raw and isinstance(raw[0], dict):
            return {
                str(item["label"]): int(item.get("index", index))
                for index, item in enumerate(raw)
            }
        return {str(label): index for index, label in enumerate(raw)}
    if isinstance(raw, dict):
        return {str(label): int(index) for label, index in raw.items()}
    raise CheckpointError(f"invalid label map file: {path}")


def _catalog_entry_for_stage(
    stage: dict[str, Any], context: ExperimentContext
) -> dict[str, Any] | None:
    initialization = stage.get("initialization", {})
    if initialization.get("type") != "catalog":
        return None
    checkpoint_id = initialization["checkpoint"]
    entries = context.experiment.get("checkpoint_catalog", {}).get("checkpoints", {})
    if checkpoint_id not in entries:
        raise CheckpointError(f"unknown checkpoint catalog entry {checkpoint_id!r}")
    return entries[checkpoint_id]


def _model_from_checkpoint(
    path: Path,
    device,
    *,
    stage: dict[str, Any] | None = None,
    context: ExperimentContext | None = None,
    expected_label_map: dict[str, int] | None = None,
):
    state, payload = _state_dict_from_file(path)
    descriptor = payload.get("model_descriptor")
    label_map = payload.get("label_map")
    catalog = (
        _catalog_entry_for_stage(stage, context)
        if stage is not None and context is not None
        else None
    )
    if (not descriptor or not label_map) and catalog:
        if catalog["validation_status"] == "unavailable":
            raise CheckpointError(
                f"checkpoint {stage['initialization']['checkpoint']!r} is unavailable"
            )
        label_map = _read_label_map(Path(catalog["label_map"]))
        model_config = catalog["model"]
        if isinstance(model_config, str):
            model_config = {"identifier": model_config, "parameters": {}}
        if not isinstance(model_config, dict) or "identifier" not in model_config:
            raise CheckpointError("legacy catalog model must name an identifier")
        adapter = catalog["adapter"]
        kind = "supcon" if "supcon" in adapter else "classifier"
        if kind == "supcon" and "projection_dim" not in model_config:
            raise CheckpointError(
                "legacy SupCon model requires model.projection_dim in the catalog"
            )
        objective = (
            {
                "type": "supervised_contrastive",
                "views": 2,
                "temperature": float(model_config.get("temperature", 0.07)),
                "projection_head": True,
                "projection_dim": int(model_config["projection_dim"]),
                "projection_hidden_dim": int(
                    model_config.get(
                        "projection_hidden_dim", model_config["projection_dim"]
                    )
                ),
            }
            if kind == "supcon"
            else {"type": "cross_entropy", "label_smoothing": 0.0}
        )
        descriptor = {
            "kind": kind,
            "identifier": model_config["identifier"],
            "parameters": model_config.get("parameters", {}),
            "num_classes": int(catalog["num_classes"]),
            "objective": objective,
        }
        payload = {
            **payload,
            "model_descriptor": descriptor,
            "label_map": label_map,
            "legacy_catalog": catalog,
            "preprocessing_status": catalog["validation_status"],
        }
    if not descriptor or not label_map:
        raise CheckpointError(
            f"checkpoint {path} lacks model_descriptor or label_map; "
            "register it in checkpoint_catalog and bind initialization.type=catalog"
        )
    label_map = {str(key): int(value) for key, value in label_map.items()}
    if expected_label_map is not None and label_map != expected_label_map:
        raise CheckpointError("checkpoint and dataset label maps differ")
    fake_stage = {
        "task": (
            "classification_training"
            if descriptor["kind"] == "classifier"
            else "classification_finetuning"
            if descriptor["kind"] == "finetune_with_projection"
            else "joint_supcon_training"
            if descriptor["kind"] == "joint_supcon"
            else "supcon_pretraining"
        ),
        "model": {
            "identifier": descriptor["identifier"],
            "parameters": descriptor.get("parameters", {}),
            "num_classes": "auto",
        },
        "initialization": {
            "type": "random",
            **descriptor.get("initialization", {}),
        },
        "objective": descriptor["objective"],
    }
    model, _ = _model_components(fake_stage, len(label_map), device)
    try:
        model.load_state_dict(_clean_state_dict(state), strict=True)
    except RuntimeError as exc:
        raise CheckpointError(f"checkpoint {path} strict load failed: {exc}") from exc
    model.eval()
    return model, label_map, payload


def _encode(model, images):
    import torch

    if hasattr(model, "encode"):
        return model.encode(images)
    if hasattr(model, "forward_features"):
        values = model.forward_features(images)
        if hasattr(model, "forward_head"):
            return model.forward_head(values, pre_logits=True)
        if values.ndim > 2:
            return values.flatten(1)
        return values
    output = model(images)
    if isinstance(output, tuple):
        output = output[0]
    return output if output.ndim == 2 else torch.flatten(output, 1)


def _embedding_stage(
    stage: dict[str, Any], context: ExperimentContext
) -> dict[str, Path]:
    import torch

    role = next(
        (
            name
            for name in ("data", "input", "query", "test")
            if name in stage["inputs"]
        ),
        None,
    )
    if role is None:
        raise TaskConfigError(f"stage {stage['id']}: data input is required")
    frame, reference = context.dataset_input(stage, role)
    label_map = context.label_map_for(reference)
    checkpoint = _checkpoint_for_stage(stage, context)
    device = _device(context.experiment["runtime"]["device"])
    model, checkpoint_labels, payload = _model_from_checkpoint(
        checkpoint,
        device,
        stage=stage,
        context=context,
        expected_label_map=label_map,
    )
    pipeline_id = stage["transforms"][role]
    transform, transform_hash = build_pipeline(
        context.experiment["transform"], pipeline_id, evaluation=True
    )
    loader = build_loader(
        frame,
        label_map,
        transform,
        stage["loaders"][role],
        base_seed=context.seed,
        stage_id=stage["id"],
        epoch=0,
        evaluation=True,
    )
    vectors: list[np.ndarray] = []
    indices: list[int] = []
    with torch.inference_mode():
        for images, _, batch_indices in loader:
            with _autocast(device, context.experiment["runtime"]["precision"]):
                values = _encode(model, images.to(device))
            vectors.append(values.detach().float().cpu().numpy())
            indices.extend(batch_indices.tolist())
    array = np.concatenate(vectors)
    metadata = frame.iloc[indices].reset_index(drop=True)
    stage_dir = context.artifacts.path / "stages" / stage["id"]
    stage_dir.mkdir(parents=True, exist_ok=True)
    vector_path = stage_dir / "embeddings.npz"
    np.savez_compressed(
        vector_path,
        sample_id=np.asarray(metadata["sample_id"].astype(str).tolist(), dtype=str),
        vectors=array.astype(np.float32),
    )
    metadata_path = stage_dir / "embedding_metadata.parquet"
    write_parquet(metadata, metadata_path)
    _write_json(
        stage_dir / "metadata.json",
        {
            "checkpoint_hash": sha256_file(checkpoint),
            "checkpoint_manifest_hash": payload.get("run_manifest_hash"),
            "transform_hash": transform_hash,
            "embedding_hash": sha256_file(vector_path),
            "samples": len(metadata),
            "dimensions": int(array.shape[1]),
            "preprocessing_status": payload.get("preprocessing_status", "verified"),
        },
    )
    return {
        "embeddings": vector_path,
        "embedding_metadata": metadata_path,
        "metadata": stage_dir / "metadata.json",
    }


def _classification_evaluation_stage(
    stage: dict[str, Any], context: ExperimentContext
) -> dict[str, Path]:
    import torch
    from sklearn.metrics import (
        confusion_matrix,
        precision_recall_fscore_support,
    )

    role = "test" if "test" in stage["inputs"] else "query"
    frame, reference = context.dataset_input(stage, role)
    label_map = context.label_map_for(reference)
    labels = [label for label, _ in sorted(label_map.items(), key=lambda item: item[1])]
    checkpoint = _checkpoint_for_stage(stage, context)
    device = _device(context.experiment["runtime"]["device"])
    model, checkpoint_labels, checkpoint_payload = _model_from_checkpoint(
        checkpoint,
        device,
        stage=stage,
        context=context,
        expected_label_map=label_map,
    )
    transform, transform_hash = build_pipeline(
        context.experiment["transform"], stage["transforms"][role], evaluation=True
    )
    loader = build_loader(
        frame,
        label_map,
        transform,
        stage["loaders"][role],
        base_seed=context.seed,
        stage_id=stage["id"],
        epoch=0,
        evaluation=True,
    )
    target_indices: list[int] = []
    predicted_indices: list[int] = []
    confidences: list[float] = []
    row_indices: list[int] = []
    with torch.inference_mode():
        for images, targets, indices in loader:
            with _autocast(device, context.experiment["runtime"]["precision"]):
                output = model(images.to(device))
                logits = output[0] if isinstance(output, tuple) else output
                if logits is None:
                    raise TaskConfigError(
                        f"stage {stage['id']}: checkpoint has no classifier head"
                    )
            probability = logits.float().softmax(dim=1)
            confidence, prediction = probability.max(dim=1)
            target_indices.extend(targets.tolist())
            predicted_indices.extend(prediction.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
            row_indices.extend(indices.tolist())
    summary = _classification_summary(target_indices, predicted_indices, len(label_map))
    precision, recall, f1, support = precision_recall_fscore_support(
        target_indices,
        predicted_indices,
        labels=list(range(len(label_map))),
        zero_division=0,
    )
    matrix = confusion_matrix(
        target_indices, predicted_indices, labels=list(range(len(label_map)))
    )
    predictions = frame.iloc[row_indices].reset_index(drop=True).copy()
    predictions["target_index"] = target_indices
    predictions["target_label"] = [labels[index] for index in target_indices]
    predictions["predicted_index"] = predicted_indices
    predictions["predicted_label"] = [labels[index] for index in predicted_indices]
    predictions["confidence"] = confidences
    predictions["stage_id"] = stage["id"]
    predictions["checkpoint_hash"] = sha256_file(checkpoint)
    per_class = pd.DataFrame(
        {
            "label": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
    stage_dir = context.artifacts.path / "stages" / stage["id"]
    stage_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = stage_dir / "predictions.parquet"
    write_parquet(predictions, predictions_path)
    per_class.to_csv(stage_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        stage_dir / "confusion_matrix.csv"
    )
    summary.update(
        {
            "samples": len(predictions),
            "checkpoint_hash": sha256_file(checkpoint),
            "transform_hash": transform_hash,
            "metric_version": 1,
            "preprocessing_status": checkpoint_payload.get(
                "preprocessing_status", "verified"
            ),
            "official_comparison_eligible": checkpoint_payload.get(
                "preprocessing_status", "verified"
            )
            == "verified",
        }
    )
    _write_json(stage_dir / "metrics.json", summary)
    return {
        "predictions": predictions_path,
        "metrics": stage_dir / "metrics.json",
        "per_class_metrics": stage_dir / "per_class_metrics.csv",
        "confusion_matrix": stage_dir / "confusion_matrix.csv",
    }


def _load_embeddings(
    context: ExperimentContext, stage: dict[str, Any], role: str
) -> tuple[np.ndarray, pd.DataFrame, Path]:
    vector_path = context.artifact_input(stage, role)
    reference = stage["inputs"][role]
    producer, _ = reference.split(".", 1)
    try:
        metadata_path = context.stage_artifacts[producer]["embedding_metadata"]
    except KeyError as exc:
        raise TaskConfigError(
            f"stage {stage['id']} input {role}: producer lacks embedding metadata"
        ) from exc
    data = np.load(vector_path, allow_pickle=False)
    vectors = np.asarray(data["vectors"], dtype=np.float32)
    sample_ids = data["sample_id"].astype(str)
    metadata = pd.read_parquet(metadata_path)
    if metadata["sample_id"].astype(str).tolist() != sample_ids.tolist():
        raise DataError(
            f"stage {stage['id']} input {role}: embedding metadata mismatch"
        )
    return vectors, metadata, vector_path


def _embedding_checkpoint_hash(
    context: ExperimentContext, stage: dict[str, Any], role: str
) -> str:
    producer = stage["inputs"][role].split(".", 1)[0]
    metadata_path = context.stage_artifacts.get(producer, {}).get("metadata")
    if metadata_path and metadata_path.is_file():
        return str(
            json.loads(metadata_path.read_text(encoding="utf-8")).get(
                "checkpoint_hash", ""
            )
        )
    return ""


def _distance(query: np.ndarray, reference: np.ndarray, kind: str) -> np.ndarray:
    if kind == "euclidean":
        return ((query[:, None] - reference[None, :]) ** 2).sum(axis=2)
    if kind == "cosine":
        query_norm = query / np.linalg.norm(query, axis=1, keepdims=True).clip(1e-12)
        ref_norm = reference / np.linalg.norm(reference, axis=1, keepdims=True).clip(
            1e-12
        )
        return 1.0 - query_norm @ ref_norm.T
    raise TaskConfigError(f"unknown distance {kind!r}")


def _prototype_stage(
    stage: dict[str, Any], context: ExperimentContext
) -> dict[str, Path]:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    support, support_meta, support_path = _load_embeddings(context, stage, "prototype")
    query, query_meta, query_path = _load_embeddings(context, stage, "query")
    overlap = set(support_meta.sample_id) & set(query_meta.sample_id)
    content_overlap = set(support_meta.content_hash) & set(query_meta.content_hash)
    if overlap or content_overlap:
        raise DataError(
            f"stage {stage['id']}: prototype/query leakage "
            f"sample={len(overlap)}, content={len(content_overlap)}"
        )
    labels = sorted(set(support_meta.label.astype(str)))
    if set(query_meta.label.astype(str)) - set(labels):
        raise DataError(f"stage {stage['id']}: query lacks prototype coverage")
    reducer = stage["prototype"]["reducer"]
    prototypes = []
    member_indices: list[int] = []
    for label in labels:
        indices = np.flatnonzero(support_meta.label.astype(str).to_numpy() == label)
        values = support[indices]
        if reducer == "mean":
            prototypes.append(values.mean(axis=0))
            member_indices.extend(indices.tolist())
        elif reducer == "medoid":
            distances = _distance(values, values, "euclidean")
            medoid = int(distances.sum(axis=1).argmin())
            prototypes.append(values[medoid])
            member_indices.append(int(indices[medoid]))
        else:
            raise TaskConfigError(f"unknown prototype reducer {reducer!r}")
    prototypes_array = np.stack(prototypes)
    if stage["prototype"].get("normalize"):
        prototypes_array /= np.linalg.norm(
            prototypes_array, axis=1, keepdims=True
        ).clip(1e-12)
        query = query / np.linalg.norm(query, axis=1, keepdims=True).clip(1e-12)
    distances = _distance(query, prototypes_array, stage["prototype"]["distance"])
    predicted = distances.argmin(axis=1)
    targets = np.asarray([labels.index(str(value)) for value in query_meta.label])
    metrics = _classification_summary(targets.tolist(), predicted.tolist(), len(labels))
    precision, recall, f1, support_count = precision_recall_fscore_support(
        targets, predicted, labels=list(range(len(labels))), zero_division=0
    )
    stage_dir = context.artifacts.path / "stages" / stage["id"]
    stage_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        stage_dir / "prototypes.npz", labels=labels, vectors=prototypes_array
    )
    membership = support_meta.iloc[member_indices].reset_index(drop=True)
    membership["reducer"] = reducer
    write_parquet(membership, stage_dir / "prototype_membership.parquet")
    predictions = query_meta.copy()
    predictions["target_label"] = query_meta.label.astype(str)
    predictions["target_index"] = targets
    predictions["predicted_label"] = [labels[index] for index in predicted]
    predictions["predicted_index"] = predicted
    predictions["confidence"] = -distances.min(axis=1)
    predictions["stage_id"] = stage["id"]
    predictions["checkpoint_hash"] = _embedding_checkpoint_hash(context, stage, "query")
    write_parquet(predictions, stage_dir / "predictions.parquet")
    pd.DataFrame(
        {
            "label": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support_count,
        }
    ).to_csv(stage_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame(
        confusion_matrix(targets, predicted, labels=list(range(len(labels)))),
        index=labels,
        columns=labels,
    ).to_csv(stage_dir / "confusion_matrix.csv")
    metrics.update(
        {
            "reducer": reducer,
            "distance": stage["prototype"]["distance"],
            "support_embedding_hash": sha256_file(support_path),
            "query_embedding_hash": sha256_file(query_path),
        }
    )
    _write_json(stage_dir / "metrics.json", metrics)
    return {
        "prototypes": stage_dir / "prototypes.npz",
        "prototype_membership": stage_dir / "prototype_membership.parquet",
        "predictions": stage_dir / "predictions.parquet",
        "metrics": stage_dir / "metrics.json",
    }


def _retrieval_stage(
    stage: dict[str, Any], context: ExperimentContext
) -> dict[str, Path]:
    query, query_meta, query_path = _load_embeddings(context, stage, "query")
    gallery, gallery_meta, gallery_path = _load_embeddings(context, stage, "gallery")
    if stage["retrieval"].get("normalize"):
        query /= np.linalg.norm(query, axis=1, keepdims=True).clip(1e-12)
        gallery /= np.linalg.norm(gallery, axis=1, keepdims=True).clip(1e-12)
    distances = _distance(query, gallery, stage["retrieval"]["distance"])
    same_source = query_path == gallery_path
    if same_source:
        np.fill_diagonal(distances, np.inf)
    order = distances.argsort(axis=1)
    query_labels = query_meta.label.astype(str).to_numpy()
    gallery_labels = gallery_meta.label.astype(str).to_numpy()
    metrics: dict[str, Any] = {}
    for raw_k in stage["retrieval"]["k"]:
        k = int(raw_k)
        if k <= 0 or k > len(gallery_labels) - int(same_source):
            raise TaskConfigError(f"stage {stage['id']}: invalid retrieval K={k}")
        top = order[:, :k]
        relevant = gallery_labels[top] == query_labels[:, None]
        metrics[f"recall@{k}"] = float(relevant.any(axis=1).mean())
        precision = np.cumsum(relevant, axis=1) / np.arange(1, k + 1)
        denominators = np.minimum(
            k,
            np.asarray(
                [
                    int((gallery_labels == label).sum()) - int(same_source)
                    for label in query_labels
                ]
            ),
        ).clip(1)
        metrics[f"map@{k}"] = float(
            ((precision * relevant).sum(axis=1) / denominators).mean()
        )
    same = query_labels[:, None] == gallery_labels[None, :]
    if same_source:
        np.fill_diagonal(same, False)
    within = distances[same]
    between = distances[~same & np.isfinite(distances)]
    metrics["within_class_mean_distance"] = (
        float(within.mean()) if within.size else None
    )
    metrics["between_class_mean_distance"] = (
        float(between.mean()) if between.size else None
    )
    metrics.update(
        {
            "distance": stage["retrieval"]["distance"],
            "query_embedding_hash": sha256_file(query_path),
            "gallery_embedding_hash": sha256_file(gallery_path),
        }
    )
    stage_dir = context.artifacts.path / "stages" / stage["id"]
    stage_dir.mkdir(parents=True, exist_ok=True)
    _write_json(stage_dir / "metrics.json", metrics)
    return {"metrics": stage_dir / "metrics.json"}


def _open_set_stage(
    stage: dict[str, Any], context: ExperimentContext
) -> dict[str, Path]:
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

    calibration, calibration_meta, calibration_path = _load_embeddings(
        context, stage, "calibration"
    )
    known, known_meta, known_path = _load_embeddings(context, stage, "known")
    unknown, unknown_meta, unknown_path = _load_embeddings(context, stage, "unknown")
    prototype_path = context.artifact_input(stage, "prototypes")
    prototypes_data = np.load(prototype_path, allow_pickle=False)
    prototypes = prototypes_data["vectors"].astype(np.float32)
    labels = prototypes_data["labels"].astype(str).tolist()
    score_method = stage["open_set"]["score"]

    def scores(values: np.ndarray) -> np.ndarray:
        if score_method == "max_cosine":
            return 1.0 - _distance(values, prototypes, "cosine").min(axis=1)
        if score_method == "negative_euclidean":
            return -_distance(values, prototypes, "euclidean").min(axis=1)
        raise TaskConfigError(f"unknown open-set score {score_method!r}")

    calibration_scores = scores(calibration)
    target_tpr = float(stage["open_set"].get("target_tpr", 0.95))
    if not 0 < target_tpr <= 1:
        raise TaskConfigError("open_set.target_tpr must be in (0, 1]")
    if stage["open_set"]["threshold_method"] != "known_tpr":
        raise TaskConfigError("open_set.threshold_method must be known_tpr")
    threshold = float(np.quantile(calibration_scores, 1.0 - target_tpr))
    known_scores = scores(known)
    unknown_scores = scores(unknown)
    truth_known = np.concatenate(
        [np.ones(len(known_scores)), np.zeros(len(unknown_scores))]
    )
    all_scores = np.concatenate([known_scores, unknown_scores])
    fpr, tpr, thresholds = roc_curve(truth_known, all_scores)
    index95 = int(np.argmin(np.abs(tpr - 0.95)))
    prediction_distance = "cosine" if score_method == "max_cosine" else "euclidean"
    known_distances = _distance(known, prototypes, prediction_distance)
    unknown_distances = _distance(unknown, prototypes, prediction_distance)
    known_predictions = known_distances.argmin(axis=1)
    unknown_predictions = unknown_distances.argmin(axis=1)
    known_targets = np.asarray([labels.index(str(value)) for value in known_meta.label])
    accepted_known = known_scores >= threshold
    accepted_unknown = unknown_scores >= threshold
    correct_known = known_predictions == known_targets
    thresholds_oscr = np.unique(
        np.concatenate(
            [
                np.asarray([np.inf]),
                known_scores,
                unknown_scores,
                np.asarray([-np.inf]),
            ]
        )
    )[::-1]
    oscr_fpr = np.asarray(
        [(unknown_scores >= value).mean() for value in thresholds_oscr]
    )
    oscr_ccr = np.asarray(
        [(correct_known & (known_scores >= value)).mean() for value in thresholds_oscr]
    )
    oscr_order = np.argsort(oscr_fpr)
    metrics = {
        "auroc": float(roc_auc_score(truth_known, all_scores)),
        "aupr": float(average_precision_score(truth_known, all_scores)),
        "fpr@95tpr": float(fpr[index95]),
        "known_accuracy": float((known_predictions == known_targets).mean()),
        "unknown_rejection_rate": float((~accepted_unknown).mean()),
        "oscr": float(
            np.sum(
                np.diff(oscr_fpr[oscr_order])
                * (oscr_ccr[oscr_order][1:] + oscr_ccr[oscr_order][:-1])
                / 2
            )
        ),
        "threshold": threshold,
        "score": score_method,
        "calibration_embedding_hash": sha256_file(calibration_path),
        "known_embedding_hash": sha256_file(known_path),
        "unknown_embedding_hash": sha256_file(unknown_path),
    }
    known_predictions_frame = known_meta.assign(
        target_label=known_meta.label.astype(str),
        target_index=known_targets,
        predicted_label=[
            labels[index] if accepted else "__unknown__"
            for index, accepted in zip(known_predictions, accepted_known, strict=True)
        ],
        predicted_index=[
            int(index) if accepted else -1
            for index, accepted in zip(known_predictions, accepted_known, strict=True)
        ],
        confidence=known_scores,
        known_score=known_scores,
        ground_truth_unknown=False,
        predicted_unknown=~accepted_known,
    )
    unknown_predictions_frame = unknown_meta.assign(
        target_label=unknown_meta.label.astype(str),
        target_index=-1,
        predicted_label=[
            labels[index] if accepted else "__unknown__"
            for index, accepted in zip(
                unknown_predictions, accepted_unknown, strict=True
            )
        ],
        predicted_index=[
            int(index) if accepted else -1
            for index, accepted in zip(
                unknown_predictions, accepted_unknown, strict=True
            )
        ],
        confidence=unknown_scores,
        known_score=unknown_scores,
        ground_truth_unknown=True,
        predicted_unknown=~accepted_unknown,
    )
    predictions = pd.concat(
        [known_predictions_frame, unknown_predictions_frame], ignore_index=True
    )
    predictions["threshold"] = threshold
    predictions["stage_id"] = stage["id"]
    predictions["checkpoint_hash"] = _embedding_checkpoint_hash(context, stage, "known")
    stage_dir = context.artifacts.path / "stages" / stage["id"]
    stage_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(predictions, stage_dir / "predictions.parquet")
    _write_json(stage_dir / "metrics.json", metrics)
    return {
        "predictions": stage_dir / "predictions.parquet",
        "metrics": stage_dir / "metrics.json",
    }


def _visualization_stage(
    stage: dict[str, Any], context: ExperimentContext
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arrays: list[np.ndarray] = []
    metadata_frames: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    for role in stage["inputs"]:
        vectors, metadata, path = _load_embeddings(context, stage, role)
        metadata = metadata.copy()
        metadata["input_role"] = role
        arrays.append(vectors)
        metadata_frames.append(metadata)
        hashes[role] = sha256_file(path)
    values = np.concatenate(arrays)
    metadata = pd.concat(metadata_frames, ignore_index=True)
    config = stage["visualization"]
    max_points = int(config.get("max_points", 0))
    seed = int(config["seed"])
    if max_points and len(values) > max_points:
        order = sorted(
            range(len(metadata)),
            key=lambda index: hashlib.sha256(
                f"{seed}\0{metadata.at[index, 'sample_id']}".encode("utf-8")
            ).hexdigest(),
        )[:max_points]
        values = values[order]
        metadata = metadata.iloc[order].reset_index(drop=True)
    algorithm = config["algorithm"]
    if algorithm == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(
            n_components=2,
            random_state=seed,
            perplexity=float(config.get("perplexity", 30.0)),
            max_iter=int(config.get("iterations", 1000)),
            init="pca",
            learning_rate="auto",
        )
    elif algorithm == "umap":
        try:
            import umap
        except ImportError as exc:
            raise DependencyError("UMAP visualization requires umap-learn") from exc
        reducer = umap.UMAP(
            n_components=2,
            random_state=seed,
            n_neighbors=int(config.get("n_neighbors", 15)),
            min_dist=float(config.get("min_dist", 0.1)),
        )
    else:
        raise TaskConfigError(f"unknown visualization algorithm {algorithm!r}")
    coordinates = reducer.fit_transform(values)
    output = metadata.copy()
    output["x"] = coordinates[:, 0]
    output["y"] = coordinates[:, 1]
    stage_dir = context.artifacts.path / "stages" / stage["id"]
    stage_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(stage_dir / "coordinates.csv", index=False)
    figure, axis = plt.subplots(figsize=(10, 8))
    for label, group in output.groupby("label", sort=True):
        axis.scatter(group.x, group.y, s=10, alpha=0.75, label=label)
    if output["label"].nunique() <= 20:
        axis.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.set_title(f"{algorithm.upper()} — {stage['id']}")
    figure.tight_layout()
    figure.savefig(stage_dir / "visualization.png", dpi=160)
    plt.close(figure)
    _write_json(
        stage_dir / "metadata.json",
        {
            "algorithm": algorithm,
            "parameters": config,
            "seed": seed,
            "input_embedding_hashes": hashes,
            "samples": len(output),
        },
    )
    return {
        "coordinates": stage_dir / "coordinates.csv",
        "image": stage_dir / "visualization.png",
        "metadata": stage_dir / "metadata.json",
    }


HANDLERS: dict[str, Callable[[dict[str, Any], ExperimentContext], dict[str, Path]]] = {
    **{name: _training_stage for name in TRAINING_TASKS},
    "classification_evaluation": _classification_evaluation_stage,
    "embedding_extraction": _embedding_stage,
    "prototype_evaluation": _prototype_stage,
    "retrieval_evaluation": _retrieval_stage,
    "open_set_evaluation": _open_set_stage,
    "embedding_visualization": _visualization_stage,
}


def _persist_stage_artifacts(context: ExperimentContext) -> None:
    _write_json(
        context.artifacts.path / "stage_artifacts.json",
        {
            stage: {name: str(path) for name, path in values.items()}
            for stage, values in context.stage_artifacts.items()
        },
    )


def _stage_log(artifacts: RunArtifacts, stage_id: str, event: str) -> None:
    path = artifacts.path / "logs" / f"{stage_id}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "event": event,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def run_experiment(resolved: ResolvedManifest) -> Path:
    datasets = _materialize(resolved)
    resolved = _finalize_resolved(resolved, datasets)
    artifacts = RunArtifacts.create(resolved, "run")
    context: ExperimentContext | None = None
    try:
        artifacts.status("running")
        _write_materialized(resolved, datasets, artifacts)
        _seed_runtime(
            int(resolved.data["seed"]), resolved.data["determinism"] == "strict"
        )
        context = ExperimentContext(resolved, artifacts, datasets)
        for stage in resolved.data["stages"]:
            artifacts.status("running", stage=stage["id"])
            _stage_log(artifacts, stage["id"], "started")
            try:
                context.stage_artifacts[stage["id"]] = HANDLERS[stage["task"]](
                    stage, context
                )
            except BaseException:
                _stage_log(artifacts, stage["id"], "failed")
                raise
            _stage_log(artifacts, stage["id"], "completed")
            _persist_stage_artifacts(context)
        artifacts.status("completed")
        return artifacts.path
    except KeyboardInterrupt as exc:
        artifacts.status("interrupted", error=exc)
        raise
    except BaseException as exc:
        artifacts.status("failed", error=exc)
        raise


def inspect_checkpoint(
    path: Path,
    catalog_path: Path | None = None,
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise CheckpointError(f"checkpoint does not exist: {path}")
    catalog_entry = None
    if catalog_path:
        resolved = ManifestResolver().resolve(catalog_path)
        if resolved.data["kind"] != "checkpoint_catalog":
            raise CheckpointError("catalog manifest kind must be checkpoint_catalog")
        entries = resolved.data["checkpoints"]
        if checkpoint_id:
            if checkpoint_id not in entries:
                raise CheckpointError(f"catalog has no entry {checkpoint_id!r}")
            catalog_entry = entries[checkpoint_id]
        else:
            matches = [
                entry
                for entry in entries.values()
                if Path(entry["path"]).resolve() == path
            ]
            if len(matches) != 1:
                raise CheckpointError(
                    "catalog lookup by path must match exactly one entry; pass --id"
                )
            catalog_entry = matches[0]
    state, payload = _state_dict_from_file(path)
    metadata = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "model_state_dict",
            "state_dict",
            "model",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "rng_state",
        }
    }
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "state_keys": sorted(state),
        "state_shapes": {
            key: list(getattr(value, "shape", ())) for key, value in state.items()
        },
        "metadata": metadata,
        "catalog": catalog_entry,
    }


def _load_resolved_run(
    run_dir: Path, *, allowed_states: set[str], command: str
) -> tuple[ResolvedManifest, RunArtifacts]:
    run_dir = run_dir.resolve()
    status_path = run_dir / "status.json"
    if not status_path.is_file():
        raise CliError(f"run has no status.json: {run_dir}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") not in allowed_states:
        allowed = ", ".join(sorted(allowed_states))
        raise CliError(f"{command} requires run status: {allowed}")
    data = yaml.safe_load(
        (run_dir / "resolved_manifest.yaml").read_text(encoding="utf-8")
    )
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    sources = tuple(
        json.loads((run_dir / "source_manifests.json").read_text(encoding="utf-8"))
    )
    for source in sources:
        source_path = Path(source["path"])
        if not source_path.is_file() or sha256_file(source_path) != source["sha256"]:
            raise CheckpointError(f"source manifest changed since run: {source_path}")
    resolved = ResolvedManifest(
        data=data,
        root_path=run_dir / "resolved_manifest.yaml",
        source_manifests=sources,
        overrides=tuple(metadata.get("overrides", [])),
        manifest_hash=metadata["manifest_hash"],
    )
    artifacts = RunArtifacts(
        run_dir,
        resolved,
        command,
        _git_info(Path(__file__).resolve().parents[3]),
    )
    return resolved, artifacts


def _existing_run_context(
    resolved: ResolvedManifest, artifacts: RunArtifacts
) -> ExperimentContext:
    datasets = _materialize(resolved)
    stored_hashes = json.loads(
        (artifacts.path / "hashes.json").read_text(encoding="utf-8")
    )
    actual_hashes = {"manifest": resolved.manifest_hash}
    for alias, dataset in datasets.items():
        actual_hashes.update(
            {f"{alias}:{key}": value for key, value in dataset.hashes.items()}
        )
    mismatches = sorted(
        key for key, value in actual_hashes.items() if stored_hashes.get(key) != value
    )
    if mismatches:
        raise CheckpointError(f"run data/view/label hashes changed: {mismatches}")
    _seed_runtime(int(resolved.data["seed"]), resolved.data["determinism"] == "strict")
    previous: dict[str, dict[str, Path]] = {}
    artifact_index = artifacts.path / "stage_artifacts.json"
    if artifact_index.is_file():
        raw = json.loads(artifact_index.read_text(encoding="utf-8"))
        previous = {
            stage: {name: Path(path) for name, path in values.items()}
            for stage, values in raw.items()
        }
        missing = [
            str(path)
            for values in previous.values()
            for path in values.values()
            if not path.exists()
        ]
        if missing:
            raise CheckpointError(f"completed stage artifacts are missing: {missing}")
    return ExperimentContext(
        resolved, artifacts, datasets, stage_artifacts=previous, resuming=True
    )


def run_stage(
    run_dir: Path,
    stage_id: str | None = None,
    *,
    allowed_tasks: set[str] | None = None,
) -> Path:
    resolved, artifacts = _load_resolved_run(
        run_dir,
        allowed_states={"ready", "failed", "interrupted"},
        command="stage",
    )
    candidates = [
        stage
        for stage in resolved.data["stages"]
        if allowed_tasks is None or stage["task"] in allowed_tasks
    ]
    if stage_id is None:
        if len(candidates) != 1:
            choices = ", ".join(stage["id"] for stage in candidates) or "<none>"
            raise CliError(f"choose a stage explicitly; available: {choices}")
        stage = candidates[0]
    else:
        matches = [stage for stage in candidates if stage["id"] == stage_id]
        if not matches:
            choices = ", ".join(stage["id"] for stage in candidates) or "<none>"
            raise CliError(
                f"stage {stage_id!r} is not valid here; available: {choices}"
            )
        stage = matches[0]

    context = _existing_run_context(resolved, artifacts)
    if stage["id"] in context.stage_artifacts:
        raise CliError(
            f"stage {stage['id']!r} is already completed; prepare a new run to repeat it"
        )
    dependencies = set(stage.get("depends_on", []))
    if stage.get("initialization", {}).get("type") == "stage":
        dependencies.add(stage["initialization"]["source_stage"])
    for reference in stage["inputs"].values():
        owner, separator, _ = reference.partition(".")
        if separator and owner not in context.datasets:
            dependencies.add(owner)
    missing = sorted(dependencies - context.stage_artifacts.keys())
    if missing:
        raise TaskConfigError(
            f"stage {stage['id']}: run prerequisite stages first: {missing}"
        )

    try:
        artifacts.status("running", stage=stage["id"])
        _stage_log(artifacts, stage["id"], "started")
        try:
            context.stage_artifacts[stage["id"]] = HANDLERS[stage["task"]](
                stage, context
            )
        except BaseException:
            _stage_log(artifacts, stage["id"], "failed")
            raise
        _stage_log(artifacts, stage["id"], "completed")
        _persist_stage_artifacts(context)
        final_state = (
            "completed"
            if len(context.stage_artifacts) == len(resolved.data["stages"])
            else "ready"
        )
        artifacts.status(final_state)
        return artifacts.path
    except KeyboardInterrupt as exc:
        artifacts.status("interrupted", stage=stage["id"], error=exc)
        raise
    except BaseException as exc:
        artifacts.status("failed", stage=stage["id"], error=exc)
        raise


def resume_experiment(run_dir: Path) -> Path:
    resolved, artifacts = _load_resolved_run(
        run_dir,
        allowed_states={"failed", "interrupted"},
        command="resume",
    )
    try:
        history_path = artifacts.path / "resume_history.json"
        history = (
            json.loads(history_path.read_text(encoding="utf-8"))
            if history_path.is_file()
            else []
        )
        history.append(
            {
                "resumed_at": datetime.now(timezone.utc).isoformat(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "environment": _environment(
                    resolved.data["runtime"], resolved.data["determinism"]
                ),
                "git": artifacts.git,
            }
        )
        _write_json(history_path, history)
        artifacts.status("running")
        context = _existing_run_context(resolved, artifacts)
        for stage in resolved.data["stages"]:
            if stage["id"] in context.stage_artifacts:
                continue
            artifacts.status("running", stage=stage["id"])
            _stage_log(artifacts, stage["id"], "resumed")
            try:
                context.stage_artifacts[stage["id"]] = HANDLERS[stage["task"]](
                    stage, context
                )
            except BaseException:
                _stage_log(artifacts, stage["id"], "failed")
                raise
            _stage_log(artifacts, stage["id"], "completed")
            _persist_stage_artifacts(context)
        artifacts.status("completed")
        return artifacts.path
    except KeyboardInterrupt as exc:
        artifacts.status("interrupted", error=exc)
        raise
    except BaseException as exc:
        artifacts.status("failed", error=exc)
        raise
