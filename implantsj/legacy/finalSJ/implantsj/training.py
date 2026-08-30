"""Train one configurable experiment from explicit manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import DATA_MODULES, ManifestDataConfig, create_data_module
from .losses import LOSS_STRATEGIES, create_loss_strategy
from .models import ModelSpec, build_training_model, choose_device


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


OPTIMIZERS = ("adamw", "sgd")
SCHEDULERS = ("cosine", "step", "none")


def create_optimizer(name: str, parameters, config: "ExperimentConfig"):
    import torch

    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
        )
    raise KeyError(f"unknown optimizer {name!r}; available={list(OPTIMIZERS)}")


def create_scheduler(name: str, optimizer, config: "ExperimentConfig"):
    import torch

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(config.epochs, 1)
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma,
        )
    if name == "none":
        return None
    raise KeyError(f"unknown scheduler {name!r}; available={list(SCHEDULERS)}")


def create_transforms(model, input_size: int):
    from timm.data import create_transform, resolve_model_data_config

    config = resolve_model_data_config(getattr(model, "backbone", model))
    config["input_size"] = (3, input_size, input_size)
    return (
        create_transform(**config, is_training=True),
        create_transform(**config, is_training=False),
    )


def _backbone_embeddings(model, loader, device: str):
    import torch

    model.eval()
    arrays: list[np.ndarray] = []
    labels: list[str] = []
    with torch.inference_mode():
        for images, _, batch_labels, _ in loader:
            images = images.to(device)
            if hasattr(model, "backbone"):
                features = model(images, return_projection=False)
            else:
                values = model.forward_features(images)
                features = (
                    model.forward_head(values, pre_logits=True)
                    if hasattr(model, "forward_head")
                    else values
                )
            arrays.append(features.detach().cpu().numpy())
            labels.extend(str(item) for item in batch_labels)
    return np.concatenate(arrays), np.asarray(labels, dtype=str)


def _mean_prototype_accuracy(model, support_loader, query_loader, device: str) -> float:
    support_x, support_y = _backbone_embeddings(model, support_loader, device)
    query_x, query_y = _backbone_embeddings(model, query_loader, device)
    label_order = sorted(set(support_y.tolist()))
    prototypes = np.stack(
        [support_x[support_y == label].mean(axis=0) for label in label_order]
    )
    distances = ((query_x[:, None] - prototypes[None, :]) ** 2).sum(axis=2)
    predictions = np.asarray(label_order, dtype=str)[distances.argmin(axis=1)]
    return float((predictions == query_y).mean())


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_labels(labels: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(labels), ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def validate_supcon_pk(
    labels: Sequence[str], batch_size: int, samples_per_class: int
) -> None:
    """Fail before model construction when P×K cannot form SupCon batches."""
    if samples_per_class < 2:
        raise ValueError("SupCon requires samples_per_class >= 2")
    if batch_size % samples_per_class:
        raise ValueError("SupCon batch_size must be divisible by samples_per_class")
    p = batch_size // samples_per_class
    counts = pd.Series([str(label) for label in labels]).value_counts()
    too_small = {
        label: int(count)
        for label, count in counts.items()
        if int(count) < samples_per_class
    }
    if too_small:
        raise ValueError(
            "SupCon P×K requires at least K distinct images per class; "
            f"K={samples_per_class}, too_small={too_small}"
        )
    if len(counts) < p:
        raise ValueError(
            f"SupCon P×K requires P={p} classes but data has {len(counts)}"
        )


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str | None
    data: str
    model: str
    pretrained: bool
    checkpoint: str | None
    checkpoint_format: str
    train_data: str
    valid_data: str
    test_data: str | None
    output_dir: str
    scope: str
    loss: str
    optimizer: str
    scheduler: str
    save_format: str
    epochs: int
    batch_size: int
    samples_per_class: int
    learning_rate: float
    weight_decay: float
    momentum: float
    scheduler_step_size: int
    scheduler_gamma: float
    label_smoothing: float
    temperature: float
    triplet_margin: float
    input_size: int
    projection_dim: int
    projection_hidden_dim: int
    num_workers: int
    seed: int
    device: str
    canonical_metadata: Mapping[str, Any] | None = None

    @classmethod
    def from_namespace(cls, args) -> "ExperimentConfig":
        values = vars(args)
        known = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in known})

    def validate(self) -> None:
        positive = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "samples_per_class": self.samples_per_class,
            "learning_rate": self.learning_rate,
            "input_size": self.input_size,
            "projection_dim": self.projection_dim,
            "projection_hidden_dim": self.projection_hidden_dim,
        }
        invalid = {name: value for name, value in positive.items() if value <= 0}
        if invalid:
            raise ValueError(f"values must be positive: {invalid}")
        if self.checkpoint and not Path(self.checkpoint).expanduser().is_file():
            raise FileNotFoundError(self.checkpoint)
        for name, value in {
            "train_data": self.train_data,
            "valid_data": self.valid_data,
            "test_data": self.test_data,
        }.items():
            if value and not Path(value).expanduser().is_file():
                raise FileNotFoundError(f"{name}: {value}")
        if self.canonical_metadata is not None and not isinstance(
            self.canonical_metadata, Mapping
        ):
            raise ValueError("canonical_metadata must be a mapping")


class Experiment:
    """Owns one model/data/loss/optimizer configuration and one output folder."""

    def __init__(self, config: ExperimentConfig) -> None:
        import timm
        import torch

        config.validate()
        seed_everything(config.seed)
        self.config = config
        self.device = choose_device(config.device)
        self.output_dir = Path(config.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.name = config.experiment_name or self.output_dir.name
        self.objective = create_loss_strategy(
            config.loss,
            label_smoothing=config.label_smoothing,
            temperature=config.temperature,
            margin=config.triplet_margin,
        )
        data_module = create_data_module(
            config.data,
            ManifestDataConfig(
                train_path=config.train_data,
                valid_path=config.valid_data,
                test_path=config.test_data,
                scope=config.scope,
                batch_size=config.batch_size,
                samples_per_class=config.samples_per_class,
                num_workers=config.num_workers,
                seed=config.seed,
            ),
        )
        data_module.prepare()
        if config.loss == "supcon":
            validate_supcon_pk(
                data_module.frames["train"]["label"].astype(str).tolist(),
                config.batch_size,
                config.samples_per_class,
            )
        self.canonical_metadata = (
            dict(config.canonical_metadata)
            if config.canonical_metadata is not None
            else None
        )
        self._validate_canonical_metadata(data_module.labels)
        self.spec = ModelSpec(
            name=self.name,
            architecture=config.model,
            initial_checkpoint=str(Path(config.checkpoint).expanduser().resolve())
            if config.checkpoint
            else None,
            checkpoint_format=config.checkpoint_format,
            pretrained=config.pretrained,
            training_kind=config.loss,
            train_scope=config.scope,
            input_size=config.input_size,
            projection_dim=config.projection_dim,
            projection_hidden_dim=config.projection_hidden_dim,
        )
        self.model = build_training_model(
            self.spec, config.loss, len(data_module.labels), self.device
        )
        train_transform, eval_transform = create_transforms(
            self.model, config.input_size
        )
        self.data = data_module.build_loaders(
            train_transform,
            eval_transform,
            two_views=self.objective.two_views,
            balanced_train=self.objective.balanced_train,
            balanced_eval=self.objective.balanced_eval,
        )
        self.optimizer = create_optimizer(
            config.optimizer, self.model.parameters(), config
        )
        self.scheduler = create_scheduler(config.scheduler, self.optimizer, config)
        self.versions = {
            "torch": torch.__version__,
            "timm": timm.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        }

    @property
    def is_canonical(self) -> bool:
        return self.canonical_metadata is not None

    def _validate_canonical_metadata(self, labels: Sequence[str]) -> None:
        if not self.is_canonical:
            return
        assert self.canonical_metadata is not None
        required = {
            "model_role",
            "protocol_fingerprint",
            "base_checkpoint_sha256",
            "assignment_sha256",
        }
        missing = sorted(required - set(self.canonical_metadata))
        if missing:
            raise ValueError(f"canonical_metadata missing fields: {missing}")
        if self.config.test_data is not None:
            raise ValueError("canonical training must not bind test_data")
        if self.config.save_format != "ckpt":
            raise ValueError(
                "canonical training requires the fixed-budget last.pt checkpoint"
            )
        if not self.config.checkpoint:
            raise ValueError("canonical training requires a Base checkpoint")
        base_sha256 = _sha256_file(self.config.checkpoint)
        if self.canonical_metadata["base_checkpoint_sha256"] != base_sha256:
            raise ValueError("canonical Base checkpoint SHA-256 mismatch")
        declared_objective = self.canonical_metadata.get("objective")
        expected_objective = (
            "pure_supcon" if self.config.loss == "supcon" else "cross_entropy"
        )
        if declared_objective is not None and declared_objective != expected_objective:
            raise ValueError("canonical objective does not match configured loss")
        declared_labels = self.canonical_metadata.get("ordered_labels")
        if declared_labels is not None and list(declared_labels) != list(labels):
            raise ValueError("canonical ordered labels do not match training data")
        declared_architecture = self.canonical_metadata.get("architecture")
        if (
            declared_architecture is not None
            and declared_architecture != self.config.model
        ):
            raise ValueError("canonical architecture does not match configured model")
        declared_budget = self.canonical_metadata.get("epochs")
        if declared_budget is not None and int(declared_budget) != self.config.epochs:
            raise ValueError("canonical epoch budget does not match configured epochs")

    def _canonical_provenance(self, epoch: int) -> dict[str, Any]:
        if not self.is_canonical:
            return {}
        assert self.canonical_metadata is not None
        metadata = self.canonical_metadata
        projection = {
            "output_dim": self.spec.projection_dim,
            "hidden_dim": self.spec.projection_hidden_dim,
        }
        return {
            "model_role": metadata["model_role"],
            "architecture": self.spec.architecture,
            "base_checkpoint_sha256": metadata["base_checkpoint_sha256"],
            "base_checkpoint_path": self.spec.initial_checkpoint,
            "scope": self.config.scope,
            "objective": "pure_supcon"
            if self.config.loss == "supcon"
            else "cross_entropy",
            "ordered_labels_sha256": _sha256_labels(self.data.labels),
            "projection": projection,
            "seed": self.config.seed,
            "epochs": self.config.epochs,
            "epoch": epoch,
            "final_epoch": epoch,
            "checkpoint_selector": "last.pt",
            "protocol_fingerprint": metadata["protocol_fingerprint"],
            "assignment_sha256": metadata["assignment_sha256"],
            "lineage": dict(metadata.get("lineage", {})),
        }

    @property
    def checkpoint_suffix(self) -> str:
        return ".safetensors" if self.config.save_format == "safetensors" else ".pt"

    def _checkpoint_payload(self, epoch: int, valid_loss: float | None) -> dict:
        payload = {
            "epoch": epoch,
            "architecture": self.spec.architecture,
            "training_kind": self.config.loss,
            "train_scope": self.config.scope,
            "num_classes": len(self.data.labels),
            "labels": self.data.labels,
            "label_to_index": self.data.label_to_index,
            "projection_dim": self.spec.projection_dim,
            "projection_hidden_dim": self.spec.projection_hidden_dim,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
            "valid_loss": valid_loss,
            "initial_checkpoint": self.spec.initial_checkpoint,
        }
        payload.update(self._canonical_provenance(epoch))
        return payload

    def _save_last(self, epoch: int, valid_loss: float | None) -> Path:
        import torch

        path = self.checkpoint_dir / f"last{self.checkpoint_suffix}"
        payload = self._checkpoint_payload(epoch, valid_loss)
        if self.config.save_format == "ckpt":
            torch.save(payload, path)
            return path
        from safetensors.torch import save_file

        state = {
            key: value.detach().contiguous().cpu()
            for key, value in payload["model_state_dict"].items()
        }
        save_file(
            state,
            str(path),
            metadata={
                "epoch": str(epoch),
                "architecture": self.spec.architecture,
                "loss": self.config.loss,
            },
        )
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    key: value
                    for key, value in payload.items()
                    if not key.endswith("state_dict")
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def _copy_best(self, last_path: Path) -> None:
        best_path = self.checkpoint_dir / f"best{self.checkpoint_suffix}"
        shutil.copy2(last_path, best_path)
        metadata = last_path.with_suffix(".json")
        if metadata.exists():
            shutil.copy2(metadata, best_path.with_suffix(".json"))

    def _write_run_files(self) -> None:
        checkpoint_path = self.checkpoint_dir / f"last{self.checkpoint_suffix}"
        resolved_spec = {
            **self.spec.to_dict(),
            "adapter": self.objective.checkpoint_adapter,
            "checkpoint": str(checkpoint_path),
            "checkpoint_format": self.config.save_format,
            "num_classes": len(self.data.labels),
            "labels": self.data.labels,
        }
        if self.is_canonical:
            resolved_spec["canonical_checkpoint"] = self._canonical_provenance(
                self.config.epochs
            )
        (self.output_dir / "model_spec.json").write_text(
            json.dumps(resolved_spec, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        run_config = {
            **asdict(self.config),
            "experiment_name_resolved": self.name,
            "device_resolved": self.device,
            "labels": self.data.labels,
            "versions": self.versions,
            "checkpoint_selection_metric": "valid_loss"
            if self.objective.uses_classifier
            else "valid_accuracy",
        }
        if self.is_canonical:
            run_config["canonical_provenance"] = self._canonical_provenance(
                self.config.epochs
            )
        (self.output_dir / "run_config.json").write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def run(self) -> Path:
        self._write_run_files()
        best_score = float("inf")
        history: list[dict] = []
        for epoch in range(1, self.config.epochs + 1):
            self.data.set_epoch(epoch)
            train_metrics = self.objective.run_epoch(
                self.model, self.data.train_loader, self.device, self.optimizer
            )
            if self.config.loss == "supcon" and self.is_canonical:
                valid_metrics = None
                valid_accuracy = _mean_prototype_accuracy(
                    self.model,
                    self.data.support_eval_loader,
                    self.data.valid_eval_loader,
                    self.device,
                )
                selection_score = -valid_accuracy
            else:
                valid_metrics = self.objective.run_epoch(
                    self.model, self.data.valid_loader, self.device
                )
                if self.objective.uses_classifier:
                    valid_accuracy = valid_metrics.accuracy
                    selection_score = valid_metrics.loss
                else:
                    valid_accuracy = _mean_prototype_accuracy(
                        self.model,
                        self.data.support_eval_loader,
                        self.data.valid_eval_loader,
                        self.device,
                    )
                    selection_score = -valid_accuracy
            test_metrics = (
                self.objective.run_epoch(self.model, self.data.test_loader, self.device)
                if self.data.test_loader is not None and not self.is_canonical
                else None
            )
            train_accuracy = (
                train_metrics.accuracy
                if self.objective.uses_classifier
                else float("nan")
            )
            if test_metrics is None:
                test_accuracy = float("nan")
            elif self.objective.uses_classifier:
                test_accuracy = test_metrics.accuracy
            else:
                assert self.data.test_eval_loader is not None
                test_accuracy = _mean_prototype_accuracy(
                    self.model,
                    self.data.support_eval_loader,
                    self.data.test_eval_loader,
                    self.device,
                )
            learning_rate = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                self.scheduler.step()
            row = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": train_metrics.loss,
                "train_accuracy": train_accuracy,
                "valid_accuracy": valid_accuracy,
            }
            if valid_metrics is None:
                row["valid_loss"] = None
                row["valid_loss_not_applicable_reason"] = (
                    "SupCon validation uses common encoder mean-prototype accuracy; "
                    "ordinary-batch SupCon loss is not applicable."
                )
            else:
                row["valid_loss"] = valid_metrics.loss
            if not self.is_canonical:
                row["test_loss"] = (
                    test_metrics.loss if test_metrics is not None else float("nan")
                )
                row["test_accuracy"] = test_accuracy
            if self.config.loss == "supcon":
                k = self.config.samples_per_class
                row["expected_cross_image_positive_fraction"] = (
                    2 * (k - 1) / (2 * k - 1)
                )
                row["actual_cross_image_positive_fraction"] = (
                    train_metrics.cross_image_positive_fraction
                )
                row["cross_image_positive_count"] = (
                    train_metrics.cross_image_positive_count
                )
                row["positive_count"] = train_metrics.positive_count
            history.append(row)
            valid_loss = valid_metrics.loss if valid_metrics is not None else None
            last_path = self._save_last(epoch, valid_loss)
            if selection_score < best_score:
                best_score = selection_score
                self._copy_best(last_path)
            pd.DataFrame(history).to_csv(self.output_dir / "history.csv", index=False)
            message = (
                f"epoch={epoch}/{self.config.epochs} "
                f"train_loss={train_metrics.loss:.5f} "
                f"valid_accuracy={valid_accuracy:.4f}"
            )
            if valid_metrics is not None:
                message += f" valid_loss={valid_metrics.loss:.5f}"
            print(message, flush=True)
        plot_curves(self.output_dir / "history.csv", self.output_dir / "curves.png")
        return self.output_dir


def plot_curves(history_csv: str | Path, output_png: str | Path) -> Path:
    from .visualization import create_curves

    return create_curves(
        argparse.Namespace(
            history=[f"experiment={Path(history_csv).expanduser().resolve()}"],
            output=str(Path(output_png).expanduser().resolve()),
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name")
    parser.add_argument("--data", choices=sorted(DATA_MODULES), default="manifest")
    parser.add_argument("--model", required=True, help="any timm architecture")
    parser.add_argument(
        "--pretrained", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--checkpoint", help="optional initial weights")
    parser.add_argument(
        "--checkpoint-format",
        choices=["auto", "ckpt", "safetensors"],
        default="auto",
    )
    parser.add_argument(
        "--train-data", "--train-manifest", dest="train_data", required=True
    )
    parser.add_argument(
        "--valid-data", "--valid-manifest", dest="valid_data", required=True
    )
    parser.add_argument("--test-data", "--test-manifest", dest="test_data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scope", choices=["closed", "all"], default="all")
    parser.add_argument("--loss", choices=sorted(LOSS_STRATEGIES), default="ce")
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="adamw")
    parser.add_argument("--scheduler", choices=SCHEDULERS, default="cosine")
    parser.add_argument(
        "--save-format", choices=["ckpt", "safetensors"], default="ckpt"
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler-step-size", type=int, default=10)
    parser.add_argument("--scheduler-gamma", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--triplet-margin", type=float, default=0.2)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--projection-hidden-dim", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    return parser


def train(args: argparse.Namespace) -> Path:
    return Experiment(ExperimentConfig.from_namespace(args)).run()


def main(argv: Sequence[str] | None = None) -> int:
    output = train(build_parser().parse_args(argv))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
