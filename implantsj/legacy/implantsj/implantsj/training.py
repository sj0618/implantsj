"""Train one configurable experiment from explicit manifests."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

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

    @classmethod
    def from_namespace(cls, args) -> "ExperimentConfig":
        return cls(**vars(args))

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
    def checkpoint_suffix(self) -> str:
        return ".safetensors" if self.config.save_format == "safetensors" else ".pt"

    def _checkpoint_payload(self, epoch: int, valid_loss: float) -> dict:
        return {
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

    def _save_last(self, epoch: int, valid_loss: float) -> Path:
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
            valid_metrics = self.objective.run_epoch(
                self.model, self.data.valid_loader, self.device
            )
            # Test is recorded only; checkpoint selection below uses validation only.
            test_metrics = (
                self.objective.run_epoch(self.model, self.data.test_loader, self.device)
                if self.data.test_loader is not None
                else None
            )
            if self.objective.uses_classifier:
                train_accuracy = train_metrics.accuracy
                valid_accuracy = valid_metrics.accuracy
                selection_score = valid_metrics.loss
            else:
                train_accuracy = float("nan")
                valid_accuracy = _mean_prototype_accuracy(
                    self.model,
                    self.data.support_eval_loader,
                    self.data.valid_eval_loader,
                    self.device,
                )
                selection_score = -valid_accuracy
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
                "valid_loss": valid_metrics.loss,
                "test_loss": test_metrics.loss
                if test_metrics is not None
                else float("nan"),
                "train_accuracy": train_accuracy,
                "valid_accuracy": valid_accuracy,
                "test_accuracy": test_accuracy,
            }
            if self.config.loss == "supcon":
                k = self.config.samples_per_class
                row["expected_cross_image_positive_fraction"] = (
                    2 * (k - 1) / (2 * k - 1)
                )
                row["actual_cross_image_positive_fraction"] = (
                    train_metrics.cross_image_positive_fraction
                )
            history.append(row)
            last_path = self._save_last(epoch, valid_metrics.loss)
            if selection_score < best_score:
                best_score = selection_score
                self._copy_best(last_path)
            pd.DataFrame(history).to_csv(self.output_dir / "history.csv", index=False)
            print(
                f"epoch={epoch}/{self.config.epochs} "
                f"train_loss={train_metrics.loss:.5f} "
                f"valid_loss={valid_metrics.loss:.5f} "
                f"valid_accuracy={valid_accuracy:.4f}",
                flush=True,
            )
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
