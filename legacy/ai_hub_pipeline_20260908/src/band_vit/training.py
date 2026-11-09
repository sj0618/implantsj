"""Train a supervised ViT classifier with optional band-attention loss."""

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

# Keep Hub downloads quiet and configure cuBLAS before CUDA is initialized.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .data import DataConfig, create_data, prepare_manifests
from .losses import CrossEntropyStrategy
from .models import (
    attach_attention_capture,
    build_classifier,
    checkpoint_payload,
    choose_device,
    create_transforms,
    patch_grid_size,
    state_from_payload,
)


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Flash/memory-efficient SDPA backward is not deterministic in this stack.
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


OPTIMIZERS = ("adamw", "sgd")
SCHEDULERS = ("cosine", "step", "none")


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str | None
    model: str
    pretrained: bool
    checkpoint: str | None
    checkpoint_format: str
    train_data: str
    valid_data: str
    test_data: str | None
    output_dir: str
    optimizer: str
    scheduler: str
    save_format: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    momentum: float
    scheduler_step_size: int
    scheduler_gamma: float
    label_smoothing: float
    input_size: int
    num_workers: int
    seed: int
    device: str
    patience: int
    band_loss_weight: float
    band_alpha: float
    band_mask_column: str
    band_attention_blocks: int
    band_strong_aug: bool
    test_attention_samples: int

    @classmethod
    def from_namespace(cls, args) -> "ExperimentConfig":
        return cls(**vars(args))

    def validate(self) -> None:
        positive = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "input_size": self.input_size,
        }
        invalid = {name: value for name, value in positive.items() if value <= 0}
        if invalid:
            raise ValueError(f"values must be positive: {invalid}")
        if self.num_workers < 0 or self.patience < 0:
            raise ValueError("num_workers and patience must be non-negative")
        if self.band_loss_weight < 0 or self.band_alpha < 0:
            raise ValueError("band loss weight and alpha must be non-negative")
        if self.band_attention_blocks <= 0:
            raise ValueError("band_attention_blocks must be positive")
        if self.test_attention_samples < -1:
            raise ValueError("test_attention_samples must be -1 (all) or non-negative")
        if not self.band_mask_column.strip():
            raise ValueError("band_mask_column must be non-empty")
        for name, value in {
            "checkpoint": self.checkpoint,
            "train_data": self.train_data,
            "valid_data": self.valid_data,
            "test_data": self.test_data,
        }.items():
            if value and not Path(value).expanduser().is_file():
                raise FileNotFoundError(f"{name}: {value}")


def create_optimizer(config: ExperimentConfig, parameters):
    import torch

    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
        )
    raise KeyError(f"unknown optimizer {config.optimizer!r}")


def create_scheduler(config: ExperimentConfig, optimizer):
    import torch

    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(config.epochs, 1)
        )
    if config.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma,
        )
    if config.scheduler == "none":
        return None
    raise KeyError(f"unknown scheduler {config.scheduler!r}")


class Experiment:
    def __init__(self, config: ExperimentConfig) -> None:
        import timm
        import torch

        config.validate()
        seed_everything(config.seed)
        self.config = config
        self.device = choose_device(config.device)
        self.name = config.experiment_name or Path(config.output_dir).name
        self.output_dir = Path(config.output_dir).expanduser().resolve()
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        data_config = DataConfig(
            train_path=config.train_data,
            valid_path=config.valid_data,
            test_path=config.test_data,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        prepared = prepare_manifests(data_config)
        self.model = build_classifier(
            config.model,
            len(prepared.labels),
            self.device,
            pretrained=config.pretrained,
            initial_checkpoint=config.checkpoint,
            checkpoint_format=config.checkpoint_format,
        )
        band_grid = None
        if config.band_loss_weight > 0:
            attach_attention_capture(self.model, config.band_attention_blocks)
            band_grid = patch_grid_size(self.model)
        train_transform, eval_transform, mean, std = create_transforms(
            self.model, config.input_size
        )
        self.data = create_data(
            data_config,
            prepared,
            train_transform,
            eval_transform,
            band_patch_grid=band_grid,
            band_image_size=config.input_size,
            band_mask_column=config.band_mask_column,
            band_strong_aug=config.band_strong_aug,
            band_mean=mean,
            band_std=std,
        )
        self.objective = CrossEntropyStrategy(
            label_smoothing=config.label_smoothing,
            band_loss_weight=config.band_loss_weight,
            band_alpha=config.band_alpha,
        )
        self.optimizer = create_optimizer(config, self.model.parameters())
        self.scheduler = create_scheduler(config, self.optimizer)
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
            "architecture": self.config.model,
            "training_kind": "ce_band"
            if self.config.band_loss_weight > 0
            else "ce",
            "num_classes": len(self.data.labels),
            "labels": self.data.labels,
            "label_to_index": self.data.label_to_index,
            "input_size": self.config.input_size,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "valid_loss": valid_loss,
            "band_loss_weight": self.config.band_loss_weight,
            "band_alpha": self.config.band_alpha,
            "band_mask_column": self.config.band_mask_column,
            "band_attention_blocks": self.config.band_attention_blocks,
            "test_attention_samples": self.config.test_attention_samples,
        }

    def _save_checkpoint(self, name: str, epoch: int, valid_loss: float) -> Path:
        import torch

        path = self.checkpoint_dir / f"{name}{self.checkpoint_suffix}"
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
                "architecture": self.config.model,
                "training_kind": payload["training_kind"],
            },
        )
        metadata = {
            key: value
            for key, value in payload.items()
            if not key.endswith("state_dict")
        }
        path.with_suffix(".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def _copy_checkpoint(self, source: Path, name: str) -> Path:
        target = self.checkpoint_dir / f"{name}{self.checkpoint_suffix}"
        shutil.copy2(source, target)
        source_metadata = source.with_suffix(".json")
        if source_metadata.is_file():
            shutil.copy2(source_metadata, target.with_suffix(".json"))
        return target

    def _write_run_config(self) -> None:
        payload = {
            **asdict(self.config),
            "experiment_name_resolved": self.name,
            "device_resolved": self.device,
            "labels": self.data.labels,
            "versions": self.versions,
            "checkpoint_selection_metric": "valid_loss",
        }
        (self.output_dir / "run_config.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _load_model_state(self, path: Path) -> None:
        payload = checkpoint_payload(path, "cpu", self.config.save_format)
        self.model.load_state_dict(state_from_payload(payload), strict=True)

    def run(self) -> Path:
        self._write_run_config()
        best_loss = float("inf")
        best_path: Path | None = None
        patience_count = 0
        history: list[dict] = []

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self.objective.run_epoch(
                self.model, self.data.train_loader, self.device, self.optimizer
            )
            valid_metrics = self.objective.run_epoch(
                self.model, self.data.valid_loader, self.device
            )
            learning_rate = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                self.scheduler.step()
            row = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": train_metrics.loss,
                "train_classification_loss": train_metrics.classification_loss,
                "train_band_loss": train_metrics.band_loss,
                "train_accuracy": train_metrics.accuracy,
                "train_macro_precision": train_metrics.macro_precision,
                "train_macro_recall": train_metrics.macro_recall,
                "train_macro_f1": train_metrics.macro_f1,
                "train_weighted_precision": train_metrics.weighted_precision,
                "train_weighted_recall": train_metrics.weighted_recall,
                "train_weighted_f1": train_metrics.weighted_f1,
                "valid_loss": valid_metrics.loss,
                "valid_accuracy": valid_metrics.accuracy,
                "valid_macro_precision": valid_metrics.macro_precision,
                "valid_macro_recall": valid_metrics.macro_recall,
                "valid_macro_f1": valid_metrics.macro_f1,
                "valid_weighted_precision": valid_metrics.weighted_precision,
                "valid_weighted_recall": valid_metrics.weighted_recall,
                "valid_weighted_f1": valid_metrics.weighted_f1,
            }
            history.append(row)
            pd.DataFrame(history).to_csv(self.output_dir / "history.csv", index=False)
            last_path = self._save_checkpoint("last", epoch, valid_metrics.loss)
            if valid_metrics.loss < best_loss:
                best_loss = valid_metrics.loss
                patience_count = 0
                best_path = self._copy_checkpoint(last_path, "best")
            else:
                patience_count += 1

            print(
                f"epoch={epoch}/{self.config.epochs} "
                f"train_loss={train_metrics.loss:.5f} "
                f"valid_loss={valid_metrics.loss:.5f} "
                f"valid_accuracy={valid_metrics.accuracy:.4f} "
                f"valid_macro_f1={valid_metrics.macro_f1:.4f}",
                flush=True,
            )
            if self.config.patience and patience_count >= self.config.patience:
                print(f"early stopping after {epoch} epochs", flush=True)
                break

        if best_path is None:
            raise RuntimeError("training completed without a checkpoint")
        self._load_model_state(best_path)
        test_epoch_metrics = (
            self.objective.run_epoch(self.model, self.data.test_loader, self.device)
            if self.data.test_loader is not None
            else None
        )

        from .evaluation import evaluate_classifier

        evaluation_root = self.output_dir / "evaluation"
        valid_output = evaluate_classifier(
            checkpoint=best_path,
            manifest=self.config.valid_data,
            output_dir=evaluation_root / "valid",
            checkpoint_format=self.config.save_format,
            input_size=self.config.input_size,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            device=self.device,
            attention_samples=0,
            attention_blocks=self.config.band_attention_blocks,
        )
        valid_report = json.loads(
            (valid_output / "metrics.json").read_text(encoding="utf-8")
        )

        test_output = None
        test_report = None
        if self.config.test_data:
            test_output = evaluate_classifier(
                checkpoint=best_path,
                manifest=self.config.test_data,
                output_dir=evaluation_root / "test",
                checkpoint_format=self.config.save_format,
                input_size=self.config.input_size,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                device=self.device,
                attention_samples=self.config.test_attention_samples,
                attention_blocks=self.config.band_attention_blocks,
            )
            test_report = json.loads(
                (test_output / "metrics.json").read_text(encoding="utf-8")
            )

        summary = {
            "best_valid_loss": best_loss,
            "best_checkpoint": str(best_path),
            "valid_metrics": valid_report,
            "valid_evaluation_dir": str(valid_output),
            "test_loss": (
                test_epoch_metrics.loss if test_epoch_metrics is not None else None
            ),
            "test_metrics": test_report,
            "test_evaluation_dir": str(test_output) if test_output else None,
            "test_attention_samples_requested": self.config.test_attention_samples,
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        plot_curves(self.output_dir / "history.csv", self.output_dir / "curves.png")
        if test_report is not None:
            print(
                f"test_accuracy={test_report['accuracy']:.4f} "
                f"test_macro_f1={test_report['macro_f1']:.4f} "
                f"test_weighted_f1={test_report['weighted_f1']:.4f}",
                flush=True,
            )
        return self.output_dir


def plot_curves(history_csv: str | Path, output_png: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), constrained_layout=True)
    axes[0].plot(frame["epoch"], frame["train_loss"], label="train")
    axes[0].plot(frame["epoch"], frame["valid_loss"], label="valid")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(frame["epoch"], frame["train_accuracy"], label="train")
    axes[1].plot(frame["epoch"], frame["valid_accuracy"], label="valid")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[2].plot(frame["epoch"], frame["train_macro_f1"], label="train")
    axes[2].plot(frame["epoch"], frame["valid_macro_f1"], label="valid")
    axes[2].set(title="Macro F1", xlabel="Epoch", ylabel="F1", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    output = Path(output_png).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name")
    parser.add_argument("--model", required=True, help="timm ViT architecture")
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
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="adamw")
    parser.add_argument("--scheduler", choices=SCHEDULERS, default="cosine")
    parser.add_argument(
        "--save-format", choices=["ckpt", "safetensors"], default="ckpt"
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler-step-size", type=int, default=10)
    parser.add_argument("--scheduler-gamma", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--band-loss-weight", type=float, default=0.0)
    parser.add_argument("--band-alpha", type=float, default=4.0)
    parser.add_argument("--band-mask-column", default="mask_path")
    parser.add_argument("--band-attention-blocks", type=int, default=6)
    parser.add_argument(
        "--band-strong-aug",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--test-attention-samples",
        type=int,
        default=-1,
        help="save test attention maps: -1=all rows, 0=none, N=first N rows",
    )
    return parser


def train(args: argparse.Namespace) -> Path:
    return Experiment(ExperimentConfig.from_namespace(args)).run()


def main(argv: Sequence[str] | None = None) -> int:
    output = train(build_parser().parse_args(argv))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
