"""Train a ViT on runtime SAM2-predicted boundary-band-only crops."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from band_vit_2.data import DataConfig
from band_vit_2.losses import CrossEntropyStrategy
from band_vit_2.models import (
    build_classifier,
    checkpoint_payload,
    choose_device,
    create_transforms,
    state_from_payload,
)
from band_vit_2.training import (
    OPTIMIZERS,
    SCHEDULERS,
    create_optimizer,
    create_scheduler,
    plot_curves,
    seed_everything,
)

from .band_preprocessing import BandCropConfig
from .data import (
    create_runtime_data,
    precompute_band_cache,
    prepare_runtime_manifests,
)
from .evaluation import evaluate_loader
from .segmentation import RuntimeSegmenter, SegmenterConfig


SELECTION_METRICS = ("valid_loss", "valid_macro_f1")


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
    min_epochs: int
    selection_metric: str
    sam2_config: str
    sam2_checkpoint: str
    sam_model_checkpoint: str
    sam_input_size: int
    sam_threshold: float
    sam_morph_kernel_size: int
    sam_batch_size: int
    band_working_size: int
    band_outer_radius: int
    band_inner_radius: int
    band_crop_margin: int
    attention_blocks: int
    test_attention_samples: int

    @classmethod
    def from_namespace(cls, args) -> "ExperimentConfig":
        return cls(**vars(args))

    def validate(self) -> None:
        for name in ("train_data", "valid_data", "test_data", "checkpoint"):
            value = getattr(self, name)
            if value and not Path(value).expanduser().is_file():
                raise FileNotFoundError(f"{name}: {value}")
        for name in (
            "epochs",
            "batch_size",
            "learning_rate",
            "input_size",
            "sam_input_size",
            "sam_batch_size",
            "band_working_size",
            "attention_blocks",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_workers < 0 or self.patience < 0 or self.min_epochs < 0:
            raise ValueError("num_workers, patience and min_epochs must be non-negative")
        if self.test_attention_samples < -1:
            raise ValueError("test_attention_samples must be -1 (all) or non-negative")
        if self.selection_metric not in SELECTION_METRICS:
            raise ValueError(f"unsupported selection_metric={self.selection_metric}")


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
        prepared = prepare_runtime_manifests(data_config)
        self.segmenter_config = SegmenterConfig(
            sam2_config=config.sam2_config,
            sam2_checkpoint=config.sam2_checkpoint,
            model_checkpoint=config.sam_model_checkpoint,
            input_size=config.sam_input_size,
            threshold=config.sam_threshold,
            morph_kernel_size=config.sam_morph_kernel_size,
            batch_size=config.sam_batch_size,
        )
        self.band_config = BandCropConfig(
            working_size=config.band_working_size,
            outer_radius=config.band_outer_radius,
            inner_radius=config.band_inner_radius,
            crop_margin=config.band_crop_margin,
            output_size=config.input_size,
        )
        self.preprocessing_metadata = {
            "input_kind": "sam2_predicted_boundary_band_only_crop",
            "mask_files_written": False,
            "cache_lifetime": "current_training_process_only",
            "segmenter_frozen": True,
            "segmenter": self.segmenter_config.metadata(),
            "band_crop": self.band_config.metadata(),
        }
        self._write_initial_config(prepared.labels)

        segmenter = RuntimeSegmenter(self.segmenter_config, self.device)
        try:
            caches, report_path = precompute_band_cache(
                prepared.frames,
                segmenter,
                self.band_config,
                self.output_dir / "preprocessing",
            )
        finally:
            segmenter.close()

        self.model = build_classifier(
            config.model,
            len(prepared.labels),
            self.device,
            pretrained=config.pretrained,
            initial_checkpoint=config.checkpoint,
            checkpoint_format=config.checkpoint_format,
        )
        train_transform, eval_transform, self.mean, self.std = create_transforms(
            self.model, config.input_size
        )
        self.data = create_runtime_data(
            data_config,
            prepared,
            caches,
            train_transform,
            eval_transform,
            report_path,
        )
        self.objective = CrossEntropyStrategy(label_smoothing=config.label_smoothing)
        self.optimizer = create_optimizer(config, self.model.parameters())
        self.scheduler = create_scheduler(config, self.optimizer)
        self.versions = {
            "torch": torch.__version__,
            "timm": timm.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        }
        self._write_run_config()

    @property
    def checkpoint_suffix(self) -> str:
        return ".safetensors" if self.config.save_format == "safetensors" else ".pt"

    def _write_initial_config(self, labels: list[str]) -> None:
        payload = {
            **asdict(self.config),
            "experiment_name_resolved": self.name,
            "device_resolved": self.device,
            "labels": labels,
            "band_vit3_preprocessing": self.preprocessing_metadata,
            "status": "runtime_preprocessing",
        }
        (self.output_dir / "run_config.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _write_run_config(self) -> None:
        path = self.output_dir / "run_config.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(
            {
                "versions": self.versions,
                "checkpoint_selection_metric": self.config.selection_metric,
                "status": "training",
            }
        )
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _checkpoint_payload(self, epoch: int, selection_value: float) -> dict:
        return {
            "epoch": epoch,
            "architecture": self.config.model,
            "training_kind": "sam2_predicted_boundary_band_only_crop",
            "num_classes": len(self.data.labels),
            "labels": self.data.labels,
            "label_to_index": self.data.label_to_index,
            "input_size": self.config.input_size,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "selection_metric": self.config.selection_metric,
            "selection_value": selection_value,
            "band_vit3_preprocessing": self.preprocessing_metadata,
        }

    def _save_checkpoint(self, name: str, epoch: int, value: float) -> Path:
        import torch

        path = self.checkpoint_dir / f"{name}{self.checkpoint_suffix}"
        payload = self._checkpoint_payload(epoch, value)
        if self.config.save_format == "ckpt":
            torch.save(payload, path)
            return path
        from safetensors.torch import save_file

        state = {
            key: tensor.detach().contiguous().cpu()
            for key, tensor in payload["model_state_dict"].items()
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

    def _load_model_state(self, path: Path) -> None:
        payload = checkpoint_payload(path, "cpu", self.config.save_format)
        self.model.load_state_dict(state_from_payload(payload), strict=True)

    def _selection_value(self, valid_metrics) -> float:
        if self.config.selection_metric == "valid_loss":
            return float(valid_metrics.loss)
        return float(valid_metrics.macro_f1)

    def _improved(self, candidate: float, best: float) -> bool:
        if self.config.selection_metric == "valid_loss":
            return candidate < best
        return candidate > best

    def run(self) -> Path:
        best_value = (
            float("inf")
            if self.config.selection_metric == "valid_loss"
            else float("-inf")
        )
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
            selection_value = self._selection_value(valid_metrics)
            last_path = self._save_checkpoint("last", epoch, selection_value)
            if self._improved(selection_value, best_value):
                best_value = selection_value
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
            if (
                self.config.patience
                and epoch >= self.config.min_epochs
                and patience_count >= self.config.patience
            ):
                print(f"early stopping after {epoch} epochs", flush=True)
                break

        if best_path is None:
            raise RuntimeError("training completed without a checkpoint")
        self._load_model_state(best_path)
        evaluation_root = self.output_dir / "evaluation"
        valid_output = evaluate_loader(
            classifier=self.model,
            loader=self.data.valid_loader,
            labels=self.data.labels,
            output_dir=evaluation_root / "valid",
            device=self.device,
            mean=self.mean,
            std=self.std,
            attention_samples=0,
            attention_blocks=self.config.attention_blocks,
            preprocessing_metadata=self.preprocessing_metadata,
        )
        valid_report = json.loads(
            (valid_output / "metrics.json").read_text(encoding="utf-8")
        )
        test_output = None
        test_report = None
        if self.data.test_loader is not None:
            test_output = evaluate_loader(
                classifier=self.model,
                loader=self.data.test_loader,
                labels=self.data.labels,
                output_dir=evaluation_root / "test",
                device=self.device,
                mean=self.mean,
                std=self.std,
                attention_samples=self.config.test_attention_samples,
                attention_blocks=self.config.attention_blocks,
                preprocessing_metadata=self.preprocessing_metadata,
            )
            test_report = json.loads(
                (test_output / "metrics.json").read_text(encoding="utf-8")
            )
        summary = {
            "selection_metric": self.config.selection_metric,
            "best_selection_value": best_value,
            "best_checkpoint": str(best_path),
            "valid_metrics": valid_report,
            "valid_evaluation_dir": str(valid_output),
            "test_metrics": test_report,
            "test_evaluation_dir": str(test_output) if test_output else None,
            "test_attention_samples_requested": (
                "all" if self.config.test_attention_samples == -1
                else self.config.test_attention_samples
            ),
            "band_preprocessing_report": str(self.data.preprocessing_report),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name")
    parser.add_argument("--model", required=True, help="timm ViT architecture")
    parser.add_argument(
        "--pretrained", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--checkpoint", help="optional initial ViT weights")
    parser.add_argument(
        "--checkpoint-format", choices=["auto", "ckpt", "safetensors"], default="auto"
    )
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--valid-data", required=True)
    parser.add_argument("--test-data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="adamw")
    parser.add_argument("--scheduler", choices=SCHEDULERS, default="cosine")
    parser.add_argument("--save-format", choices=["ckpt", "safetensors"], default="ckpt")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler-step-size", type=int, default=10)
    parser.add_argument("--scheduler-gamma", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument(
        "--selection-metric", choices=SELECTION_METRICS, default="valid_macro_f1"
    )
    parser.add_argument(
        "--sam2-config", default="configs/sam2.1/sam2.1_hiera_t.yaml"
    )
    parser.add_argument(
        "--sam2-checkpoint",
        default="/workspace/implant_seg/checkpoints/sam2.1_hiera_tiny.pt",
    )
    parser.add_argument(
        "--sam-model-checkpoint",
        default=(
            "/workspace/implant_seg/model_pt/"
            "best_sam2.1_tiny_adapter_unet.pth"
        ),
    )
    parser.add_argument("--sam-input-size", type=int, default=512)
    parser.add_argument("--sam-threshold", type=float, default=0.5)
    parser.add_argument("--sam-morph-kernel-size", type=int, default=3)
    parser.add_argument("--sam-batch-size", type=int, default=2)
    parser.add_argument("--band-working-size", type=int, default=224)
    parser.add_argument("--band-outer-radius", type=int, default=15)
    parser.add_argument("--band-inner-radius", type=int, default=15)
    parser.add_argument("--band-crop-margin", type=int, default=4)
    parser.add_argument("--attention-blocks", type=int, default=6)
    parser.add_argument("--test-attention-samples", type=int, default=-1)
    return parser


def train(args: argparse.Namespace) -> Path:
    return Experiment(ExperimentConfig.from_namespace(args)).run()


def main(argv: Sequence[str] | None = None) -> int:
    output = train(build_parser().parse_args(argv))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

