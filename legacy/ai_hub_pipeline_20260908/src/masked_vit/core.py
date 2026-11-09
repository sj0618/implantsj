"""masked_vit 자체 완결용 공용 헬퍼 모음.

기존에 band_vit_2에서 import하던 것들을 전부 흡수한 파일:
  - manifest 읽기 / square padding          (<- band_vit_2/data.py)
  - ViT 생성 / 체크포인트 / attention 캡처   (<- band_vit_2/models.py)
  - CrossEntropy 학습 objective             (<- band_vit_2/losses.py)
  - optimizer / scheduler / seed / 곡선 plot (<- band_vit_2/training.py)
  - 분류 metric / confusion matrix / attention 시각화 (<- band_vit_2/evaluation.py)
로직은 원본과 동일. 이 패키지는 다른 패키지를 일절 import하지 않는다.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


# =====================================================================
# manifest / padding (<- band_vit_2/data.py)
# =====================================================================

def square_padding(size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Return centered left/top/right/bottom padding for a square canvas."""
    width, height = size
    side = max(width, height)
    horizontal = side - width
    vertical = side - height
    left = horizontal // 2
    top = vertical // 2
    return left, top, horizontal - left, vertical - top


def pad_to_square(image: Image.Image, fill=0) -> Image.Image:
    return ImageOps.expand(image, border=square_padding(image.size), fill=fill)


class SquarePad:
    def __init__(self, fill=0) -> None:
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        return pad_to_square(image, fill=self.fill)


def _resolve_path(value, manifest_path: Path, *, optional: bool = False) -> str:
    if pd.isna(value) or not str(value).strip():
        if optional:
            return ""
        raise ValueError(f"empty path in manifest {manifest_path}")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return str(path.resolve())


def read_manifest(path: str | Path) -> pd.DataFrame:
    manifest_path = Path(path).expanduser().resolve()
    frame = pd.read_csv(manifest_path)
    if not {"path", "label"}.issubset(frame.columns):
        raise ValueError(f"manifest {manifest_path} must contain path,label")
    if frame.empty:
        raise ValueError(f"manifest {manifest_path} is empty")

    frame = frame.copy()
    frame["path"] = frame["path"].map(
        lambda value: _resolve_path(value, manifest_path)
    )
    frame["label"] = frame["label"].astype(str).str.strip()
    if frame["label"].eq("").any():
        raise ValueError(f"manifest {manifest_path} contains empty labels")
    for column in (
        name for name in frame.columns if name != "path" and name.endswith("_path")
    ):
        frame[column] = frame[column].map(
            lambda value: _resolve_path(value, manifest_path, optional=True)
        )
    if frame["path"].duplicated().any():
        raise ValueError(f"manifest {manifest_path} contains duplicate image paths")
    return frame.reset_index(drop=True)


@dataclass(frozen=True)
class DataConfig:
    train_path: str
    valid_path: str
    test_path: str | None
    batch_size: int
    num_workers: int


@dataclass(frozen=True)
class PreparedManifests:
    frames: dict[str, pd.DataFrame]
    labels: list[str]
    label_to_index: dict[str, int]


def prepare_manifests(config: DataConfig) -> PreparedManifests:
    frames = {
        "train": read_manifest(config.train_path),
        "valid": read_manifest(config.valid_path),
    }
    if config.test_path:
        frames["test"] = read_manifest(config.test_path)

    labels = sorted(frames["train"]["label"].unique().tolist())
    label_to_index = {label: index for index, label in enumerate(labels)}
    missing_valid = sorted(set(labels) - set(frames["valid"]["label"]))
    if missing_valid:
        raise ValueError(
            f"validation must contain every training label; missing={missing_valid}"
        )
    for split in ("valid", "test"):
        if split not in frames:
            continue
        extra = sorted(set(frames[split]["label"]) - set(labels))
        if extra:
            raise ValueError(
                f"{split} contains labels absent from training; extra={extra}"
            )
    return PreparedManifests(frames, labels, label_to_index)


# =====================================================================
# ViT model / checkpoint / attention 캡처 (<- band_vit_2/models.py)
# =====================================================================

def _imports():
    try:
        import timm
        import torch
        from timm.data import create_transform, resolve_model_data_config
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("model execution requires torch and timm") from exc
    return torch, timm, create_transform, resolve_model_data_config


def choose_device(requested: str = "auto") -> str:
    torch, _, _, _ = _imports()
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clean_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in state.items():
        cleaned = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "model."):
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):]
                    changed = True
        result[cleaned] = value
    return result


def checkpoint_payload(
    path: str | Path,
    device: str = "cpu",
    checkpoint_format: str = "auto",
) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    if checkpoint_format == "auto":
        checkpoint_format = (
            "safetensors" if checkpoint_path.suffix == ".safetensors" else "torch"
        )
    if checkpoint_format == "ckpt":
        checkpoint_format = "torch"
    if checkpoint_format == "safetensors":
        from safetensors.torch import load_file

        metadata_path = checkpoint_path.with_suffix(".json")
        payload = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else {}
        )
        payload["model_state_dict"] = load_file(str(checkpoint_path), device=device)
        return payload
    if checkpoint_format != "torch":
        raise ValueError("checkpoint_format must be auto, ckpt, torch or safetensors")

    torch, _, _, _ = _imports()
    try:
        loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        loaded = torch.load(checkpoint_path, map_location=device)
    if isinstance(loaded, dict):
        return loaded
    return {"model_state_dict": loaded}


def state_from_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    if payload and all(hasattr(value, "shape") for value in payload.values()):
        return payload
    raise ValueError("checkpoint has no model state mapping")


def _load_compatible_weights(model, payload: Mapping[str, Any]) -> tuple[int, int]:
    source = clean_state_dict(state_from_payload(payload))
    candidates = [
        source,
        {key.removeprefix("backbone."): value for key, value in source.items()},
    ]
    target = model.state_dict()
    compatible = max(
        (
            {
                key: value
                for key, value in candidate.items()
                if key in target and tuple(value.shape) == tuple(target[key].shape)
            }
            for candidate in candidates
        ),
        key=len,
    )
    if not compatible:
        raise ValueError("checkpoint has no weights compatible with the model")
    model.load_state_dict(compatible, strict=False)
    return len(compatible), len(target) - len(compatible)


def build_classifier(
    architecture: str,
    num_classes: int,
    device: str,
    *,
    pretrained: bool = True,
    initial_checkpoint: str | Path | None = None,
    checkpoint_format: str = "auto",
):
    _, timm, _, _ = _imports()
    model = timm.create_model(
        architecture,
        pretrained=pretrained and not initial_checkpoint,
        num_classes=num_classes,
    )
    if initial_checkpoint:
        payload = checkpoint_payload(
            initial_checkpoint, "cpu", checkpoint_format=checkpoint_format
        )
        _load_compatible_weights(model, payload)
    return model.to(device)


def load_classifier_checkpoint(
    path: str | Path,
    device: str,
    *,
    architecture: str | None = None,
    checkpoint_format: str = "auto",
):
    _, timm, _, _ = _imports()
    payload = checkpoint_payload(path, "cpu", checkpoint_format)
    labels = [str(value) for value in payload.get("labels", [])]
    if not labels and payload.get("label_to_index"):
        labels = [
            label
            for label, _ in sorted(
                payload["label_to_index"].items(), key=lambda item: int(item[1])
            )
        ]
    resolved_architecture = architecture or payload.get("architecture")
    num_classes = int(payload.get("num_classes") or len(labels))
    if not resolved_architecture:
        raise ValueError("checkpoint has no architecture; pass --model")
    if num_classes <= 0:
        raise ValueError("checkpoint has no labels or num_classes")

    model = timm.create_model(
        resolved_architecture,
        pretrained=False,
        num_classes=num_classes,
    )
    state = clean_state_dict(state_from_payload(payload))
    missing, unexpected = model.load_state_dict(state, strict=False)
    meaningful_missing = [
        key for key in missing if not key.endswith("num_batches_tracked")
    ]
    if meaningful_missing or unexpected:
        raise ValueError(
            "checkpoint does not match classifier; "
            f"missing={meaningful_missing[:8]}, unexpected={list(unexpected)[:8]}"
        )
    metadata = {
        **{
            key: value
            for key, value in payload.items()
            if not key.endswith("state_dict")
        },
        "architecture": resolved_architecture,
        "num_classes": num_classes,
        "labels": labels,
        "checkpoint": str(Path(path).expanduser().resolve()),
    }
    return model.to(device), metadata


def create_transforms(model, input_size: int):
    _, _, _, resolve_model_data_config = _imports()
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    config = resolve_model_data_config(model)
    mean = tuple(config["mean"])
    std = tuple(config["std"])
    resize = transforms.Resize(
        (input_size, input_size), interpolation=InterpolationMode.BILINEAR
    )
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose(
        [
            SquarePad(fill=0),
            resize,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(
                degrees=10, interpolation=InterpolationMode.BILINEAR, fill=0
            ),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            SquarePad(fill=0),
            resize,
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform, mean, std


def patch_grid_size(model) -> tuple[int, int]:
    patch_embed = getattr(model, "patch_embed", None)
    grid = getattr(patch_embed, "grid_size", None)
    if grid is None:
        raise TypeError(
            "band attention requires a ViT-style model with patch_embed.grid_size"
        )
    if isinstance(grid, int):
        return grid, grid
    if len(grid) != 2:
        raise ValueError(f"expected a 2D patch grid, got {grid!r}")
    return int(grid[0]), int(grid[1])


def attach_attention_capture(model, num_blocks: int = 6):
    blocks = getattr(model, "blocks", None)
    if blocks is None or not len(blocks):
        raise TypeError(
            "band attention requires a ViT-style model with transformer blocks"
        )
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")

    for handle in getattr(model, "_band_attention_handles", []):
        handle.remove()
    stores: list[dict[str, Any]] = []
    handles = []
    for block in blocks[-min(num_blocks, len(blocks)):]:
        attention = getattr(block, "attn", None)
        attention_drop = getattr(attention, "attn_drop", None)
        if attention_drop is None:
            raise TypeError(
                "band attention requires transformer blocks with attn.attn_drop"
            )
        if hasattr(attention, "fused_attn"):
            attention.fused_attn = False
        store: dict[str, Any] = {}

        def capture(_module, inputs, _output, *, target=store):
            target["attention"] = inputs[0]

        handles.append(attention_drop.register_forward_hook(capture))
        stores.append(store)
    model._band_attention_stores = stores
    model._band_attention_handles = handles
    return model


def get_cls_patch_attention(model):
    torch, _, _, _ = _imports()
    stores = getattr(model, "_band_attention_stores", None)
    if not stores:
        raise RuntimeError("attention capture is not attached to this model")
    prefix_tokens = int(getattr(model, "num_prefix_tokens", 1))
    values = []
    for store in stores:
        attention = store.get("attention")
        if attention is None:
            raise RuntimeError("run a model forward pass before reading attention")
        if attention.ndim != 4:
            raise ValueError(
                f"expected attention [batch, heads, tokens, tokens], got {attention.shape}"
            )
        values.append(attention.mean(dim=1)[:, 0, prefix_tokens:])
    result = torch.stack(values).mean(dim=0)
    return result / result.sum(dim=1, keepdim=True).clamp_min(1e-8)


# =====================================================================
# 학습 objective (<- band_vit_2/losses.py)
# =====================================================================

@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float = float("nan")
    classification_loss: float = float("nan")
    band_loss: float = float("nan")
    macro_precision: float = float("nan")
    macro_recall: float = float("nan")
    macro_f1: float = float("nan")
    weighted_precision: float = float("nan")
    weighted_recall: float = float("nan")
    weighted_f1: float = float("nan")


class CrossEntropyStrategy:
    """Standard classifier objective (masked_vit은 band loss를 쓰지 않음)."""

    checkpoint_adapter = "classifier_checkpoint"

    def __init__(
        self,
        *,
        label_smoothing: float = 0.0,
        band_loss_weight: float = 0.0,
        band_alpha: float = 4.0,
        **_,
    ) -> None:
        import torch

        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.band_loss_weight = band_loss_weight
        self.band_alpha = band_alpha

    def run_epoch(self, model, loader, device: str, optimizer=None) -> EpochMetrics:
        import torch
        import torch.nn.functional as functional
        from sklearn.metrics import precision_recall_fscore_support

        training = optimizer is not None
        model.train(training)
        total_loss = 0.0
        total_classification_loss = 0.0
        total_band_loss = 0.0
        total = 0
        correct = 0
        used_band = False
        all_targets: list[int] = []
        all_predictions: list[int] = []

        for batch in loader:
            if len(batch) == 5:
                images, targets, _, _, band_masks = batch
            else:
                images, targets, _, _ = batch
                band_masks = None
            images = images.to(device)
            targets = targets.to(device)
            known = targets >= 0
            if not known.any():
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                logits = model(images[known])
                known_targets = targets[known]
                classification_loss = self.criterion(logits, known_targets)
                band_loss = torch.zeros((), device=device)
                if training and self.band_loss_weight > 0:
                    if band_masks is None:
                        raise ValueError(
                            "band_loss_weight > 0 requires band masks in the training loader"
                        )
                    band_masks = band_masks.to(device).float()[known]
                    attention = get_cls_patch_attention(model).float()
                    if attention.shape != band_masks.shape:
                        raise ValueError(
                            "attention/mask patch count mismatch: "
                            f"attention={tuple(attention.shape)}, "
                            f"mask={tuple(band_masks.shape)}"
                        )
                    prior = 1.0 + self.band_alpha * band_masks
                    prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    band_loss = functional.kl_div(
                        attention.clamp_min(1e-8).log(),
                        prior,
                        reduction="batchmean",
                    )
                    used_band = True
                loss = classification_loss + self.band_loss_weight * band_loss
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

            predictions = logits.argmax(dim=1)
            count = int(known.sum().item())
            total_loss += float(loss.detach()) * count
            total_classification_loss += float(classification_loss.detach()) * count
            total_band_loss += float(band_loss.detach()) * count
            total += count
            correct += int(predictions.eq(known_targets).sum())
            all_targets.extend(known_targets.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())

        if all_targets:
            macro = precision_recall_fscore_support(
                all_targets,
                all_predictions,
                average="macro",
                zero_division=0,
            )
            weighted = precision_recall_fscore_support(
                all_targets,
                all_predictions,
                average="weighted",
                zero_division=0,
            )
        else:
            macro = weighted = (float("nan"),) * 4

        return EpochMetrics(
            loss=total_loss / max(total, 1),
            accuracy=correct / max(total, 1),
            classification_loss=total_classification_loss / max(total, 1),
            band_loss=(
                total_band_loss / max(total, 1) if used_band else float("nan")
            ),
            macro_precision=float(macro[0]),
            macro_recall=float(macro[1]),
            macro_f1=float(macro[2]),
            weighted_precision=float(weighted[0]),
            weighted_recall=float(weighted[1]),
            weighted_f1=float(weighted[2]),
        )


# =====================================================================
# optimizer / scheduler / seed / plot (<- band_vit_2/training.py)
# =====================================================================

OPTIMIZERS = ("adamw", "sgd")
SCHEDULERS = ("cosine", "step", "none")


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


def create_optimizer(config, parameters):
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


def create_scheduler(config, optimizer):
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


# =====================================================================
# 분류 metric / 시각화 (<- band_vit_2/evaluation.py)
# =====================================================================

def _classification_metrics(
    targets: list[int],
    predictions: list[int],
    labels: list[str],
) -> tuple[dict, pd.DataFrame, np.ndarray, dict]:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    indices = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=indices,
        zero_division=0,
    )
    macro = precision_recall_fscore_support(
        targets, predictions, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        targets, predictions, average="weighted", zero_division=0
    )
    accuracy = float(np.mean(np.asarray(targets) == np.asarray(predictions)))
    metrics = {
        "samples": len(targets),
        "accuracy": accuracy,
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
    }
    per_class = pd.DataFrame(
        {
            "label": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
    )
    matrix = confusion_matrix(targets, predictions, labels=indices)
    report = classification_report(
        targets,
        predictions,
        labels=indices,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return metrics, per_class, matrix, report


def _save_confusion_matrix(matrix: np.ndarray, labels: list[str], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = max(8.0, min(24.0, len(labels) * 0.55))
    fig, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )
    axis.tick_params(axis="x", labelrotation=90, labelsize=8)
    axis.tick_params(axis="y", labelsize=8)
    threshold = float(matrix.max()) / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[row, column] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned or "sample"


def _attention_method(blocks: int, prefix_tokens: int) -> dict:
    return {
        "name": "mean_cls_to_patch_self_attention",
        "display_name": "Mean CLS-to-Patch Self-Attention",
        "description": (
            "Raw ViT self-attention probabilities are taken from the CLS query "
            "to image-patch keys in the last transformer blocks. Attention is "
            "averaged across heads and then across blocks, and normalized to "
            "sum to one for each image."
        ),
        "not_attention_rollout": True,
        "not_grad_cam": True,
        "source_blocks": f"last_{blocks}_transformer_blocks",
        "head_aggregation": "arithmetic_mean",
        "block_aggregation": "arithmetic_mean",
        "query_token": "CLS token (token index 0)",
        "key_tokens": f"patch tokens after excluding {prefix_tokens} prefix token(s)",
        "patch_normalization": "sum_to_one_per_image",
        "visualization_scaling": "min_max_for_color_display_only",
        "upsampling": "bilinear_from_patch_grid_to_model_input_resolution",
        "image_basis": "square-padded and resized model input, then unnormalized",
        "same_attention_used_by_band_loss": True,
    }


def _save_attention_visualization(
    *,
    image_tensor,
    attention_tensor,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    patch_grid: tuple[int, int],
    overlay_path: Path,
    raw_path: Path,
    true_label: str,
    predicted_label: str,
    confidence: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch.nn.functional as functional

    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = image * np.asarray(std) + np.asarray(mean)
    image = np.clip(image, 0.0, 1.0)

    patch_attention = attention_tensor.detach().cpu().reshape(patch_grid)
    np.save(raw_path, patch_attention.numpy())
    heatmap = functional.interpolate(
        patch_attention.reshape(1, 1, *patch_grid),
        size=image.shape[:2],
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    low = float(heatmap.min())
    high = float(heatmap.max())
    display_heatmap = (
        (heatmap - low) / (high - low) if high > low else np.zeros_like(heatmap)
    )
    colored = plt.get_cmap("jet")(display_heatmap)[..., :3]
    overlay = np.clip(0.55 * image + 0.45 * colored, 0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_title("Model input")
    axes[1].imshow(display_heatmap, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title("CLS-to-patch attention")
    axes[2].imshow(overlay)
    axes[2].set_title("Attention overlay")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(
        f"true={true_label} | pred={predicted_label} | confidence={confidence:.4f}",
        fontsize=11,
    )
    fig.savefig(overlay_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
