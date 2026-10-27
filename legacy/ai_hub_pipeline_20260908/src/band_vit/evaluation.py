"""Evaluate a fine-tuned ViT classifier directly on manifest images."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .data import ManifestDataset, read_manifest
from .models import (
    attach_attention_capture,
    choose_device,
    create_transforms,
    get_cls_patch_attention,
    load_classifier_checkpoint,
    patch_grid_size,
)


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


def evaluate_classifier(
    *,
    checkpoint: str | Path,
    manifest: str | Path,
    output_dir: str | Path,
    model: str | None = None,
    checkpoint_format: str = "auto",
    input_size: int | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = "auto",
    attention_samples: int = -1,
    attention_blocks: int = 6,
) -> Path:
    import torch

    if attention_samples < -1:
        raise ValueError("attention_samples must be -1 (all) or non-negative")
    if attention_blocks <= 0:
        raise ValueError("attention_blocks must be positive")

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolved_device = choose_device(device)
    classifier, metadata = load_classifier_checkpoint(
        checkpoint,
        resolved_device,
        architecture=model,
        checkpoint_format=checkpoint_format,
    )
    frame = read_manifest(manifest)
    attention_limit = len(frame) if attention_samples == -1 else attention_samples
    labels = list(metadata.get("labels", []))
    if not labels:
        labels = sorted(frame["label"].unique().tolist())
    if len(labels) != int(metadata["num_classes"]):
        raise ValueError(
            f"checkpoint expects {metadata['num_classes']} classes but labels={labels}"
        )
    extra = sorted(set(frame["label"]) - set(labels))
    if extra:
        raise ValueError(f"manifest contains labels absent from checkpoint: {extra}")

    resolved_input_size = int(input_size or metadata.get("input_size") or 224)
    _, transform, mean, std = create_transforms(classifier, resolved_input_size)
    label_to_index = {label: index for index, label in enumerate(labels)}
    dataset = ManifestDataset(frame, label_to_index, transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    attention_root = output / "attention_maps"
    attention_rows: list[dict] = []
    method: dict | None = None
    patch_grid: tuple[int, int] | None = None
    if attention_limit > 0:
        attach_attention_capture(classifier, attention_blocks)
        patch_grid = patch_grid_size(classifier)
        actual_blocks = min(attention_blocks, len(classifier.blocks))
        prefix_tokens = int(getattr(classifier, "num_prefix_tokens", 1))
        method = _attention_method(actual_blocks, prefix_tokens)
        attention_root.mkdir(parents=True, exist_ok=True)
        (attention_root / "method.json").write_text(
            json.dumps(method, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    classifier.eval()
    targets: list[int] = []
    predictions: list[int] = []
    rows: list[dict] = []
    score_chunks: list = []           # Stage 5 fuse용 [N, C] 확률 행렬
    score_paths: list[str] = []
    attention_saved = 0
    with torch.inference_mode():
        for images, batch_targets, batch_labels, paths in loader:
            device_images = images.to(resolved_device)
            logits = classifier(device_images)
            batch_attention = (
                get_cls_patch_attention(classifier).detach().cpu()
                if attention_saved < attention_limit
                else None
            )
            probabilities = torch.softmax(logits, dim=1).cpu()
            batch_predictions = probabilities.argmax(dim=1)
            targets.extend(batch_targets.tolist())
            predictions.extend(batch_predictions.tolist())
            score_chunks.append(probabilities.numpy())
            score_paths.extend(str(p) for p in paths)

            for local_index, (
                path,
                true_label,
                true_index,
                pred_index,
                probability,
            ) in enumerate(
                zip(
                    paths,
                    batch_labels,
                    batch_targets.tolist(),
                    batch_predictions.tolist(),
                    probabilities.numpy(),
                )
            ):
                confidence = float(probability[pred_index])
                row = {
                    "path": str(path),
                    "true_index": int(true_index),
                    "true_label": str(true_label),
                    "pred_index": int(pred_index),
                    "pred_label": labels[int(pred_index)],
                    "confidence": confidence,
                    "correct": bool(true_index == pred_index),
                }
                if (
                    batch_attention is not None
                    and attention_saved < attention_limit
                    and patch_grid is not None
                ):
                    stem = _safe_name(Path(str(path)).stem)
                    true_name = _safe_name(str(true_label))
                    pred_name = _safe_name(labels[int(pred_index)])
                    base = (
                        f"{attention_saved:03d}_{stem}_"
                        f"true-{true_name}_pred-{pred_name}"
                    )
                    overlay_path = attention_root / f"{base}_overlay.png"
                    raw_path = attention_root / f"{base}_patch_attention.npy"
                    _save_attention_visualization(
                        image_tensor=images[local_index],
                        attention_tensor=batch_attention[local_index],
                        mean=mean,
                        std=std,
                        patch_grid=patch_grid,
                        overlay_path=overlay_path,
                        raw_path=raw_path,
                        true_label=str(true_label),
                        predicted_label=labels[int(pred_index)],
                        confidence=confidence,
                    )
                    row["attention_overlay"] = str(overlay_path)
                    row["attention_raw"] = str(raw_path)
                    attention_rows.append(
                        {
                            "sample_index": attention_saved,
                            "path": str(path),
                            "true_label": str(true_label),
                            "pred_label": labels[int(pred_index)],
                            "confidence": confidence,
                            "correct": bool(true_index == pred_index),
                            "overlay_path": str(overlay_path),
                            "raw_attention_path": str(raw_path),
                        }
                    )
                    attention_saved += 1
                rows.append(row)

    metrics, per_class, matrix, report = _classification_metrics(
        targets, predictions, labels
    )
    metrics_payload = {
        **metrics,
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "manifest": str(Path(manifest).expanduser().resolve()),
        "device": resolved_device,
        "input_size": resolved_input_size,
        "labels": labels,
        "attention_samples_requested": (
            "all" if attention_samples == -1 else attention_samples
        ),
        "attention_samples_saved": attention_saved,
        "attention_map_method": method,
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(output / "predictions.csv", index=False)
    per_class.to_csv(output / "per_class_metrics.csv", index=False)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        output / "confusion_matrix.csv"
    )
    _save_confusion_matrix(matrix, labels, output / "confusion_matrix.png")
    # Stage 5 fuse 입력: [N, C] 클래스별 확률 (클수록 좋음)
    if score_chunks:
        score_frame = pd.DataFrame(np.concatenate(score_chunks, axis=0), columns=labels)
        score_frame.insert(0, "path", score_paths)
        score_frame.to_csv(output / "scores.csv", index=False)
    if attention_rows:
        pd.DataFrame(attention_rows).to_csv(
            attention_root / "index.csv", index=False
        )
    print(
        f"accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"weighted_f1={metrics['weighted_f1']:.4f} "
        f"samples={metrics['samples']} "
        f"attention_maps={attention_saved}"
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", help="override checkpoint architecture")
    parser.add_argument(
        "--checkpoint-format",
        choices=["auto", "ckpt", "safetensors"],
        default="auto",
    )
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--attention-samples",
        type=int,
        default=-1,
        help="save attention maps: -1=all rows, 0=none, N=first N rows",
    )
    parser.add_argument(
        "--attention-blocks",
        type=int,
        default=6,
        help="number of final ViT transformer blocks to average",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = evaluate_classifier(
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        output_dir=args.output_dir,
        model=args.model,
        checkpoint_format=args.checkpoint_format,
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        attention_samples=args.attention_samples,
        attention_blocks=args.attention_blocks,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
