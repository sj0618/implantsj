"""Generate class-specific gradient-weighted attention rollout maps.

This module is intentionally separate from the regular masked_vit evaluator so
the existing raw CLS-to-patch attention outputs remain untouched.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .core import (
    _classification_metrics,
    _safe_name,
    _save_confusion_matrix,
    attach_attention_capture,
    choose_device,
    create_transforms,
    load_classifier_checkpoint,
    patch_grid_size,
    read_manifest,
)

from .band_preprocessing import BandCropConfig, INPUT_MODES
from .data import CachedBandDataset, precompute_band_cache
from .evaluation import INPUT_KINDS, _patch_band_fractions, band_focus_stats
from .segmentation import RuntimeSegmenter, SegmenterConfig


def _config_from_metadata(cls, payload: dict, overrides: dict | None = None):
    names = {field.name for field in fields(cls)}
    values = {name: payload[name] for name in names if name in payload}
    for name, value in (overrides or {}).items():
        if value is not None:
            values[name] = value
    return cls(**values)


def _gradient_rollout_method(
    *, blocks: int, total_blocks: int, prefix_tokens: int
) -> dict:
    source = (
        f"all_{total_blocks}_transformer_blocks"
        if blocks == total_blocks
        else f"last_{blocks}_transformer_blocks"
    )
    return {
        "name": "gradient_weighted_attention_rollout",
        "display_name": "Gradient-weighted Attention Rollout",
        "description": (
            "For the predicted class logit, gradients are computed with respect "
            "to ViT self-attention probabilities. Positive attention-gradient "
            "products are averaged across heads and composed across transformer "
            "blocks with residual propagation. The CLS-to-patch relevance is "
            "normalized for visualization."
        ),
        "class_specific": True,
        "target": "predicted_class_logit",
        "source_blocks": source,
        "gradient_weighting": "positive_part(attention * d_target/d_attention)",
        "head_aggregation": "arithmetic_mean",
        "block_composition": "residual_rollout_R_equals_R_plus_CAM_times_R",
        "query_token": "CLS token (token index 0)",
        "key_tokens": f"patch tokens after excluding {prefix_tokens} prefix token(s)",
        "patch_normalization": "sum_to_one_per_image_after_rollout",
        "visualization_scaling": "min_max_for_color_display_only",
        "upsampling": "bilinear_from_patch_grid_to_model_input_resolution",
        "image_basis": "runtime SAM2-predicted boundary-band-only tight crop",
        "not_raw_cls_attention": True,
        "not_grad_cam": True,
    }


def _gradient_weighted_rollout(
    *,
    classifier,
    target_score,
    sample_index: int,
    prefix_tokens: int,
    retain_graph: bool,
):
    """Return class-specific CLS-to-patch rollout for one batch sample."""
    import torch

    stores = getattr(classifier, "_band_attention_stores", None)
    if not stores:
        raise RuntimeError("attention capture is not attached to the classifier")
    attentions = [store.get("attention") for store in stores]
    if any(attention is None for attention in attentions):
        raise RuntimeError("run a model forward pass before computing rollout")
    gradients = torch.autograd.grad(
        target_score,
        attentions,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=False,
    )

    tokens = int(attentions[0].shape[-1])
    relevance = torch.eye(
        tokens,
        dtype=attentions[0].dtype,
        device=attentions[0].device,
    )
    for attention, gradient in zip(attentions, gradients):
        if attention.ndim != 4:
            raise ValueError(
                "expected attention [batch, heads, tokens, tokens], "
                f"got {tuple(attention.shape)}"
            )
        camera = (attention[sample_index] * gradient[sample_index]).clamp(min=0)
        camera = camera.mean(dim=0)
        relevance = relevance + camera @ relevance

    patch_relevance = relevance[0, prefix_tokens:]
    patch_relevance = patch_relevance.clamp(min=0)
    return patch_relevance / patch_relevance.sum().clamp_min(1e-8)


def _save_rollout_visualization(
    *,
    image_tensor,
    rollout_tensor,
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

    patch_rollout = rollout_tensor.detach().cpu().reshape(patch_grid)
    np.save(raw_path, patch_rollout.numpy())
    heatmap = functional.interpolate(
        patch_rollout.reshape(1, 1, *patch_grid),
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
    axes[1].set_title("Gradient-weighted attention rollout")
    axes[2].imshow(overlay)
    axes[2].set_title("Class-specific rollout overlay")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(
        f"target=predicted:{predicted_label} | true={true_label} | "
        f"confidence={confidence:.4f}",
        fontsize=11,
    )
    fig.savefig(overlay_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_loader_rollouts(
    *,
    classifier,
    loader,
    labels: list[str],
    output_dir: str | Path,
    device: str,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    rollout_samples: int = -1,
    rollout_blocks: int = -1,
    preprocessing_metadata: dict | None = None,
    band_masks: "list | None" = None,
    input_kind: str = INPUT_KINDS["band"],
) -> Path:
    import torch

    if rollout_samples < -1:
        raise ValueError("rollout_samples must be -1 (all) or non-negative")
    if rollout_blocks == 0 or rollout_blocks < -1:
        raise ValueError("rollout_blocks must be -1 (all) or positive")

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    total_samples = len(loader.dataset)
    rollout_limit = total_samples if rollout_samples == -1 else rollout_samples
    total_blocks = len(classifier.blocks)
    actual_blocks = total_blocks if rollout_blocks == -1 else min(
        rollout_blocks, total_blocks
    )
    attach_attention_capture(classifier, actual_blocks)
    patch_grid = patch_grid_size(classifier)
    prefix_tokens = int(getattr(classifier, "num_prefix_tokens", 1))
    method = _gradient_rollout_method(
        blocks=actual_blocks,
        total_blocks=total_blocks,
        prefix_tokens=prefix_tokens,
    )
    (output / "method.json").write_text(
        json.dumps(method, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    classifier.eval()
    targets: list[int] = []
    predictions: list[int] = []
    rows: list[dict] = []
    focus_records: list[dict] = []
    saved = 0
    sample_offset = 0

    for images, batch_targets, batch_labels, paths in loader:
        device_images = images.to(device)
        explain_count = max(0, min(len(images), rollout_limit - saved))
        classifier.zero_grad(set_to_none=True)
        if explain_count:
            with torch.enable_grad():
                logits = classifier(device_images)
        else:
            with torch.inference_mode():
                logits = classifier(device_images)

        probabilities = torch.softmax(logits.detach(), dim=1).cpu()
        batch_predictions = probabilities.argmax(dim=1)
        targets.extend(batch_targets.tolist())
        predictions.extend(batch_predictions.tolist())

        batch_rollouts = []
        for local_index in range(explain_count):
            predicted_index = int(batch_predictions[local_index])
            batch_rollouts.append(
                _gradient_weighted_rollout(
                    classifier=classifier,
                    target_score=logits[local_index, predicted_index],
                    sample_index=local_index,
                    prefix_tokens=prefix_tokens,
                    retain_graph=local_index < explain_count - 1,
                ).detach().cpu()
            )

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
            if local_index < explain_count:
                base = (
                    f"{saved:03d}_{_safe_name(Path(str(path)).stem)}_"
                    f"true-{_safe_name(str(true_label))}_"
                    f"pred-{_safe_name(labels[int(pred_index)])}"
                )
                overlay_path = output / f"{base}_gradient_rollout_overlay.png"
                raw_path = output / f"{base}_gradient_rollout.npy"
                _save_rollout_visualization(
                    image_tensor=images[local_index],
                    rollout_tensor=batch_rollouts[local_index],
                    mean=mean,
                    std=std,
                    patch_grid=patch_grid,
                    overlay_path=overlay_path,
                    raw_path=raw_path,
                    true_label=str(true_label),
                    predicted_label=labels[int(pred_index)],
                    confidence=confidence,
                )
                row["gradient_rollout_overlay"] = str(overlay_path)
                row["gradient_rollout_raw"] = str(raw_path)
                if band_masks is not None:
                    dataset_index = sample_offset + local_index
                    if 0 <= dataset_index < len(band_masks):
                        focus = band_focus_stats(
                            batch_rollouts[local_index].numpy(),
                            _patch_band_fractions(
                                band_masks[dataset_index], patch_grid
                            ),
                        )
                        row.update(focus)
                        if focus:
                            focus_records.append(focus)
                saved += 1
            rows.append(row)
        sample_offset += len(images)

        print(
            f"gradient rollout: {min(len(targets), total_samples)}/{total_samples} "
            f"saved={saved}",
            flush=True,
        )

    metrics, per_class, matrix, report = _classification_metrics(
        targets, predictions, labels
    )
    band_focus_summary = None
    if focus_records:
        masses = np.array([r["attention_band_mass"] for r in focus_records])
        gains = np.array(
            [r["focus_gain"] for r in focus_records if np.isfinite(r["focus_gain"])]
        )
        fractions = np.array([r["band_patch_fraction"] for r in focus_records])
        band_focus_summary = {
            "samples": len(focus_records),
            "rollout_band_mass_mean": float(masses.mean()),
            "rollout_band_mass_median": float(np.median(masses)),
            "band_patch_fraction_mean": float(fractions.mean()),
            "focus_gain_mean": float(gains.mean()) if gains.size else None,
            "focus_gain_median": float(np.median(gains)) if gains.size else None,
            "focus_gain_gt_1_ratio": (
                float((gains > 1.0).mean()) if gains.size else None
            ),
            "definition": (
                "rollout_band_mass = share of gradient-rollout mass on band "
                "patches; focus_gain = band mass / band area fraction "
                "(uniform baseline); gain > 1 means the model concentrates "
                "on the boundary band"
            ),
        }
    metrics_payload = {
        **metrics,
        "device": device,
        "labels": labels,
        "input_kind": input_kind,
        "band_focus": band_focus_summary,
        "gradient_rollout_samples_requested": (
            "all" if rollout_samples == -1 else rollout_samples
        ),
        "gradient_rollout_samples_saved": saved,
        "gradient_rollout_method": method,
        "preprocessing": preprocessing_metadata,
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(output / "index.csv", index=False)
    per_class.to_csv(output / "per_class_metrics.csv", index=False)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        output / "confusion_matrix.csv"
    )
    _save_confusion_matrix(matrix, labels, output / "confusion_matrix.png")
    print(
        f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"samples={metrics['samples']} gradient_rollouts={saved}",
        flush=True,
    )
    return output


def generate_classifier_rollouts(
    *,
    checkpoint: str | Path,
    manifest: str | Path,
    output_dir: str | Path,
    checkpoint_format: str = "auto",
    batch_size: int = 1,
    num_workers: int = 0,
    device: str = "auto",
    rollout_samples: int = -1,
    rollout_blocks: int = -1,
    max_samples: int | None = None,
    sam_model_checkpoint: str | None = None,
    sam2_config: str | None = None,
    sam2_checkpoint: str | None = None,
    input_mode: str = "band",
) -> Path:
    import torch

    if input_mode not in INPUT_MODES:
        raise ValueError(f"input_mode must be one of {INPUT_MODES}")

    resolved_device = choose_device(device)
    classifier, metadata = load_classifier_checkpoint(
        checkpoint, "cpu", checkpoint_format=checkpoint_format
    )
    labels = list(metadata.get("labels", []))
    preprocessing = metadata.get("band_vit3_preprocessing")
    if not preprocessing:
        raise ValueError("checkpoint has no band_vit3_preprocessing metadata")
    segmenter_config = _config_from_metadata(
        SegmenterConfig,
        preprocessing["segmenter"],
        {"model_checkpoint": sam_model_checkpoint,
         "sam2_config": sam2_config,
         "sam2_checkpoint": sam2_checkpoint},
    )
    band_config = _config_from_metadata(BandCropConfig, preprocessing["band_crop"])
    frame = read_manifest(manifest)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        frame = frame.iloc[:max_samples].reset_index(drop=True)
    extra = sorted(set(frame["label"]) - set(labels))
    if extra:
        raise ValueError(f"manifest labels absent from checkpoint: {extra}")

    output = Path(output_dir).expanduser().resolve()
    segmenter = RuntimeSegmenter(segmenter_config, resolved_device)
    try:
        caches, mask_caches, _ = precompute_band_cache(
            {"evaluation": frame},
            segmenter,
            band_config,
            output / "preprocessing",
            input_mode=input_mode,
        )
    finally:
        segmenter.close()

    input_size = int(metadata.get("input_size") or band_config.output_size)
    _, transform, mean, std = create_transforms(classifier, input_size)
    label_to_index = {label: index for index, label in enumerate(labels)}
    dataset = CachedBandDataset(
        frame, caches["evaluation"], label_to_index, transform
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    classifier = classifier.to(resolved_device)
    result = generate_loader_rollouts(
        classifier=classifier,
        loader=loader,
        labels=labels,
        output_dir=output,
        device=resolved_device,
        mean=mean,
        std=std,
        rollout_samples=rollout_samples,
        rollout_blocks=rollout_blocks,
        preprocessing_metadata={**preprocessing, "evaluation_input_mode": input_mode},
        band_masks=mask_caches["evaluation"],
        input_kind=INPUT_KINDS[input_mode],
    )
    metrics_path = result / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["checkpoint"] = str(Path(checkpoint).expanduser().resolve())
    metrics["manifest"] = str(Path(manifest).expanduser().resolve())
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--checkpoint-format",
        choices=["auto", "ckpt", "safetensors"],
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rollout-samples", type=int, default=-1)
    parser.add_argument(
        "--rollout-blocks",
        type=int,
        default=-1,
        help="-1 uses all transformer blocks; otherwise uses the last N blocks",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--sam-model-checkpoint")
    parser.add_argument("--sam2-config",
                        help="비우면 체크포인트 메타데이터의 값을 쓴다")
    parser.add_argument("--sam2-checkpoint",
                        help="학습 때와 경로가 달라졌을 때 덮어쓴다")
    parser.add_argument(
        "--input-mode",
        choices=list(INPUT_MODES),
        default="band",
        help=(
            "band: training-style band-only crops; original: unmasked pixels "
            "in the same band bbox crop, aligned with the training input"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = generate_classifier_rollouts(
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        output_dir=args.output_dir,
        checkpoint_format=args.checkpoint_format,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        rollout_samples=args.rollout_samples,
        rollout_blocks=args.rollout_blocks,
        max_samples=args.max_samples,
        sam_model_checkpoint=args.sam_model_checkpoint,
        sam2_config=args.sam2_config,
        sam2_checkpoint=args.sam2_checkpoint,
        input_mode=args.input_mode,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
