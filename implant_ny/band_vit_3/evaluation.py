"""Evaluate a band-only ViT and save metrics plus attention maps."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Sequence

import pandas as pd

from band_vit_2.data import read_manifest
from band_vit_2.evaluation import (
    _attention_method,
    _classification_metrics,
    _safe_name,
    _save_attention_visualization,
    _save_confusion_matrix,
)
from band_vit_2.models import (
    attach_attention_capture,
    choose_device,
    create_transforms,
    get_cls_patch_attention,
    load_classifier_checkpoint,
    patch_grid_size,
)

from .band_preprocessing import BandCropConfig
from .data import CachedBandDataset, precompute_band_cache
from .segmentation import RuntimeSegmenter, SegmenterConfig


def evaluate_loader(
    *,
    classifier,
    loader,
    labels: list[str],
    output_dir: str | Path,
    device: str,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    attention_samples: int = -1,
    attention_blocks: int = 6,
    preprocessing_metadata: dict | None = None,
) -> Path:
    import torch

    if attention_samples < -1:
        raise ValueError("attention_samples must be -1 (all) or non-negative")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    total_samples = len(loader.dataset)
    attention_limit = total_samples if attention_samples == -1 else attention_samples
    attention_root = output / "attention_maps"
    attention_rows: list[dict] = []
    method = None
    patch_grid = None
    if attention_limit > 0:
        attach_attention_capture(classifier, attention_blocks)
        patch_grid = patch_grid_size(classifier)
        actual_blocks = min(attention_blocks, len(classifier.blocks))
        prefix_tokens = int(getattr(classifier, "num_prefix_tokens", 1))
        method = _attention_method(actual_blocks, prefix_tokens)
        method["image_basis"] = (
            "runtime SAM2-predicted boundary-band-only tight crop"
        )
        method["segmentation_mask_saved_to_disk"] = False
        method["same_attention_used_by_band_loss"] = False
        attention_root.mkdir(parents=True, exist_ok=True)
        (attention_root / "method.json").write_text(
            json.dumps(method, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    classifier.eval()
    targets: list[int] = []
    predictions: list[int] = []
    rows: list[dict] = []
    attention_saved = 0
    with torch.inference_mode():
        for images, batch_targets, batch_labels, paths in loader:
            logits = classifier(images.to(device))
            batch_attention = (
                get_cls_patch_attention(classifier).detach().cpu()
                if attention_saved < attention_limit
                else None
            )
            probabilities = torch.softmax(logits, dim=1).cpu()
            batch_predictions = probabilities.argmax(dim=1)
            targets.extend(batch_targets.tolist())
            predictions.extend(batch_predictions.tolist())

            for local_index, (path, true_label, true_index, pred_index, probability) in enumerate(
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
                    base = (
                        f"{attention_saved:03d}_{_safe_name(Path(str(path)).stem)}_"
                        f"true-{_safe_name(str(true_label))}_"
                        f"pred-{_safe_name(labels[int(pred_index)])}"
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
        "device": device,
        "labels": labels,
        "input_kind": "sam2_predicted_boundary_band_only_crop",
        "attention_samples_requested": (
            "all" if attention_samples == -1 else attention_samples
        ),
        "attention_samples_saved": attention_saved,
        "attention_map_method": method,
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
    pd.DataFrame(rows).to_csv(output / "predictions.csv", index=False)
    per_class.to_csv(output / "per_class_metrics.csv", index=False)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        output / "confusion_matrix.csv"
    )
    _save_confusion_matrix(matrix, labels, output / "confusion_matrix.png")
    if attention_rows:
        pd.DataFrame(attention_rows).to_csv(attention_root / "index.csv", index=False)
    print(
        f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"weighted_f1={metrics['weighted_f1']:.4f} samples={metrics['samples']} "
        f"attention_maps={attention_saved}",
        flush=True,
    )
    return output


def _config_from_metadata(cls, payload: dict, overrides: dict | None = None):
    names = {field.name for field in fields(cls)}
    values = {name: payload[name] for name in names if name in payload}
    for name, value in (overrides or {}).items():
        if value is not None:
            values[name] = value
    return cls(**values)


def evaluate_classifier(
    *,
    checkpoint: str | Path,
    manifest: str | Path,
    output_dir: str | Path,
    checkpoint_format: str = "auto",
    batch_size: int = 32,
    num_workers: int = 0,
    device: str = "auto",
    attention_samples: int = -1,
    attention_blocks: int = 6,
    sam_model_checkpoint: str | None = None,
) -> Path:
    import torch

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
        {"model_checkpoint": sam_model_checkpoint},
    )
    band_config = _config_from_metadata(BandCropConfig, preprocessing["band_crop"])
    frame = read_manifest(manifest)
    extra = sorted(set(frame["label"]) - set(labels))
    if extra:
        raise ValueError(f"manifest labels absent from checkpoint: {extra}")
    output = Path(output_dir).expanduser().resolve()
    segmenter = RuntimeSegmenter(segmenter_config, resolved_device)
    try:
        caches, _ = precompute_band_cache(
            {"evaluation": frame}, segmenter, band_config, output / "preprocessing"
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
    result = evaluate_loader(
        classifier=classifier,
        loader=loader,
        labels=labels,
        output_dir=output,
        device=resolved_device,
        mean=mean,
        std=std,
        attention_samples=attention_samples,
        attention_blocks=attention_blocks,
        preprocessing_metadata=preprocessing,
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
        "--checkpoint-format", choices=["auto", "ckpt", "safetensors"], default="auto"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attention-samples", type=int, default=-1)
    parser.add_argument("--attention-blocks", type=int, default=6)
    parser.add_argument("--sam-model-checkpoint")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = evaluate_classifier(
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        output_dir=args.output_dir,
        checkpoint_format=args.checkpoint_format,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        attention_samples=args.attention_samples,
        attention_blocks=args.attention_blocks,
        sam_model_checkpoint=args.sam_model_checkpoint,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

