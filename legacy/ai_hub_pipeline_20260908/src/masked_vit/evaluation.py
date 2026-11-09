"""Evaluate a band-trained ViT and save metrics plus attention maps.

Supports feeding the model band-only crops (training distribution) or the
original unmasked pixels (--input-mode original) to test
whether a band-trained model still attends to the implant boundary. When band
masks are available, per-sample attention-on-band statistics are reported.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Sequence

import pandas as pd

from .core import (
    _attention_method,
    _classification_metrics,
    _safe_name,
    _save_attention_visualization,
    _save_confusion_matrix,
    attach_attention_capture,
    choose_device,
    create_transforms,
    get_cls_patch_attention,
    load_classifier_checkpoint,
    patch_grid_size,
    read_manifest,
)

from .band_preprocessing import BandCropConfig, INPUT_MODES
from .data import CachedBandDataset, precompute_band_cache
from .segmentation import RuntimeSegmenter, SegmenterConfig




INPUT_KINDS = {
    "band": "sam2_predicted_boundary_band_only_crop",
    "original": "original_pixels_band_bbox_crop",
}


def _patch_band_fractions(band_mask: "np.ndarray", patch_grid) -> "np.ndarray":
    """Per-patch fraction of boundary-band pixels, row-major over the grid."""
    import cv2
    import numpy as np

    grid_h, grid_w = patch_grid
    fractions = cv2.resize(
        (band_mask > 127).astype(np.float32),
        (grid_w, grid_h),
        interpolation=cv2.INTER_AREA,
    )
    return fractions.reshape(-1)


def band_focus_stats(attention: "np.ndarray", band_fractions: "np.ndarray") -> dict:
    """How much attention mass lies on the boundary band, vs. uniform chance.

    focus_gain > 1 means the model concentrates on the band more than a
    uniform attention distribution would.
    """
    import numpy as np

    attention = np.asarray(attention, dtype=np.float64).reshape(-1)
    total = float(attention.sum())
    if total <= 0 or attention.size != band_fractions.size:
        return {}
    weights = attention / total
    band_mass = float(np.dot(weights, band_fractions))
    band_fraction = float(band_fractions.mean())
    return {
        "attention_band_mass": band_mass,
        "band_patch_fraction": band_fraction,
        "focus_gain": band_mass / band_fraction if band_fraction > 0 else float("nan"),
    }


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
    band_masks: "list | None" = None,
    input_kind: str = INPUT_KINDS["band"],
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
        method["image_basis"] = input_kind
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
    focus_records: list[dict] = []
    score_chunks: list = []           # Stage 5 fuse용 [N, C] 확률 행렬
    score_paths: list[str] = []
    attention_saved = 0
    sample_offset = 0
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
            score_chunks.append(probabilities.numpy())
            score_paths.extend(str(p) for p in paths)

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
                    if band_masks is not None and patch_grid is not None:
                        dataset_index = sample_offset + local_index
                        if 0 <= dataset_index < len(band_masks):
                            focus = band_focus_stats(
                                batch_attention[local_index].numpy(),
                                _patch_band_fractions(
                                    band_masks[dataset_index], patch_grid
                                ),
                            )
                            row.update(focus)
                            if focus:
                                focus_records.append(focus)
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
            sample_offset += len(images)

    metrics, per_class, matrix, report = _classification_metrics(
        targets, predictions, labels
    )
    band_focus_summary = None
    if focus_records:
        import numpy as _np

        masses = _np.array([r["attention_band_mass"] for r in focus_records])
        gains = _np.array(
            [r["focus_gain"] for r in focus_records if _np.isfinite(r["focus_gain"])]
        )
        fractions = _np.array([r["band_patch_fraction"] for r in focus_records])
        band_focus_summary = {
            "samples": len(focus_records),
            "attention_band_mass_mean": float(masses.mean()),
            "attention_band_mass_median": float(_np.median(masses)),
            "band_patch_fraction_mean": float(fractions.mean()),
            "focus_gain_mean": float(gains.mean()) if gains.size else None,
            "focus_gain_median": float(_np.median(gains)) if gains.size else None,
            "focus_gain_gt_1_ratio": (
                float((gains > 1.0).mean()) if gains.size else None
            ),
            "definition": (
                "attention_band_mass = share of CLS->patch attention on band "
                "patches; focus_gain = band mass / band area fraction "
                "(uniform-attention baseline); gain > 1 means the model "
                "concentrates on the boundary band"
            ),
        }
    metrics_payload = {
        **metrics,
        "device": device,
        "labels": labels,
        "input_kind": input_kind,
        "band_focus": band_focus_summary,
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
    # Stage 5 fuse 입력: [N, C] 클래스별 확률 (클수록 좋음)
    if score_chunks:
        import numpy as _np

        score_frame = pd.DataFrame(_np.concatenate(score_chunks, axis=0), columns=labels)
        score_frame.insert(0, "path", score_paths)
        score_frame.to_csv(output / "scores.csv", index=False)
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
        "--checkpoint-format", choices=["auto", "ckpt", "safetensors"], default="auto"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attention-samples", type=int, default=-1)
    parser.add_argument("--attention-blocks", type=int, default=6)
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
        sam2_config=args.sam2_config,
        sam2_checkpoint=args.sam2_checkpoint,
        input_mode=args.input_mode,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

