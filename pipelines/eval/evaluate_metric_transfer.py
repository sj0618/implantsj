#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import create_loader, label_mapping, load_manifest
from src.metric_learning import extract_embeddings, prototype_predict
from src.metrics import compute_metrics, save_confusion_matrix, save_metrics_json, save_predictions
from src.models import MetricModel, unwrap_state_dict
from src.utils import write_json


def _normalize_state_dict(state: dict) -> dict:
    normalized = {}
    for key, value in state.items():
        clean_key = str(key)
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module."):]
        if clean_key.startswith("model."):
            clean_key = clean_key[len("model."):]
        normalized[clean_key] = value
    return normalized


def load_metric_checkpoint(
    checkpoint: Path,
    device: torch.device,
    model_name: str | None = None,
    image_size: int | None = None,
    embedding_dim: int | None = None,
    dropout: float | None = None,
    drop_path_rate: float | None = None,
) -> Tuple[MetricModel, int, dict]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) or {}

    resolved_model_name = model_name or ckpt_args.get("model_name") or "vit_base_patch16_224"
    resolved_image_size = int(image_size or ckpt_args.get("image_size") or 224)
    resolved_embedding_dim = int(embedding_dim or ckpt_args.get("embedding_dim") or 128)
    resolved_dropout = float(dropout if dropout is not None else ckpt_args.get("dropout", 0.0))
    resolved_drop_path_rate = float(
        drop_path_rate if drop_path_rate is not None else ckpt_args.get("drop_path_rate", 0.0)
    )

    model = MetricModel(
        model_name=resolved_model_name,
        embedding_dim=resolved_embedding_dim,
        pretrained=False,
        dropout=resolved_dropout,
        drop_path_rate=resolved_drop_path_rate,
    )
    state = _normalize_state_dict(unwrap_state_dict(ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(device).eval()

    metadata = {
        "checkpoint": str(checkpoint),
        "model_name": resolved_model_name,
        "image_size": resolved_image_size,
        "embedding_dim": resolved_embedding_dim,
        "dropout": resolved_dropout,
        "drop_path_rate": resolved_drop_path_rate,
        "loaded_class_to_idx": {str(k): int(v) for k, v in (ckpt.get("class_to_idx", {}) or {}).items()},
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }
    return model, resolved_image_size, metadata


def evaluate_transfer_split(
    model,
    manifest: Path,
    class_to_idx: Dict[str, int],
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    support_split: str,
    query_split: str,
    out_dir: Path | None = None,
):
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    support_loader = create_loader(
        manifest_csv=manifest,
        split=support_split,
        class_to_idx=class_to_idx,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        train=False,
        strong_aug=False,
        return_path=True,
    )
    query_loader = create_loader(
        manifest_csv=manifest,
        split=query_split,
        class_to_idx=class_to_idx,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        train=False,
        strong_aug=False,
        return_path=True,
    )

    support_z, support_y, _ = extract_embeddings(model, support_loader, device)
    query_z, query_y, paths = extract_embeddings(model, query_loader, device)
    pred, conf = prototype_predict(support_z, support_y, query_z)
    metrics = compute_metrics(query_y, pred, class_names)

    if out_dir is not None:
        save_metrics_json(out_dir / "metrics" / f"{query_split}_prototype_metrics.json", metrics)
        save_confusion_matrix(
            out_dir / "confusion_matrices" / f"{query_split}_prototype.csv",
            out_dir / "confusion_matrices" / f"{query_split}_prototype.png",
            query_y,
            pred,
            class_names,
        )
        rows = []
        for p, t, pr, cf in zip(paths, query_y, pred, conf):
            rows.append(
                {
                    "path": p,
                    "true_idx": int(t),
                    "true_label": idx_to_class[int(t)],
                    "pred_idx": int(pr),
                    "pred_label": idx_to_class[int(pr)],
                    "similarity": float(cf),
                    "correct": bool(int(t) == int(pr)),
                }
            )
        save_predictions(out_dir / "metrics" / f"{query_split}_prototype_predictions.csv", rows)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a metric-learning checkpoint on a different manifest (e.g. seen-27 train -> unseen-15 eval)."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True, help="Held-out manifest with support/query splits.")
    parser.add_argument("--support-split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--query-splits", nargs="+", default=["valid", "test"], choices=["train", "valid", "test"])
    parser.add_argument("--output-root", default="/workspace/implant_outputs")
    parser.add_argument("--experiment-name", default="metric_transfer_eval")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--drop-path-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--require-disjoint",
        action="store_true",
        help="Fail if eval labels overlap with checkpoint train labels.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA is not available. Pass --allow-cpu only for debugging.")

    manifest = Path(args.manifest)
    eval_df = load_manifest(manifest, split=None)
    eval_class_to_idx, _ = label_mapping(eval_df)
    eval_labels = sorted(eval_class_to_idx.keys())

    model, image_size, metadata = load_metric_checkpoint(
        checkpoint=Path(args.checkpoint),
        device=device,
        model_name=args.model_name,
        image_size=args.image_size,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        drop_path_rate=args.drop_path_rate,
    )

    train_labels = sorted(metadata["loaded_class_to_idx"].keys())
    overlap = sorted(set(eval_labels) & set(train_labels))
    if args.require_disjoint and overlap:
        raise RuntimeError(f"Eval labels overlap with checkpoint train labels: {overlap}")

    run_dir = Path(args.output_root) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.experiment_name}"
    for sub in ["metrics", "confusion_matrices"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    counts_df = eval_df.groupby(["split", "label"]).size().reset_index(name="count")
    metadata.update(
        {
            "device": str(device),
            "manifest": str(manifest),
            "support_split": args.support_split,
            "query_splits": list(args.query_splits),
            "eval_class_to_idx": eval_class_to_idx,
            "eval_labels": eval_labels,
            "train_labels_from_checkpoint": train_labels,
            "overlap_labels": overlap,
            "sample_counts": [
                {"split": str(row.split), "label": str(row.label), "count": int(row.count)}
                for row in counts_df.itertuples(index=False)
            ],
        }
    )
    write_json(run_dir / "args.json", metadata)

    results = {}
    for split in args.query_splits:
        results[split] = evaluate_transfer_split(
            model=model,
            manifest=manifest,
            class_to_idx=eval_class_to_idx,
            image_size=image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            support_split=args.support_split,
            query_split=split,
            out_dir=run_dir,
        )

    print(f"RUN_DIR={run_dir}")
    for split, metrics in results.items():
        print(
            f"{split}: accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} weighted_f1={metrics['weighted_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
