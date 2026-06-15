#!/usr/bin/env python3
"""
Convenience runner for 6-label SupCon ViT metric training.
No venv. No YAML. No bash. Run directly in RunPod Web Terminal.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import build_manifest, detect_layout, load_manifest, make_stratified_splits, print_manifest_summary
from src.metric_learning import MetricArgs, train_metric


def _normalize_labels(labels: list[str] | None) -> list[str] | None:
    if labels is None:
        return None
    return [str(label) for label in labels]


def _filter_labels(df: pd.DataFrame, labels: list[str] | None, exclude_labels: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["label"].astype(str)
    if labels:
        allowed = set(labels)
        df = df[df["label"].isin(allowed)].copy()
        missing = sorted(allowed - set(df["label"].unique().tolist()))
        if missing:
            raise RuntimeError(f"Requested labels not found in manifest: {missing}")
    if exclude_labels:
        blocked = set(exclude_labels)
        df = df[~df["label"].isin(blocked)].copy()
    if df.empty:
        raise RuntimeError("No rows left after label filtering.")
    return df.reset_index(drop=True)


def _validate_6label_manifest(df: pd.DataFrame) -> None:
    labels = sorted(df["label"].astype(str).unique().tolist())
    if len(labels) != 6:
        raise RuntimeError(
            "This runner expects exactly 6 labels after filtering. "
            f"Got {len(labels)} labels: {labels}. "
            "Use --labels <six labels> or --exclude-labels <labels to drop>."
        )
    if "split" not in df.columns:
        raise RuntimeError("Manifest must include a split column.")
    split_counts = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    if "train" not in split_counts.index or "valid" not in split_counts.index:
        raise RuntimeError("Manifest must contain at least train and valid splits.")
    missing_train = sorted(label for label in labels if int(split_counts.loc["train"].get(label, 0)) == 0)
    missing_valid = sorted(label for label in labels if int(split_counts.loc["valid"].get(label, 0)) == 0)
    if missing_train or missing_valid:
        raise RuntimeError(
            "Each of the 6 labels must appear in train and valid splits. "
            f"missing_train={missing_train}, missing_valid={missing_valid}"
        )


def build_6label_manifest(
    data_root: Path,
    manifest: Path,
    labels: list[str] | None,
    exclude_labels: list[str],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    layout = detect_layout(data_root)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    if layout == "split_class_folder":
        df = build_manifest(data_root, manifest, layout="split_class_folder")
        df = _filter_labels(df, labels, exclude_labels)
        _validate_6label_manifest(df)
        df.to_csv(manifest, index=False)
        return df

    unsplit = manifest.with_name(manifest.stem + "_unsplit.csv")
    split_source = manifest.with_name(manifest.stem + "_filtered_unsplit.csv")
    df = build_manifest(data_root, unsplit, layout="class_folder")
    df = _filter_labels(df, labels, exclude_labels)
    df.to_csv(split_source, index=False)
    out_df = make_stratified_splits(
        split_source,
        manifest,
        train=train_ratio,
        valid=valid_ratio,
        test=test_ratio,
        seed=seed,
    )
    _validate_6label_manifest(out_df)
    return out_df


def main() -> None:
    p = argparse.ArgumentParser(description="6-label SupCon ViT metric training.")
    p.add_argument("--data-root", default="/workspace/data/large_multiclass")
    p.add_argument("--manifest", default="/workspace/data/manifests/large_multiclass_supcon_6label.csv")
    p.add_argument("--output-root", default="/workspace/implant_outputs")
    p.add_argument("--experiment-name", default="supcon_vit_6label")
    p.add_argument("--labels", nargs="+", default=None, help="Optional exact six labels to keep.")
    p.add_argument("--exclude-labels", nargs="+", default=[], help="Optional labels to drop before training.")
    p.add_argument("--model-name", default="vit_base_patch16_224", help="Example: vit_base_patch16_224, vit_small_patch16_224")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--drop-path-rate", type=float, default=0.1)
    p.add_argument("--train-ratio", type=float, default=8)
    p.add_argument("--valid-ratio", type=float, default=1)
    p.add_argument("--test-ratio", type=float, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--rebuild-manifest", action="store_true")
    args = p.parse_args()

    data_root = Path(args.data_root)
    manifest = Path(args.manifest)
    labels = _normalize_labels(args.labels)
    exclude_labels = _normalize_labels(args.exclude_labels) or []

    if args.rebuild_manifest or not manifest.exists() or labels or exclude_labels:
        build_6label_manifest(
            data_root=data_root,
            manifest=manifest,
            labels=labels,
            exclude_labels=exclude_labels,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    else:
        _validate_6label_manifest(load_manifest(manifest, split=None))

    print_manifest_summary(manifest)

    run_args = MetricArgs(
        manifest=str(manifest),
        output_root=args.output_root,
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        pretrained=True,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_name="supcon",
        temperature=args.temperature,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        drop_path_rate=args.drop_path_rate,
        seed=args.seed,
        amp=True,
        patience=args.patience,
        allow_cpu=args.allow_cpu,
    )
    train_metric(run_args)


if __name__ == "__main__":
    main()
