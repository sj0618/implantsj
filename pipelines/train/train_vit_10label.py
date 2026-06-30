#!/usr/bin/env python3
"""
Convenience runner for combined 3-label + 7-label ViT supervised training.
Supports standard train/valid/test training and stratified KFold training.
No venv. No YAML. No bash. Run directly in RunPod Web Terminal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import print_manifest_summary
from src.kfold import build_stratified_kfold_manifests
from src.supervised import SupervisedArgs, train_supervised
from train_vit_3label import DEFAULT_LABELS as LABELS_3, build_3label_manifest
from train_vit_7label import DEFAULT_LABELS as LABELS_7, build_filtered_split_manifest

DEFAULT_DATA_ROOT_3 = ROOT / "data" / "3label"
DEFAULT_DATA_ROOT_7 = ROOT / "data" / "7label"
DEFAULT_LABELS = [*LABELS_3, *LABELS_7]


def build_10label_manifest(
    data_root_3: Path,
    data_root_7: Path,
    manifest: Path,
    labels_3: list[str],
    labels_7: list[str],
) -> None:
    tmp_dir = manifest.parent
    tmp_3 = tmp_dir / "_tmp_plain_vit_10label_3label.csv"
    tmp_7 = tmp_dir / "_tmp_plain_vit_10label_7label.csv"

    build_3label_manifest(data_root_3, tmp_3, labels_3)
    build_filtered_split_manifest(data_root_7, tmp_7, labels_7)

    df = pd.concat([pd.read_csv(tmp_3), pd.read_csv(tmp_7)], ignore_index=True)
    labels = [*labels_3, *labels_7]
    expected = sorted(labels)
    actual = sorted(df["label"].astype(str).unique().tolist())
    if actual != expected:
        raise RuntimeError(f"Label mismatch. expected={expected}, actual={actual}")

    duplicates = df["path"].astype(str).duplicated()
    if bool(duplicates.any()):
        dup_paths = df.loc[duplicates, "path"].head(10).tolist()
        raise RuntimeError(f"Duplicate image paths found in combined manifest: {dup_paths}")

    split_counts = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    missing = {
        split: sorted(label for label in labels if int(split_counts.loc[split].get(label, 0)) == 0)
        for split in ["train", "valid", "test"]
        if split in split_counts.index
    }
    missing = {split: split_labels for split, split_labels in missing.items() if split_labels}
    if missing:
        raise RuntimeError(f"Each label must appear in every split. missing={missing}")

    manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest, index=False)
    for tmp in (tmp_3, tmp_7):
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _make_run_args(args: argparse.Namespace, manifest: Path, experiment_name: str, seed: int) -> SupervisedArgs:
    return SupervisedArgs(
        manifest=str(manifest),
        output_root=args.output_root,
        experiment_name=experiment_name,
        model_name=args.model_name,
        pretrained=args.pretrained,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        dropout=args.dropout,
        drop_path_rate=args.drop_path_rate,
        seed=seed,
        amp=True,
        patience=args.patience,
        monitor="macro_f1",
        use_class_weights=args.use_class_weights,
        strong_aug=not args.no_strong_aug,
        allow_cpu=args.allow_cpu,
    )


def _load_best_metrics(run_dir: Path) -> dict[str, object]:
    metrics_path = run_dir / "metrics" / "best_valid_metrics.json"
    if not metrics_path.is_file():
        return {}
    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {f"best_valid_{key}": value for key, value in data.items() if isinstance(value, (int, float, str, bool))}


def _run_kfold(args: argparse.Namespace, manifest: Path) -> None:
    df = pd.read_csv(manifest)
    fold_root = manifest.parent / f"{manifest.stem}_kfold{args.kfold}_seed{args.seed}"
    fold_manifests, fold_summary = build_stratified_kfold_manifests(
        df=df,
        out_dir=fold_root,
        base_name=manifest.stem,
        k=args.kfold,
        seed=args.seed,
    )
    print(f"KFold manifests written to: {fold_root}")
    print(fold_summary.pivot_table(index=["fold", "split"], columns="label", values="count", fill_value=0))

    if args.dry_run:
        print("Dry run complete; no training was started.")
        return

    result_rows: list[dict[str, object]] = []
    for fold_idx, fold_manifest in enumerate(fold_manifests, start=1):
        experiment_name = f"{args.experiment_name}_kfold{args.kfold}_fold{fold_idx:02d}"
        fold_seed = args.seed + fold_idx - 1
        print("\n" + "=" * 80)
        print(f"Starting supervised KFold {fold_idx}/{args.kfold}: {experiment_name}")
        print_manifest_summary(fold_manifest)
        run_dir = train_supervised(_make_run_args(args, fold_manifest, experiment_name, fold_seed))
        row: dict[str, object] = {
            "fold": fold_idx,
            "manifest": str(fold_manifest),
            "run_dir": str(run_dir),
            "seed": fold_seed,
        }
        row.update(_load_best_metrics(run_dir))
        result_rows.append(row)
        results_path = Path(args.output_root) / f"{args.experiment_name}_kfold{args.kfold}_results.csv"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result_rows).to_csv(results_path, index=False)
        print(f"Updated KFold results: {results_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Plain ViT baseline for combined 10-label implant classification.")
    p.add_argument("--data-root-3", default=str(DEFAULT_DATA_ROOT_3))
    p.add_argument("--data-root-7", default=str(DEFAULT_DATA_ROOT_7))
    p.add_argument("--manifest", default=str(ROOT / "data" / "manifests" / "plain_vit_10label.csv"))
    p.add_argument("--output-root", default=str(ROOT / "outputs"))
    p.add_argument("--experiment-name", default="plain_vit_10label")
    p.add_argument("--labels-3", nargs="+", default=LABELS_3)
    p.add_argument("--labels-7", nargs="+", default=LABELS_7)
    p.add_argument("--model-name", default="vit_base_patch16_224", help="Example: vit_base_patch16_224, vit_small_patch16_224")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--drop-path-rate", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--use-class-weights", action="store_true", default=True)
    p.add_argument("--no-class-weights", dest="use_class_weights", action="store_false")
    p.add_argument("--no-strong-aug", action="store_true")
    p.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--kfold", type=int, default=0, help="Run stratified KFold training; use --kfold 5 for 5 folds.")
    p.add_argument("--dry-run", action="store_true", help="Build/print manifests only; do not train.")
    p.add_argument("--allow-cpu", action="store_true")
    args = p.parse_args()

    labels_3 = [str(label) for label in args.labels_3]
    labels_7 = [str(label) for label in args.labels_7]
    if len(labels_3) != 3:
        raise RuntimeError(f"This script expects exactly 3 labels in --labels-3, got {len(labels_3)}: {labels_3}")
    if len(labels_7) != 7:
        raise RuntimeError(f"This script expects exactly 7 labels in --labels-7, got {len(labels_7)}: {labels_7}")
    overlap = sorted(set(labels_3) & set(labels_7))
    if overlap:
        raise RuntimeError(f"3-label and 7-label sets must be disjoint. overlap={overlap}")

    manifest = Path(args.manifest)
    build_10label_manifest(Path(args.data_root_3), Path(args.data_root_7), manifest, labels_3, labels_7)
    print_manifest_summary(manifest)

    if args.kfold:
        _run_kfold(args, manifest)
        return

    if args.dry_run:
        print("Dry run complete; no training was started.")
        return

    train_supervised(_make_run_args(args, manifest, args.experiment_name, args.seed))


if __name__ == "__main__":
    main()
