#!/usr/bin/env python3
"""
Convenience runner for 7-label SupCon ViT metric training.
No venv. No YAML. No bash. Run directly in RunPod Web Terminal.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import IMAGE_EXTENSIONS, print_manifest_summary
from src.metric_learning import MetricArgs, train_metric

DEFAULT_LABELS = ["ADIN", "Dentium", "DIONAVI", "MIS", "NORIS", "nobel", "osstem"]


def build_filtered_split_manifest(data_root: Path, manifest: Path, labels: list[str]) -> None:
    rows: list[dict[str, str]] = []
    allowed = set(labels)

    for split in ["train", "valid", "test"]:
        split_dir = data_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        found = {
            p.name
            for p in split_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        }
        missing = sorted(allowed - found)
        extra = sorted(found - allowed)
        if missing:
            raise RuntimeError(f"{split} split is missing labels: {missing}")
        if extra:
            print(f"[warn] ignoring labels in {split}: {extra}")

        for label in labels:
            for path in sorted((split_dir / label).rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    rows.append({
                        "path": str(path.resolve()),
                        "label": label,
                        "split": split,
                    })

    if not rows:
        raise RuntimeError(f"No image files found under {data_root}")

    df = pd.DataFrame(rows)
    actual = sorted(df["label"].astype(str).unique().tolist())
    expected = sorted(labels)
    if actual != expected:
        raise RuntimeError(f"Label mismatch. expected={expected}, actual={actual}")

    manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="7-label SupCon ViT metric training.")
    p.add_argument("--data-root", default=str(ROOT / "data" / "7label"))
    p.add_argument("--manifest", default=str(ROOT / "data" / "manifests" / "supcon_vit_7label.csv"))
    p.add_argument("--output-root", default=str(ROOT / "outputs"))
    p.add_argument("--experiment-name", default="supcon_vit_7label")
    p.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    p.add_argument("--model-name", default="vit_base_patch16_224", help="Example: vit_base_patch16_224, vit_small_patch16_224")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--drop-path-rate", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--allow-cpu", action="store_true")
    args = p.parse_args()

    labels = [str(label) for label in args.labels]
    if len(labels) != 7:
        raise RuntimeError(f"This script expects exactly 7 labels, got {len(labels)}: {labels}")

    manifest = Path(args.manifest)
    build_filtered_split_manifest(Path(args.data_root), manifest, labels)
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
