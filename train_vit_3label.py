#!/usr/bin/env python3
"""
Convenience runner for 3-label ViT supervised training.
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

from src.data import IMAGE_EXTENSIONS, print_manifest_summary
from src.supervised import SupervisedArgs, train_supervised

DEFAULT_LABELS = ["Bego", "Bicon", "ITI"]
DEFAULT_DATA_ROOT = Path("/workspace/3label") if Path("/workspace/3label").is_dir() else ROOT / "data" / "3label"
LABEL_ALIASES = {
    "bego": "Bego",
    "Bego": "Bego",
    "Bicon": "Bicon",
    "bicon": "Bicon",
    "ITI": "ITI",
    "iti": "ITI",
}


def _canonical_label(label: str) -> str | None:
    return LABEL_ALIASES.get(str(label).strip())


def _rows_from_coco_split(split_dir: Path, split: str, allowed: set[str]) -> list[dict[str, str]]:
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.is_file():
        return []

    data = json.loads(ann_path.read_text())
    categories = {
        int(cat["id"]): _canonical_label(str(cat.get("name", "")))
        for cat in data.get("categories", [])
    }
    images = {
        int(img["id"]): str(img.get("file_name", ""))
        for img in data.get("images", [])
    }

    image_to_label: dict[int, str] = {}
    for ann in data.get("annotations", []):
        label = categories.get(int(ann.get("category_id", -1)))
        image_id = int(ann.get("image_id", -1))
        if label in allowed and image_id in images:
            image_to_label.setdefault(image_id, label)

    rows: list[dict[str, str]] = []
    for image_id, label in image_to_label.items():
        path = split_dir / images[image_id]
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            rows.append({"path": str(path.resolve()), "label": label, "split": split})
    return rows


def build_3label_manifest(data_root: Path, manifest: Path, labels: list[str]) -> None:
    rows: list[dict[str, str]] = []
    allowed = set(labels)

    for split in ["train", "valid", "test"]:
        split_dir = data_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
            label = _canonical_label(class_dir.name)
            if label not in allowed:
                print(f"[warn] ignoring label folder in {split}: {class_dir.name}")
                continue
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    rows.append({"path": str(path.resolve()), "label": label, "split": split})

        seen_direct = {Path(row["path"]).name for row in rows if row["split"] == split}
        for row in _rows_from_coco_split(split_dir, split, allowed):
            if Path(row["path"]).name not in seen_direct:
                rows.append(row)

    if not rows:
        raise RuntimeError(f"No image files found under {data_root}")

    df = pd.DataFrame(rows)
    actual = sorted(df["label"].astype(str).unique().tolist())
    expected = sorted(labels)
    if actual != expected:
        raise RuntimeError(f"Label mismatch. expected={expected}, actual={actual}")

    split_counts = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    missing = {
        split: sorted(label for label in labels if int(split_counts.loc[split].get(label, 0)) == 0)
        for split in ["train", "valid", "test"]
        if split in split_counts.index
    }
    missing = {split: labels_ for split, labels_ in missing.items() if labels_}
    if missing:
        raise RuntimeError(f"Each label must appear in every split. missing={missing}")

    manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Plain ViT baseline for 3-label implant classification.")
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--manifest", default=str(ROOT / "data" / "manifests" / "plain_vit_3label.csv"))
    p.add_argument("--output-root", default=str(ROOT / "outputs"))
    p.add_argument("--experiment-name", default="plain_vit_3label")
    p.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
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
    p.add_argument("--use-class-weights", action="store_true", default=True)
    p.add_argument("--no-class-weights", dest="use_class_weights", action="store_false")
    p.add_argument("--no-strong-aug", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--rebuild-manifest", action="store_true")
    args = p.parse_args()

    data_root = Path(args.data_root)
    manifest = Path(args.manifest)
    labels = [str(label) for label in args.labels]
    if len(labels) != 3:
        raise RuntimeError(f"This script expects exactly 3 labels, got {len(labels)}: {labels}")

    if args.rebuild_manifest or not manifest.exists():
        build_3label_manifest(data_root, manifest, labels)
    print_manifest_summary(manifest)

    run_args = SupervisedArgs(
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
        label_smoothing=args.label_smoothing,
        dropout=args.dropout,
        drop_path_rate=args.drop_path_rate,
        seed=args.seed,
        amp=True,
        patience=12,
        monitor="macro_f1",
        use_class_weights=args.use_class_weights,
        strong_aug=not args.no_strong_aug,
        allow_cpu=args.allow_cpu,
    )
    train_supervised(run_args)


if __name__ == "__main__":
    main()
