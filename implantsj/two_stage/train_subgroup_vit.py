#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_manifest, print_manifest_summary
from src.supervised import SupervisedArgs, train_supervised


def main():
    p = argparse.ArgumentParser(description="Build a subgroup manifest and optionally train a subgroup ViT.")
    p.add_argument("--group", required=True, help="Group name, e.g. NORIS_osstem")
    p.add_argument("--groups-json", default=str(ROOT / "two_stage" / "confusion_groups.json"))
    p.add_argument("--data-root", default=str(ROOT / "data" / "two_stage"))
    p.add_argument("--manifest-dir", default=str(ROOT / "data" / "manifests"))
    p.add_argument("--output-root", default=str(ROOT / "outputs" / "submodels"))
    p.add_argument("--model-name", default="vit_base_patch16_224")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--build-only", action="store_true")
    args = p.parse_args()

    groups = json.loads(Path(args.groups_json).read_text(encoding="utf-8"))
    if args.group not in groups:
        raise KeyError(f"Unknown group {args.group}. Available: {sorted(groups)}")

    group_root = Path(args.data_root) / args.group
    manifest = Path(args.manifest_dir) / f"{args.group}.csv"
    build_manifest(group_root, manifest, layout="split_class_folder")
    print_manifest_summary(manifest)
    if args.build_only:
        return

    train_supervised(
        SupervisedArgs(
            manifest=str(manifest),
            output_root=args.output_root,
            experiment_name=args.group,
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            lr=args.lr,
            allow_cpu=args.allow_cpu,
            use_class_weights=True,
        )
    )


if __name__ == "__main__":
    main()
