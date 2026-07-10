#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.closed_network_data import (
    build_class_folder_map,
    build_closed_network_dataframe,
    build_seen27_manifest,
    build_unseen15_manifest,
    infer_seen_unseen_labels,
    label_counts,
    materialize_imagefolder,
    split_train_valid,
    write_label_list,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare closed-network TL1/TS1/VL1/VS1 implant data for VLM and metric-learning pipelines."
    )
    parser.add_argument("--train-json-root", required=True, help="Example: /data/TL1")
    parser.add_argument("--train-image-root", required=True, help="Example: /data/TS1")
    parser.add_argument("--test-json-root", required=True, help="Example: /data/VL1")
    parser.add_argument("--test-image-root", required=True, help="Example: /data/VS1")
    parser.add_argument("--output-root", default=str(ROOT / "data" / "closed_network"))
    parser.add_argument("--label-key", default="모델명")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Validation split ratio drawn from original train only.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seen-label-count", type=int, default=27)
    parser.add_argument("--unseen-label-count", type=int, default=15)
    parser.add_argument(
        "--materialize-imagefolders",
        action="store_true",
        help="Also build ImageFolder-style directories for 42-label / seen27 / unseen15.",
    )
    parser.add_argument("--index-class-folders", action="store_true", help="Use 01.label style class folders.")
    parser.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    manifests_dir = output_root / "manifests"
    labels_dir = output_root / "label_splits"
    image_root = output_root / "imagefolders"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    base_df = build_closed_network_dataframe(
        train_image_root=args.train_image_root,
        train_json_root=args.train_json_root,
        test_image_root=args.test_image_root,
        test_json_root=args.test_json_root,
        label_keys=(args.label_key,),
    )
    split_df = split_train_valid(base_df, valid_ratio=args.valid_ratio, seed=args.seed)

    seen_labels, unseen_labels, stats_df = infer_seen_unseen_labels(
        split_df,
        seen_label_count=args.seen_label_count,
        unseen_label_count=args.unseen_label_count,
    )
    manifest_42 = split_df.copy()
    manifest_seen27 = build_seen27_manifest(split_df, seen_labels)
    manifest_unseen15 = build_unseen15_manifest(split_df, unseen_labels)

    out_42 = write_manifest(manifest_42, manifests_dir / "closed_network_42label.csv")
    out_seen27 = write_manifest(manifest_seen27, manifests_dir / "closed_network_seen27.csv")
    out_unseen15 = write_manifest(manifest_unseen15, manifests_dir / "closed_network_unseen15.csv")

    stats_df["selected_split"] = stats_df["label"].map(
        lambda x: "unseen15" if str(x) in set(unseen_labels) else "seen27"
    )
    stats_path = labels_dir / "label_counts_train_source.csv"
    stats_df.to_csv(stats_path, index=False)
    seen_labels_path = write_label_list(seen_labels, labels_dir / "seen27_labels.txt")
    unseen_labels_path = write_label_list(unseen_labels, labels_dir / "unseen15_labels.txt")

    summary = {
        "train_json_root": str(args.train_json_root),
        "train_image_root": str(args.train_image_root),
        "test_json_root": str(args.test_json_root),
        "test_image_root": str(args.test_image_root),
        "label_key": args.label_key,
        "valid_ratio": args.valid_ratio,
        "seed": args.seed,
        "total_rows": int(len(split_df)),
        "total_labels": int(split_df["label"].astype(str).nunique()),
        "seen_label_count": len(seen_labels),
        "unseen_label_count": len(unseen_labels),
        "manifests": {
            "all_42": str(out_42),
            "seen_27": str(out_seen27),
            "unseen_15": str(out_unseen15),
        },
        "label_files": {
            "stats": str(stats_path),
            "seen_27": str(seen_labels_path),
            "unseen_15": str(unseen_labels_path),
        },
        "split_counts_42": {
            split: {str(k): int(v) for k, v in counts.items()}
            for split, counts in {
                "train": label_counts(manifest_42, split="train"),
                "valid": label_counts(manifest_42, split="valid"),
                "test": label_counts(manifest_42, split="test"),
            }.items()
        },
    }

    if args.materialize_imagefolders:
        class_map_42 = build_class_folder_map(manifest_42["label"].astype(str).unique().tolist(), indexed=args.index_class_folders)
        class_map_27 = build_class_folder_map(seen_labels, indexed=args.index_class_folders)
        class_map_15 = build_class_folder_map(unseen_labels, indexed=args.index_class_folders)

        materialize_imagefolder(
            manifest_42,
            image_root / "data_42label",
            class_folder_map=class_map_42,
            link_mode=args.link_mode,
        )
        materialize_imagefolder(
            manifest_seen27,
            image_root / "data_metric_seen27",
            class_folder_map=class_map_27,
            link_mode=args.link_mode,
        )
        materialize_imagefolder(
            manifest_unseen15,
            image_root / "data_metric_unseen15",
            class_folder_map=class_map_15,
            split_name_map={"test": "test"},
            link_mode=args.link_mode,
        )
        pd.DataFrame(
            [{"raw_label": k, "folder_name": v} for k, v in sorted(class_map_42.items())]
        ).to_csv(labels_dir / "class_folder_map_42label.csv", index=False)
        pd.DataFrame(
            [{"raw_label": k, "folder_name": v} for k, v in sorted(class_map_27.items())]
        ).to_csv(labels_dir / "class_folder_map_seen27.csv", index=False)
        pd.DataFrame(
            [{"raw_label": k, "folder_name": v} for k, v in sorted(class_map_15.items())]
        ).to_csv(labels_dir / "class_folder_map_unseen15.csv", index=False)
        summary["imagefolders"] = {
            "all_42": str(image_root / "data_42label"),
            "seen_27": str(image_root / "data_metric_seen27"),
            "unseen_15": str(image_root / "data_metric_unseen15"),
        }

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"SUMMARY={summary_path}")
    print(f"MANIFEST_42={out_42}")
    print(f"MANIFEST_SEEN27={out_seen27}")
    print(f"MANIFEST_UNSEEN15={out_unseen15}")
    print(f"SEEN27_LABELS={seen_labels_path}")
    print(f"UNSEEN15_LABELS={unseen_labels_path}")


if __name__ == "__main__":
    main()
