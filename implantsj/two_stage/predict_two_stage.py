#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_manifest, create_loader, label_mapping, load_manifest
from src.models import create_classifier


def checkpoint_from_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        direct = path / "checkpoints" / "best.pt"
        if direct.exists():
            return direct
        direct = path / "best.pt"
        if direct.exists():
            return direct
        raise FileNotFoundError(f"No best checkpoint found under directory: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def checkpoint_for_group(submodel_root: Path, group: str) -> Path:
    direct = submodel_root / group
    if direct.exists():
        return checkpoint_from_path(direct)

    matches = sorted(
        [p for p in submodel_root.glob(f"*_{group}") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if matches:
        return checkpoint_from_path(matches[0])

    return checkpoint_from_path(direct)


def load_classifier(checkpoint: Path, manifest: Path, device, model_name: str | None = None, image_size: int | None = None):
    ckpt = torch.load(checkpoint, map_location="cpu")
    class_to_idx = {str(k): int(v) for k, v in ckpt.get("class_to_idx", {}).items()}
    if not class_to_idx:
        class_to_idx, _ = label_mapping(load_manifest(manifest, split=None))
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    ckpt_args = ckpt.get("args", {}) or {}
    resolved_model = model_name or ckpt_args.get("model_name") or "vit_base_patch16_224"
    resolved_size = int(image_size or ckpt_args.get("image_size", 224))

    model = create_classifier(resolved_model, num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, class_to_idx, idx_to_class, resolved_size


def predict_batch(model, x, idx_to_class):
    probs = torch.softmax(model(x), dim=1).cpu()
    conf, pred = probs.max(dim=1)
    labels = [idx_to_class[int(i)] for i in pred.tolist()]
    return labels, conf.tolist()


def main():
    p = argparse.ArgumentParser(description="Run main ViT prediction, then refine configured confusion groups.")
    p.add_argument("--main-checkpoint", required=True)
    p.add_argument("--manifest", default=str(ROOT / "data" / "manifests" / "7label.csv"))
    p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    p.add_argument("--groups-json", default=str(ROOT / "two_stage" / "confusion_groups.json"))
    p.add_argument("--submodel-root", default=str(ROOT / "outputs" / "submodels"))
    p.add_argument("--output", default=str(ROOT / "outputs" / "two_stage_predictions.csv"))
    p.add_argument("--summary-output", default=str(ROOT / "outputs" / "two_stage_summary.json"))
    p.add_argument("--model-name", default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--allow-cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA is not available. Pass --allow-cpu only for debugging.")

    manifest = Path(args.manifest)
    groups = json.loads(Path(args.groups_json).read_text(encoding="utf-8"))
    label_to_group = {label: group for group, labels in groups.items() for label in labels}

    main_model, main_class_to_idx, main_idx_to_class, image_size = load_classifier(
        checkpoint_from_path(args.main_checkpoint), manifest, device, args.model_name, args.image_size
    )
    loader = create_loader(
        manifest, args.split, main_class_to_idx, image_size, args.batch_size, args.num_workers,
        train=False, strong_aug=False, return_path=True
    )

    submodel_root = Path(args.submodel_root)

    submodels = {}

    def group_manifest_path(group: str) -> Path:
        group_checkpoint = checkpoint_for_group(submodel_root, group)
        manifest_path = ROOT / "data" / "manifests" / f"{group}.csv"
        if not manifest_path.exists():
            build_manifest(ROOT / "data" / "two_stage" / group, manifest_path, layout="split_class_folder")
        return group_checkpoint, manifest_path

    def get_submodel(group: str):
        if group not in submodels:
            group_checkpoint, manifest_path = group_manifest_path(group)
            submodels[group] = load_classifier(group_checkpoint, manifest_path, device, args.model_name, args.image_size)
        return submodels[group]

    rows = []
    main_pred_counts = Counter()
    second_stage_counts = Counter()
    with torch.no_grad():
        for x, y, paths in loader:
            x = x.to(device, non_blocking=True)
            main_labels, main_conf = predict_batch(main_model, x, main_idx_to_class)
            true_labels = [main_idx_to_class[int(i)] for i in y.tolist()]
            main_pred_counts.update(main_labels)

            final_labels = list(main_labels)
            final_conf = list(main_conf)
            used_group = [""] * len(main_labels)

            # Only samples whose first-stage prediction is one of the configured
            # confusion labels are sent to a second-stage subgroup classifier.
            triggered_groups = sorted(set(label_to_group.get(label, "") for label in main_labels) - {""})
            for group in triggered_groups:
                indices = [i for i, label in enumerate(main_labels) if label_to_group.get(label) == group]
                if not indices:
                    continue
                second_stage_counts[group] += len(indices)
                sub_model, _, sub_idx_to_class, _ = get_submodel(group)
                sub_x = x[indices]
                sub_labels, sub_conf = predict_batch(sub_model, sub_x, sub_idx_to_class)
                for local_i, label, conf in zip(indices, sub_labels, sub_conf):
                    final_labels[local_i] = label
                    final_conf[local_i] = conf
                    used_group[local_i] = group

            for path, true, main_label, mconf, final, fconf, group in zip(
                paths, true_labels, main_labels, main_conf, final_labels, final_conf, used_group
            ):
                rows.append(
                    {
                        "path": path,
                        "true_label": true,
                        "main_pred_label": main_label,
                        "main_confidence": float(mconf),
                        "used_submodel": group,
                        "second_stage_evaluated": bool(group),
                        "final_pred_label": final,
                        "final_confidence": float(fconf),
                        "correct": true == final,
                    }
                )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "path", "true_label", "main_pred_label", "main_confidence",
            "used_submodel", "second_stage_evaluated", "final_pred_label", "final_confidence", "correct",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "total_samples": len(rows),
        "second_stage_samples": int(sum(second_stage_counts.values())),
        "first_stage_only_samples": int(len(rows) - sum(second_stage_counts.values())),
        "second_stage_by_group": dict(second_stage_counts),
        "main_pred_counts": dict(main_pred_counts),
        "groups": groups,
        "trigger_rule": "Run second-stage only when the first-stage predicted label belongs to a configured confusion group.",
    }
    summary_out = Path(args.summary_output)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} two-stage predictions to {out}")
    print(f"Second-stage evaluated {summary['second_stage_samples']} / {summary['total_samples']} samples")
    print(f"Wrote summary to {summary_out}")


if __name__ == "__main__":
    main()
