#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluate_3label_metric_from_7label import (
    ImagePathDataset,
    build_3label_manifest,
    extract_embeddings,
    load_stage1_metric_model,
    prototype_predict,
)
from src.data import label_mapping
from src.metrics import compute_metrics, save_confusion_matrix, save_metrics_json, save_predictions


DEFAULT_GROUPS = {
    "Bego_Bicon_ITI": ["Bego", "Bicon", "ITI"],
}


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(1e-12, np.linalg.norm(x, axis=1, keepdims=True))


def group_knn_predict(
    train_z: np.ndarray,
    train_y: np.ndarray,
    query_z: np.ndarray,
    allowed_classes: list[int],
    k: int,
):
    mask = np.isin(train_y, np.asarray(allowed_classes))
    bank_z = l2_normalize(train_z[mask])
    bank_y = train_y[mask]
    q = l2_normalize(query_z)
    sims = q @ bank_z.T
    k_eff = min(k, bank_z.shape[0])
    top_idx = np.argpartition(-sims, kth=k_eff - 1, axis=1)[:, :k_eff]
    preds, confs = [], []
    for row_i, inds in enumerate(top_idx):
        votes: dict[int, float] = {}
        for j in inds:
            cls = int(bank_y[j])
            votes[cls] = votes.get(cls, 0.0) + float(max(sims[row_i, j], 0.0))
        pred = max(votes.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        preds.append(pred)
        confs.append(votes[pred] / max(1e-12, sum(votes.values())))
    return np.asarray(preds), np.asarray(confs)


def evaluate_split_twostage(
    model,
    manifest: Path,
    class_to_idx: dict[str, int],
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    split: str,
    out_dir: Path,
    stage1_use_projection: bool,
    stage2_use_projection: bool,
    groups: dict[str, list[str]],
    k: int,
):
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    label_to_group = {label: group for group, labels in groups.items() for label in labels}
    group_to_indices = {
        group: [class_to_idx[label] for label in labels]
        for group, labels in groups.items()
    }

    train_ds = ImagePathDataset(manifest, "train", class_to_idx, image_size)
    query_ds = ImagePathDataset(manifest, split, class_to_idx, image_size)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    query_loader = torch.utils.data.DataLoader(query_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    train_z1, train_y, _ = extract_embeddings(model, train_loader, device, stage1_use_projection)
    query_z1, query_y, paths = extract_embeddings(model, query_loader, device, stage1_use_projection)
    main_pred, main_conf = prototype_predict(train_z1, train_y, query_z1)

    if stage2_use_projection == stage1_use_projection:
        train_z2, query_z2 = train_z1, query_z1
    else:
        train_z2, _, _ = extract_embeddings(model, train_loader, device, stage2_use_projection)
        query_z2, _, _ = extract_embeddings(model, query_loader, device, stage2_use_projection)

    final_pred = main_pred.copy()
    final_conf = main_conf.copy()
    used_groups = np.asarray([""] * len(main_pred), dtype=object)
    for group, allowed in group_to_indices.items():
        trigger_labels = set(class_to_idx[label] for label in groups[group])
        trigger = np.asarray([int(p) in trigger_labels for p in main_pred])
        if not trigger.any():
            continue
        refined, refined_conf = group_knn_predict(train_z2, train_y, query_z2[trigger], allowed, k=k)
        final_pred[trigger] = refined
        final_conf[trigger] = refined_conf
        used_groups[trigger] = group

    metrics = compute_metrics(query_y, final_pred, class_names)
    save_metrics_json(out_dir / "metrics" / f"{split}_twostage_metrics.json", metrics)
    save_confusion_matrix(out_dir / "confusion_matrices" / f"{split}_twostage.csv", out_dir / "confusion_matrices" / f"{split}_twostage.png", query_y, final_pred, class_names)

    rows = []
    for p, t, mp, mc, fp, fc, group in zip(paths, query_y, main_pred, main_conf, final_pred, final_conf, used_groups):
        rows.append({
            "path": p,
            "true_idx": int(t),
            "true_label": idx_to_class[int(t)],
            "main_pred_idx": int(mp),
            "main_pred_label": idx_to_class[int(mp)],
            "main_confidence": float(mc),
            "used_submodel": str(group),
            "second_stage_evaluated": bool(group),
            "final_pred_idx": int(fp),
            "final_pred_label": idx_to_class[int(fp)],
            "final_confidence": float(fc),
            "correct": bool(int(t) == int(fp)),
        })
    save_predictions(out_dir / "metrics" / f"{split}_twostage_predictions.csv", rows)
    return metrics, int(np.count_nonzero(used_groups != ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a local two-stage metric refiner to 3-label evaluation using a 7-label trained checkpoint.")
    parser.add_argument("--checkpoint", default="/workspace/data_1_2/outputs_two_stage_metric/stage1/best_stage1.pth")
    parser.add_argument("--data-root", default="/workspace/implant_python_only_final/data/3label")
    parser.add_argument("--manifest", default="/workspace/implant_python_only_final/data/manifests/3label.csv")
    parser.add_argument("--output-root", default="/workspace/implant_python_only_final/outputs")
    parser.add_argument("--experiment-name", default="twostage_metric_7label_to_3label")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--proj-hidden-dim", type=int, default=None)
    parser.add_argument("--proj-output-dim", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--stage1-feature", choices=["projection", "backbone"], default="projection")
    parser.add_argument("--stage2-feature", choices=["projection", "backbone"], default="backbone")
    parser.add_argument("--groups-json", default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA is not available. Pass --allow-cpu only for debugging.")

    manifest = Path(args.manifest)
    df = build_3label_manifest(Path(args.data_root), manifest)
    class_to_idx, _ = label_mapping(df)
    groups = DEFAULT_GROUPS if args.groups_json is None else json.loads(Path(args.groups_json).read_text(encoding="utf-8"))

    run_dir = Path(args.output_root) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.experiment_name}"
    for sub in ["metrics", "confusion_matrices"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    model, image_size, metadata = load_stage1_metric_model(Path(args.checkpoint), args, device)
    stage1_use_projection = args.stage1_feature == "projection"
    stage2_use_projection = args.stage2_feature == "projection"
    metadata.update({
        "device": str(device),
        "manifest": str(manifest),
        "class_to_idx": class_to_idx,
        "groups": groups,
        "k": args.k,
        "stage1_feature": args.stage1_feature,
        "stage2_feature": args.stage2_feature,
    })
    (run_dir / "args.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    results = {}
    second_stage_counts = {}
    for split in ["valid", "test"]:
        metrics, count = evaluate_split_twostage(
            model=model,
            manifest=manifest,
            class_to_idx=class_to_idx,
            image_size=image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            split=split,
            out_dir=run_dir,
            stage1_use_projection=stage1_use_projection,
            stage2_use_projection=stage2_use_projection,
            groups=groups,
            k=args.k,
        )
        results[split] = metrics
        second_stage_counts[split] = count

    print(f"RUN_DIR={run_dir}")
    for split, metrics in results.items():
        print(
            f"{split}: accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"weighted_f1={metrics['weighted_f1']:.4f} "
            f"second_stage={second_stage_counts[split]}"
        )


if __name__ == "__main__":
    main()
