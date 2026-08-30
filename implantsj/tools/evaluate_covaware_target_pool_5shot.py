#!/usr/bin/env python3
"""Evaluate a checkpoint on all target images with repeated strict 5-shot episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from pipelines.eval.evaluate_3label_metric_from_7label import (
    ImagePathDataset,
    extract_embeddings,
    load_stage1_metric_model,
)
from tools.improve_unseen_mahalanobis import candidate_scorers, mahalanobis_scores


def metric_rows(truth: np.ndarray, scores: np.ndarray) -> tuple[dict, np.ndarray]:
    prediction = scores.argmin(axis=1)
    _, _, f1, _ = precision_recall_fscore_support(
        truth, prediction, labels=np.arange(scores.shape[1]), average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
    }, prediction


def class_weighted_rda_scores(
    support_x: np.ndarray,
    support_y: np.ndarray,
    query_x: np.ndarray,
    classes: int,
    special_class: int,
    special_weight: float,
    default_weight: float = 0.75,
    logdet_weight: float = 1.0,
) -> np.ndarray:
    centers = np.stack([support_x[support_y == index].mean(axis=0) for index in range(classes)])
    residuals = support_x - centers[support_y]
    pooled_cov = LedoitWolf().fit(residuals).covariance_.astype(np.float64)
    class_covariances = np.stack(
        [LedoitWolf().fit(support_x[support_y == index] - centers[index]).covariance_ for index in range(classes)]
    ).astype(np.float64)
    weights = np.full(classes, default_weight, dtype=np.float64)
    weights[special_class] = special_weight
    covariances = (1.0 - weights[:, None, None]) * pooled_cov[None] + weights[:, None, None] * class_covariances
    scales = np.trace(covariances, axis1=1, axis2=2) / covariances.shape[1]
    covariances += np.eye(covariances.shape[1])[None] * np.maximum(scales * 1e-6, 1e-12)[:, None, None]
    precisions = np.stack([np.linalg.inv(covariance) for covariance in covariances])
    logdets = np.asarray(
        [np.log(np.maximum(np.linalg.eigvalsh(covariance), np.finfo(float).tiny)).sum() for covariance in covariances]
    )
    return mahalanobis_scores(query_x, centers, precisions) + logdet_weight * logdets[None]


def evaluate_embedding(
    embedding: np.ndarray,
    labels: np.ndarray,
    paths: np.ndarray,
    class_names: list[str],
    output: Path,
    seeds: tuple[int, ...],
    nucleoss_rda_weight: float | None = None,
    all_rda_weight: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    per_class_rows: list[dict] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        support_indices = np.concatenate(
            [rng.choice(np.flatnonzero(labels == index), 5, replace=False) for index in range(len(class_names))]
        )
        query_mask = np.ones(len(labels), dtype=bool)
        query_mask[support_indices] = False
        support_x, support_y = embedding[support_indices], labels[support_indices]
        query_x, query_y = embedding[query_mask], labels[query_mask]
        query_paths = paths[query_mask]
        support_norm = support_x / np.maximum(np.linalg.norm(support_x, axis=1, keepdims=True), 1e-12)
        query_norm = query_x / np.maximum(np.linalg.norm(query_x, axis=1, keepdims=True), 1e-12)
        centers = np.stack([support_norm[support_y == index].mean(0) for index in range(len(class_names))])
        center_norm = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
        scorers = {
            "euclidean": ((query_norm[:, None] - centers[None]) ** 2).sum(2),
            "manhattan": np.abs(query_norm[:, None] - centers[None]).sum(2),
            "cosine": 1.0 - query_norm @ center_norm.T,
        }
        fixed = {name: scorer for name, _, scorer in candidate_scorers(support_norm, support_y, len(class_names), seed)}
        scorers["pooled_ledoit_wolf"] = fixed["pooled_ledoit_wolf"](query_norm)
        scorers["rda_0.75_logdet_1"] = fixed["rda_0.75_logdet_1"](query_norm)
        if nucleoss_rda_weight is not None:
            scorers[f"rda_nucleoss_{nucleoss_rda_weight:g}_logdet_1"] = class_weighted_rda_scores(
                support_norm,
                support_y,
                query_norm,
                len(class_names),
                class_names.index("nucleoss"),
                nucleoss_rda_weight,
            )
        if all_rda_weight is not None:
            scorers[f"rda_all_{all_rda_weight:g}_logdet_1"] = class_weighted_rda_scores(
                support_norm,
                support_y,
                query_norm,
                len(class_names),
                0,
                all_rda_weight,
                default_weight=all_rda_weight,
            )
        seed_output = output / f"seed{seed}"
        seed_output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"path": paths[support_indices], "label": [class_names[index] for index in support_y]}).to_csv(
            seed_output / "support_membership.csv", index=False
        )
        for method, scores in scorers.items():
            metrics, prediction = metric_rows(query_y, scores)
            rows.append({"seed": seed, "method": method, "n": len(query_y), **metrics})
            precision, recall, f1, support = precision_recall_fscore_support(
                query_y, prediction, labels=np.arange(len(class_names)), average=None, zero_division=0
            )
            per_class_rows.extend(
                {
                    "seed": seed,
                    "method": method,
                    "label": class_names[index],
                    "precision": precision[index],
                    "recall": recall[index],
                    "f1": f1[index],
                    "support": support[index],
                }
                for index in range(len(class_names))
            )
            pd.DataFrame(
                {
                    "path": query_paths,
                    "true_label": [class_names[index] for index in query_y],
                    "predicted_label": [class_names[index] for index in prediction],
                    "correct": query_y == prediction,
                }
            ).to_csv(seed_output / f"predictions_{method}.csv", index=False)
            pd.DataFrame(confusion_matrix(query_y, prediction, labels=np.arange(len(class_names))), index=class_names, columns=class_names).to_csv(
                seed_output / f"confusion_{method}.csv"
            )
    metrics_frame = pd.DataFrame(rows)
    per_class_frame = pd.DataFrame(per_class_rows)
    metrics_frame.to_csv(output / "results.csv", index=False)
    per_class_frame.to_csv(output / "per_class.csv", index=False)
    metrics_frame.groupby("method")[["accuracy", "macro_f1", "balanced_accuracy"]].mean().to_csv(output / "summary.csv")
    return metrics_frame, per_class_frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--nucleoss-rda-weight", type=float, default=None)
    parser.add_argument("--rda-weight", type=float, default=None)
    args = parser.parse_args()
    device_name = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    device = torch.device(device_name)
    manifest = Path(args.manifest).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(manifest)
    class_names = sorted(frame.label.astype(str).unique())
    class_to_idx = {label: index for index, label in enumerate(class_names)}
    model_args = SimpleNamespace(model_name=None, image_size=None, proj_hidden_dim=None, proj_output_dim=None, backbone_only=False)
    model, image_size, metadata = load_stage1_metric_model(Path(args.checkpoint), model_args, device)
    embeddings: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    for use_projection, name in ((False, "backbone"), (True, "projection")):
        chunks = []
        for split in ("train", "valid", "test"):
            dataset = ImagePathDataset(manifest, split, class_to_idx, image_size)
            loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            chunks.append(extract_embeddings(model, loader, device, use_projection))
        embeddings[name] = (
            np.concatenate([chunk[0] for chunk in chunks]),
            np.concatenate([chunk[1] for chunk in chunks]),
            np.asarray(sum((chunk[2] for chunk in chunks), [])),
        )
        evaluate_embedding(
            *embeddings[name],
            class_names,
            output / name,
            (0, 1, 2, 3, 4, 42),
            args.nucleoss_rda_weight,
            args.rda_weight,
        )
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "manifest": str(manifest),
                "device": device_name,
                "target_images": len(frame),
                "support_shots_per_class": 5,
                "support_total": 5 * len(class_names),
                "query_total": len(frame) - 5 * len(class_names),
                "classes": class_names,
                "seeds": [0, 1, 2, 3, 4, 42],
                "covariance": "pooled Ledoit-Wolf from the same five support images per class",
                "target_validation_used": False,
                "nucleoss_rda_weight": args.nucleoss_rda_weight,
                "rda_weight": args.rda_weight,
                "model_metadata": metadata,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("evaluation complete", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
