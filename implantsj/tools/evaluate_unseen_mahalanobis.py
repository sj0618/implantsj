#!/usr/bin/env python3
"""Evaluate a metric checkpoint on unseen labels with target-train prototypes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.covariance import LedoitWolf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from pipelines.eval.evaluate_3label_metric_from_7label import (
    ImagePathDataset,
    extract_embeddings,
    load_stage1_metric_model,
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def embed_split(model, manifest, class_to_idx, image_size, batch_size, device, split, use_projection):
    dataset = ImagePathDataset(manifest, split, class_to_idx, image_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return extract_embeddings(model, loader, device, use_projection)


def distance_matrices(query, prototypes, precision):
    delta = query[:, None, :] - prototypes[None, :, :]
    query64 = np.asarray(query, dtype=np.float64)
    prototypes64 = np.asarray(prototypes, dtype=np.float64)
    delta64 = query64[:, None, :] - prototypes64[None, :, :]
    precision64 = np.asarray(precision, dtype=np.float64)
    q_norm = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    p_norm = prototypes / np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
    return {
        "euclidean": np.linalg.norm(delta, axis=2),
        "manhattan": np.abs(delta).sum(axis=2),
        "cosine": 1.0 - q_norm @ p_norm.T,
        "mahalanobis": np.sqrt(
            np.maximum(np.einsum("nkd,df,nkf->nk", delta64, precision64, delta64), 0.0)
        ),
    }


def metric_row(split, distance, truth, prediction):
    return {
        "split": split,
        "prototype": "mean",
        "distance": distance,
        "n": len(truth),
        "accuracy": accuracy_score(truth, prediction),
        "macro_f1": precision_recall_fscore_support(truth, prediction, average="macro", zero_division=0)[2],
        "balanced_accuracy": balanced_accuracy_score(truth, prediction),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--embedding", choices=["projection", "backbone"], default="projection")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    manifest = Path(args.manifest).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(manifest)
    required = {"path", "label", "split"}
    if not required <= set(frame.columns):
        raise ValueError(f"manifest needs {sorted(required)}")
    labels = sorted(frame.label.astype(str).unique())
    if set(frame.split.astype(str)) < {"train", "valid", "test"}:
        raise ValueError("manifest needs train, valid, and test splits")
    class_to_idx = {label: index for index, label in enumerate(labels)}
    device_name = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    device = torch.device(device_name)
    model_args = SimpleNamespace(
        model_name=args.model_name,
        image_size=None,
        proj_hidden_dim=None,
        proj_output_dim=None,
        backbone_only=args.embedding == "backbone",
    )
    model, image_size, metadata = load_stage1_metric_model(checkpoint, model_args, device)
    use_projection = args.embedding == "projection"

    embedded = {
        split: embed_split(
            model, manifest, class_to_idx, image_size, args.batch_size, device, split, use_projection
        )
        for split in ("train", "valid", "test")
    }
    train_x, train_y, train_paths = embedded["train"]
    prototypes = np.stack([train_x[train_y == class_to_idx[label]].mean(axis=0) for label in labels])
    residuals = np.concatenate(
        [train_x[train_y == class_to_idx[label]] - prototypes[index] for index, label in enumerate(labels)]
    )
    covariance = LedoitWolf().fit(residuals)
    precision = covariance.precision_

    metric_rows = []
    per_class_rows = []
    prediction_rows = []
    for split in ("valid", "test"):
        query_x, query_y, paths = embedded[split]
        truth = np.asarray(labels)[query_y]
        for distance, matrix in distance_matrices(query_x, prototypes, precision).items():
            prediction = np.asarray(labels)[matrix.argmin(axis=1)]
            metric_rows.append(metric_row(split, distance, truth, prediction))
            p, r, f1, support = precision_recall_fscore_support(
                truth, prediction, labels=labels, average=None, zero_division=0
            )
            per_class_rows.extend(
                {
                    "split": split,
                    "distance": distance,
                    "label": label,
                    "precision": p[index],
                    "recall": r[index],
                    "f1": f1[index],
                    "support": support[index],
                }
                for index, label in enumerate(labels)
            )
            prediction_rows.extend(
                {
                    "split": split,
                    "distance": distance,
                    "path": path,
                    "true_label": true,
                    "predicted_label": predicted,
                    "distance_value": matrix[index, labels.index(predicted)],
                    "correct": true == predicted,
                }
                for index, (path, true, predicted) in enumerate(zip(paths, truth, prediction))
            )
            pd.DataFrame(confusion_matrix(truth, prediction, labels=labels), index=labels, columns=labels).to_csv(
                output / f"confusion_{split}_{distance}.csv"
            )

    test_x, test_y, test_paths = embedded["test"]
    test_truth = np.asarray(labels)[test_y]
    train_hashes = {sha256(path) for path in train_paths}
    keep = np.asarray([sha256(path) not in train_hashes for path in test_paths])
    if not keep.any():
        raise RuntimeError(
            f"All {len(test_paths)} test images are byte-identical to a train image, so the "
            f"duplicate-free split is empty. train and test are probably built from the same "
            f"files; rebuild the dataset splits before evaluating."
        )
    duplicate_free_rows = []
    for distance, matrix in distance_matrices(test_x[keep], prototypes, precision).items():
        prediction = np.asarray(labels)[matrix.argmin(axis=1)]
        row = metric_row("test_duplicate_free", distance, test_truth[keep], prediction)
        row["excluded_exact_duplicates"] = int((~keep).sum())
        duplicate_free_rows.append(row)

    direct = cdist(test_x[:20], prototypes, metric="mahalanobis", VI=precision)
    calculated = distance_matrices(test_x[:20], prototypes, precision)["mahalanobis"]
    max_error = float(np.max(np.abs(direct - calculated)))
    if max_error > 1e-8:
        raise AssertionError(f"Mahalanobis verification failed: {max_error}")

    pd.DataFrame(metric_rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(duplicate_free_rows).to_csv(output / "metrics_duplicate_free.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(output / "per_class_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output / "predictions.csv.gz", index=False, compression="gzip")
    np.savez_compressed(output / "mahalanobis_model.npz", labels=np.asarray(labels), prototypes=prototypes, precision=precision)
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "manifest": str(manifest),
                "manifest_sha256": sha256(manifest),
                "embedding": args.embedding,
                "model_metadata": metadata,
                "prototype": "target-train class mean",
                "covariance": "target-train pooled within-class residual Ledoit-Wolf",
                "target_valid_test_used_for_fitting": False,
                "scipy_max_absolute_error": max_error,
                "excluded_exact_duplicates": int((~keep).sum()),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(pd.DataFrame(metric_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
