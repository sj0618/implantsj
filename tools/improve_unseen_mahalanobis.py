#!/usr/bin/env python3
"""Tune train-only Mahalanobis variants on validation, then evaluate test once."""

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
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf, OAS
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

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


def embed_split(model, manifest, class_to_idx, image_size, batch_size, device, split, projection):
    dataset = ImagePathDataset(manifest, split, class_to_idx, image_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return extract_embeddings(model, loader, device, projection)


def class_means(features: np.ndarray, targets: np.ndarray, classes: int) -> np.ndarray:
    return np.stack([features[targets == index].mean(axis=0) for index in range(classes)])


def mahalanobis_scores(query: np.ndarray, centers: np.ndarray, precision: np.ndarray) -> np.ndarray:
    delta = np.asarray(query[:, None, :] - centers[None, :, :], dtype=np.float64)
    precision = np.asarray(precision, dtype=np.float64)
    if precision.ndim == 2:
        return np.einsum("nkd,de,nke->nk", delta, precision, delta)
    return np.einsum("nkd,kde,nke->nk", delta, precision, delta)


def normalize_rows(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    return features / np.maximum(
        np.linalg.norm(features, axis=1, keepdims=True), np.finfo(np.float64).eps
    )


def spectral_fit(train_x, train_y, classes, prototype_x, prototype_y):
    normalized_train = normalize_rows(train_x)
    full_centers = class_means(normalized_train, train_y, classes)
    covariance = LedoitWolf().fit(normalized_train - full_centers[train_y]).covariance_
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(covariance, dtype=np.float64))
    scale = max(float(np.maximum(eigenvalues, 0.0).mean()), np.finfo(np.float64).tiny)
    eigenvalues = np.maximum(eigenvalues, scale * 1e-6)
    normalized_prototypes = normalize_rows(prototype_x)
    return (
        class_means(normalized_prototypes, prototype_y, classes),
        normalized_prototypes.mean(axis=0),
        eigenvectors,
        eigenvalues,
        float(eigenvalues.mean()),
    )


def fractional_scores(query, centers, eigenvectors, eigenvalues, mean_eigenvalue, alpha):
    delta = normalize_rows(query)[:, None, :] - centers[None, :, :]
    projected = np.einsum("nkd,de->nke", delta, eigenvectors)
    weights = (eigenvalues / mean_eigenvalue) ** (-alpha)
    return np.einsum("nkd,d,nkd->nk", projected, weights, projected)


def centered_angular_scores(
    query, centers, global_center, eigenvectors, eigenvalues, mean_eigenvalue, alpha
):
    weights = (eigenvalues / mean_eigenvalue) ** (-alpha / 2.0)
    transformed_query = normalize_rows((normalize_rows(query) - global_center) @ eigenvectors * weights)
    transformed_centers = normalize_rows((centers - global_center) @ eigenvectors * weights)
    return 1.0 - transformed_query @ transformed_centers.T


def score_metrics(truth: np.ndarray, scores: np.ndarray) -> tuple[dict, np.ndarray]:
    prediction = scores.argmin(axis=1)
    return {
        "n": len(truth),
        "accuracy": accuracy_score(truth, prediction),
        "macro_f1": precision_recall_fscore_support(
            truth, prediction, average="macro", zero_division=0
        )[2],
        "balanced_accuracy": balanced_accuracy_score(truth, prediction),
    }, prediction


def pooled_fit(
    features: np.ndarray,
    targets: np.ndarray,
    classes: int,
    estimator=LedoitWolf,
    centers: np.ndarray | None = None,
):
    centers = class_means(features, targets, classes) if centers is None else centers
    fit = estimator().fit(features - centers[targets])
    return centers, fit.covariance_.astype(np.float64), fit.precision_.astype(np.float64)


def candidate_scorers(
    train_x,
    train_y,
    classes,
    seed,
    prototype_x=None,
    prototype_y=None,
    include_experimental=False,
):
    prototype_x = train_x if prototype_x is None else prototype_x
    prototype_y = train_y if prototype_y is None else prototype_y
    full_centers, pooled_cov, pooled_precision = pooled_fit(train_x, train_y, classes)
    centers = class_means(prototype_x, prototype_y, classes)
    scorers = [("pooled_ledoit_wolf", {}, lambda q: mahalanobis_scores(q, centers, pooled_precision))]

    if include_experimental:
        spectral = spectral_fit(train_x, train_y, classes, prototype_x, prototype_y)
        for alpha in (0.5, 0.75):
            scorers.append(
                (
                    f"fractional_{alpha:g}",
                    {"spectral_alpha": alpha, "experimental_fixed_only": True},
                    lambda q, a=alpha, s=spectral: fractional_scores(q, s[0], s[2], s[3], s[4], a),
                )
            )
        for alpha in (0.5, 0.75):
            scorers.append(
                (
                    f"centered_angular_{alpha:g}",
                    {"spectral_alpha": alpha, "experimental_fixed_only": True},
                    lambda q, a=alpha, s=spectral: centered_angular_scores(
                        q, s[0], s[1], s[2], s[3], s[4], a
                    ),
                )
            )

    _, _, oas_precision = pooled_fit(train_x, train_y, classes, OAS, full_centers)
    scorers.append(("pooled_oas", {}, lambda q: mahalanobis_scores(q, centers, oas_precision)))

    residuals = train_x - full_centers[train_y]
    variance = residuals.var(axis=0, ddof=1)
    diagonal_precision = np.diag(1.0 / np.maximum(variance, np.median(variance) * 1e-4))
    scorers.append(("pooled_diagonal", {}, lambda q: mahalanobis_scores(q, centers, diagonal_precision)))

    rng = np.random.default_rng(seed)
    balanced_indices = np.concatenate(
        [
            rng.choice(np.flatnonzero(train_y == index), np.bincount(train_y).min(), replace=False)
            for index in range(classes)
        ]
    )
    balanced_residuals = train_x[balanced_indices] - full_centers[train_y[balanced_indices]]
    balanced_precision = LedoitWolf().fit(balanced_residuals).precision_
    scorers.append(
        ("class_balanced_ledoit_wolf", {}, lambda q: mahalanobis_scores(q, centers, balanced_precision))
    )

    class_covariances = np.stack(
        [LedoitWolf().fit(train_x[train_y == index] - full_centers[index]).covariance_ for index in range(classes)]
    ).astype(np.float64)
    equal_covariance = class_covariances.mean(axis=0)
    for weight in (0.25, 0.5, 0.75, 1.0):
        precision = np.linalg.inv((1.0 - weight) * pooled_cov + weight * equal_covariance)
        scorers.append(
            (
                f"equal_class_covariance_{weight:g}",
                {"equal_class_covariance_weight": weight},
                lambda q, p=precision: mahalanobis_scores(q, centers, p),
            )
        )

    dimensions = [value for value in (32, 64, 128, 256) if value < min(train_x.shape)]
    for balanced, fit_x in ((False, train_x), (True, train_x[balanced_indices])):
        if not dimensions:
            break
        pca = PCA(n_components=max(dimensions), svd_solver="randomized", random_state=seed).fit(fit_x)
        transformed_train = pca.transform(train_x)
        transformed_prototype = pca.transform(prototype_x)
        for dimension in dimensions:
            _, _, pca_precision = pooled_fit(
                transformed_train[:, :dimension], train_y, classes
            )
            pca_centers = class_means(transformed_prototype[:, :dimension], prototype_y, classes)
            name = f"{'balanced_' if balanced else ''}pca_{dimension}"
            scorers.append(
                (
                    name,
                    {"pca_dimensions": dimension, "balanced_pca_fit": balanced},
                    lambda q, p=pca, d=dimension, c=pca_centers, vi=pca_precision: mahalanobis_scores(
                        p.transform(q)[:, :d], c, vi
                    ),
                )
            )

    for weight in (0.25, 0.5, 0.75, 1.0):
        covariances = (1.0 - weight) * pooled_cov[None, :, :] + weight * class_covariances
        scales = np.trace(covariances, axis1=1, axis2=2) / covariances.shape[1]
        covariances += np.eye(covariances.shape[1])[None, :, :] * np.maximum(scales * 1e-6, 1e-12)[:, None, None]
        precisions = np.stack([np.linalg.inv(covariance) for covariance in covariances])
        logdets = np.asarray(
            [np.log(np.maximum(np.linalg.eigvalsh(covariance), np.finfo(float).tiny)).sum() for covariance in covariances]
        )
        for logdet_weight in (0.0, 1.0):
            scorers.append(
                (
                    f"rda_{weight:g}_logdet_{logdet_weight:g}",
                    {"class_covariance_weight": weight, "logdet_weight": logdet_weight},
                    lambda q, p=precisions, l=logdets, w=logdet_weight: mahalanobis_scores(q, centers, p)
                    + w * l[None, :],
                )
            )

    if prototype_x is train_x:
        whitening = np.linalg.cholesky(pooled_precision)
        train_white = train_x @ whitening
        for clusters in (2, 3, 4, 5):
            cluster_centers = [
                KMeans(n_clusters=clusters, random_state=seed, n_init=10)
                .fit(train_white[train_y == index])
                .cluster_centers_
                for index in range(classes)
            ]

            def multi_scores(query, w=whitening, cs=cluster_centers):
                query_white = query @ w
                return np.stack(
                    [((query_white[:, None, :] - center[None, :, :]) ** 2).sum(axis=2).min(axis=1) for center in cs],
                    axis=1,
                )

            scorers.append((f"multi_prototype_{clusters}", {"prototypes_per_class": clusters}, multi_scores))

    return scorers


def self_check() -> None:
    rng = np.random.default_rng(7)
    train_x = np.vstack([rng.normal(index * 3, 0.4, size=(30, 6)) for index in range(3)])
    train_y = np.repeat(np.arange(3), 30)
    centers, _, precision = pooled_fit(train_x, train_y, 3)
    query = train_x[:5]
    expected = cdist(query, centers, metric="mahalanobis", VI=precision) ** 2
    assert np.allclose(mahalanobis_scores(query, centers, precision), expected, atol=1e-9)
    scorers = candidate_scorers(train_x, train_y, 3, 7)
    assert scorers[0][0] == "pooled_ledoit_wolf" and all(fn(query).shape == (5, 3) for _, _, fn in scorers)
    experimental = candidate_scorers(train_x, train_y, 3, 7, include_experimental=True)
    experimental_map = {name: scorer for name, _, scorer in experimental}
    names = {
        "fractional_0.5",
        "fractional_0.75",
        "centered_angular_0.5",
        "centered_angular_0.75",
    }
    assert names.isdisjoint({name for name, _, _ in scorers}) and names <= experimental_map.keys()
    for name in names:
        for sample, shape in ((query, (5, 3)), (np.zeros((1, 6)), (1, 3)), (np.empty((0, 6)), (0, 3))):
            scores = experimental_map[name](sample)
            assert scores.shape == shape and np.isfinite(scores).all()
    assert not np.allclose(
        experimental_map["centered_angular_0.5"](query),
        experimental_map["centered_angular_0.75"](query),
        atol=1e-12,
    )
    spectral = spectral_fit(train_x, train_y, 3, train_x, train_y)
    floored_precision = (spectral[2] * (1.0 / spectral[3])) @ spectral[2].T
    pooled = mahalanobis_scores(normalize_rows(query), spectral[0], floored_precision)
    fractional_one = fractional_scores(query, spectral[0], spectral[2], spectral[3], spectral[4], 1.0)
    assert np.allclose(fractional_one / spectral[4], pooled, atol=1e-9)
    assert np.array_equal(fractional_one.argmin(axis=1), pooled.argmin(axis=1))
    few_x = rng.normal(size=(30, 64))
    few_y = np.repeat(np.arange(3), 10)
    few_scorers = candidate_scorers(few_x, few_y, 3, 7, include_experimental=True)
    assert all(fn(few_x[:5]).shape == (5, 3) for name, _, fn in few_scorers if name in names)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--manifest")
    parser.add_argument("--output-dir")
    parser.add_argument("--model-name")
    parser.add_argument("--embedding", choices=["projection", "backbone"], default="backbone")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prototype-shots", type=int)
    parser.add_argument("--covariance-from-prototypes", action="store_true")
    parser.add_argument("--fixed-method")
    parser.add_argument("--embedding-cache")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("self-check passed")
        return 0
    if not all((args.checkpoint, args.manifest, args.output_dir)):
        parser.error("--checkpoint, --manifest and --output-dir are required")
    if args.covariance_from_prototypes and args.prototype_shots is None:
        parser.error("--covariance-from-prototypes requires --prototype-shots")

    checkpoint = Path(args.checkpoint).resolve()
    manifest = Path(args.manifest).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(manifest)
    if not {"path", "label", "split"} <= set(frame.columns):
        raise ValueError("manifest needs path, label and split")
    if not {"train", "valid", "test"} <= set(frame.split.astype(str)):
        raise ValueError("manifest needs train, valid and test splits")
    labels = sorted(frame.label.astype(str).unique())
    class_to_idx = {label: index for index, label in enumerate(labels)}
    device_name = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    device = torch.device(device_name)
    if args.embedding_cache:
        cache = np.load(args.embedding_cache)
        if cache["labels"].tolist() != labels:
            raise ValueError("embedding cache labels differ from manifest")
        embedded = {
            split: (cache[f"{split}_x"], cache[f"{split}_y"], cache[f"{split}_paths"].tolist())
            for split in ("train", "valid", "test")
        }
        for split in embedded:
            if embedded[split][2] != frame.loc[frame.split == split, "path"].tolist():
                raise ValueError(f"embedding cache {split} paths differ from manifest")
        metadata = {"embedding_cache": str(Path(args.embedding_cache).resolve())}
    else:
        model_args = SimpleNamespace(
            model_name=args.model_name,
            image_size=None,
            proj_hidden_dim=None,
            proj_output_dim=None,
            backbone_only=args.embedding == "backbone",
        )
        model, image_size, metadata = load_stage1_metric_model(checkpoint, model_args, device)
        embedded = {
            split: embed_split(
                model,
                manifest,
                class_to_idx,
                image_size,
                args.batch_size,
                device,
                split,
                args.embedding == "projection",
            )
            for split in ("train", "valid", "test")
        }
    train_x, train_y, train_paths = embedded["train"]
    valid_x, valid_y, valid_paths = embedded["valid"]
    test_x, test_y, test_paths = embedded["test"]
    prototype_indices = np.arange(len(train_y))
    if args.prototype_shots is None:
        prototype_x, prototype_y = train_x, train_y
    else:
        if args.prototype_shots < 1 or np.bincount(train_y).min() < args.prototype_shots:
            raise ValueError("--prototype-shots must be positive and available for every class")
        rng = np.random.default_rng(args.seed)
        prototype_indices = np.concatenate(
            [
                rng.choice(np.flatnonzero(train_y == index), args.prototype_shots, replace=False)
                for index in range(len(labels))
            ]
        )
        prototype_x, prototype_y = train_x[prototype_indices], train_y[prototype_indices]
    covariance_x, covariance_y = (
        (prototype_x, prototype_y) if args.covariance_from_prototypes else (train_x, train_y)
    )
    scorers = candidate_scorers(
        covariance_x,
        covariance_y,
        len(labels),
        args.seed,
        prototype_x,
        prototype_y,
        include_experimental=args.fixed_method is not None,
    )

    if args.fixed_method is not None:
        selected = next((item for item in scorers if item[0] == args.fixed_method), None)
        if selected is None:
            raise ValueError(f"unknown --fixed-method: {args.fixed_method}")
        selected_name = args.fixed_method
        validation = pd.DataFrame(columns=["method", "n", "accuracy", "macro_f1", "balanced_accuracy"])
    else:
        validation_rows = []
        for name, config, scorer in scorers:
            metrics, _ = score_metrics(valid_y, scorer(valid_x))
            validation_rows.append({"method": name, **config, **metrics})
        validation = pd.DataFrame(validation_rows).sort_values(
            ["macro_f1", "balanced_accuracy", "accuracy", "method"], ascending=[False, False, False, True]
        )
        selected_name = str(validation.iloc[0].method)
        selected = next(item for item in scorers if item[0] == selected_name)
    baseline = scorers[0]

    test_rows = []
    per_class_rows = []
    predictions = []
    used_paths = np.asarray(train_paths)[prototype_indices] if args.covariance_from_prototypes else train_paths
    train_hashes = {sha256(path) for path in used_paths}
    duplicate_free = np.asarray([sha256(path) not in train_hashes for path in test_paths])
    for name, config, scorer in {baseline[0]: baseline, selected[0]: selected}.values():
        scores = scorer(test_x)
        metrics, prediction = score_metrics(test_y, scores)
        test_rows.append({"method": name, "split": "test", **config, **metrics})
        clean_metrics, _ = score_metrics(test_y[duplicate_free], scorer(test_x[duplicate_free]))
        test_rows.append(
            {
                "method": name,
                "split": "test_duplicate_free",
                "excluded_exact_duplicates": int((~duplicate_free).sum()),
                **config,
                **clean_metrics,
            }
        )
        precision, recall, f1, support = precision_recall_fscore_support(
            test_y, prediction, labels=np.arange(len(labels)), average=None, zero_division=0
        )
        per_class_rows.extend(
            {
                "method": name,
                "label": label,
                "precision": precision[index],
                "recall": recall[index],
                "f1": f1[index],
                "support": support[index],
            }
            for index, label in enumerate(labels)
        )
        if name == selected_name:
            predictions = [
                {
                    "path": path,
                    "true_label": labels[truth],
                    "predicted_label": labels[predicted],
                    "correct": truth == predicted,
                }
                for path, truth, predicted in zip(test_paths, test_y, prediction)
            ]
            pd.DataFrame(
                confusion_matrix(test_y, prediction, labels=np.arange(len(labels))), index=labels, columns=labels
            ).to_csv(output / "confusion_test_selected.csv")

    np.savez_compressed(
        output / "embedding_cache.npz",
        train_x=train_x,
        train_y=train_y,
        train_paths=np.asarray(train_paths),
        valid_x=valid_x,
        valid_y=valid_y,
        valid_paths=np.asarray(valid_paths),
        test_x=test_x,
        test_y=test_y,
        test_paths=np.asarray(test_paths),
        labels=np.asarray(labels),
    )
    validation.to_csv(output / "validation_candidates.csv", index=False)
    pd.DataFrame(test_rows).to_csv(output / "test_selected_vs_baseline.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(output / "test_per_class.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "test_predictions_selected.csv", index=False)
    pd.DataFrame(
        {
            "path": np.asarray(train_paths)[prototype_indices],
            "label": np.asarray(labels)[prototype_y],
        }
    ).to_csv(output / "prototype_membership.csv", index=False)
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "manifest": str(manifest),
                "manifest_sha256": sha256(manifest),
                "embedding": args.embedding,
                "device": device_name,
                "model_metadata": metadata,
                "fit_data": "prototype samples only" if args.covariance_from_prototypes else "target train only",
                "prototype": "class mean",
                "prototype_shots_per_class": args.prototype_shots or "all",
                "covariance_samples": (
                    f"{args.prototype_shots} prototype images per class"
                    if args.covariance_from_prototypes
                    else "all target-train images"
                ),
                "selection_data": "not used; method fixed" if args.fixed_method else "target validation only",
                "test_used_for_selection": False,
                "selection_metric": None if args.fixed_method else "validation macro_f1; balanced_accuracy, accuracy tie-break",
                "candidate_count": len(scorers),
                "selected_method": selected_name,
                "excluded_exact_duplicates": int((~duplicate_free).sum()),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("target validation not used" if validation.empty else validation.head(10).to_string(index=False))
    print(pd.DataFrame(test_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
