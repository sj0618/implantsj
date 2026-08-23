"""Extract embeddings and run prototype-gallery evaluation."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from .contracts import validate_partition
from .models import ModelSpec, choose_device, create_adapter, load_model_spec


@dataclass
class EmbeddingTable:
    features: np.ndarray
    labels: np.ndarray
    paths: np.ndarray
    splits: np.ndarray
    sets: np.ndarray
    metadata: dict[str, Any]

    def validate(self) -> "EmbeddingTable":
        row_count = len(self.labels)
        if self.features.ndim != 2 or any(
            len(values) != row_count
            for values in (self.features, self.paths, self.splits, self.sets)
        ):
            raise ValueError(
                "embedding arrays must have the same row count and features must be 2-D"
            )
        if row_count == 0 or not np.isfinite(self.features).all():
            raise ValueError("embeddings must be non-empty and finite")
        if len(set(self.paths.tolist())) != row_count:
            raise ValueError("embedding paths must be unique")
        if any(not str(label).strip() for label in self.labels):
            raise ValueError("embedding labels must be non-empty")
        return self

    def save(self, path: str | Path) -> Path:
        output = Path(path).expanduser().resolve()
        if output.suffix != ".npz":
            output = output.with_suffix(".npz")
        self.validate()
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            features=self.features.astype(np.float32),
            labels=self.labels.astype(str),
            paths=self.paths.astype(str),
            splits=self.splits.astype(str),
            sets=self.sets.astype(str),
        )
        output.with_suffix(".json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingTable":
        source = Path(path).expanduser().resolve()
        arrays = np.load(source, allow_pickle=False)
        metadata_path = source.with_suffix(".json")
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        count = len(arrays["labels"])
        return cls(
            features=np.asarray(arrays["features"], dtype=np.float32),
            labels=np.asarray(arrays["labels"], dtype=str),
            paths=np.asarray(arrays["paths"], dtype=str),
            splits=np.asarray(arrays["splits"], dtype=str)
            if "splits" in arrays
            else np.full(count, "", dtype=str),
            sets=np.asarray(arrays["sets"], dtype=str)
            if "sets" in arrays
            else np.full(count, "", dtype=str),
            metadata=metadata,
        ).validate()


class _EmbeddingDataset:
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, str(row.label), str(row.path), str(row.split), str(row.set)


def read_embedding_manifest(path: str | Path) -> pd.DataFrame:
    manifest_path = Path(path).expanduser().resolve()
    frame = pd.read_csv(manifest_path)
    if not {"path", "label"}.issubset(frame.columns):
        raise ValueError(f"manifest {manifest_path} must contain path,label")
    if frame.empty or frame[["path", "label"]].isna().any().any():
        raise ValueError(
            f"manifest {manifest_path} path,label values must be non-empty"
        )
    if any(
        frame[column].astype(str).str.strip().eq("").any()
        for column in ("path", "label")
    ):
        raise ValueError(
            f"manifest {manifest_path} path,label values must be non-empty"
        )
    frame = frame.copy()

    def resolve(value: object) -> str:
        image_path = Path(str(value)).expanduser()
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        return str(image_path.resolve())

    frame["path"] = frame["path"].map(resolve)
    if frame["path"].duplicated().any():
        raise ValueError(f"manifest {manifest_path} contains duplicate paths")
    if "split" not in frame:
        frame["split"] = ""
    if "set" not in frame:
        frame["set"] = ""
    return frame


def _embedding_diagnostics(
    features: np.ndarray, seed: int = 2026
) -> dict[str, float | int]:
    norms = np.linalg.norm(features, axis=1)
    dimension_std = features.std(axis=0)
    rng = np.random.default_rng(seed)
    count = min(len(features), 1000)
    indices = rng.choice(len(features), size=count, replace=False)
    normalized = features[indices] / np.maximum(
        np.linalg.norm(features[indices], axis=1, keepdims=True), 1e-12
    )
    if len(normalized) > 1:
        cosine = normalized @ normalized.T
        upper = cosine[np.triu_indices(len(normalized), k=1)]
        mean_pairwise_cosine = float(upper.mean())
    else:
        mean_pairwise_cosine = float("nan")
    return {
        "samples": int(len(features)),
        "dimension": int(features.shape[1]),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "mean_dimension_std": float(dimension_std.mean()),
        "near_zero_dimensions": int((dimension_std < 1e-8).sum()),
        "sampled_mean_pairwise_cosine": mean_pairwise_cosine,
    }


def extract_embeddings(
    *,
    model_spec_path: str | Path | None = None,
    model_name: str | None = None,
    model: str | None = None,
    adapter: str = "timm_pretrained",
    pretrained: bool = True,
    embedding_source: str = "backbone",
    input_size: int = 224,
    projection_dim: int = 128,
    projection_hidden_dim: int = 512,
    num_classes: int | None = None,
    manifest_path: str | Path,
    output_path: str | Path,
    checkpoint_path: str | Path | None = None,
    checkpoint_format: str = "auto",
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = "auto",
    overwrite: bool = False,
    canonical_checkpoint: Mapping[str, Any] | None = None,
) -> Path:
    import torch

    output = Path(output_path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"embedding cache exists: {output}; pass --overwrite to replace"
        )
    frame = read_embedding_manifest(manifest_path)
    resolved_device = choose_device(device)
    if model_spec_path:
        spec = load_model_spec(model_spec_path, model_name)
    else:
        if not model:
            raise ValueError("either model_spec_path or model is required")
        spec = ModelSpec(
            name=model_name
            or (Path(checkpoint_path).stem if checkpoint_path else model),
            adapter=adapter,
            architecture=model,
            checkpoint=str(Path(checkpoint_path).expanduser().resolve())
            if checkpoint_path
            else None,
            checkpoint_format=checkpoint_format,
            pretrained=pretrained,
            embedding_source=embedding_source,
            input_size=input_size,
            projection_dim=projection_dim,
            projection_hidden_dim=projection_hidden_dim,
            extra={
                **({"num_classes": num_classes} if num_classes else {}),
                **(
                    {"canonical_checkpoint": dict(canonical_checkpoint)}
                    if canonical_checkpoint is not None
                    else {}
                ),
            },
        )
    if model_spec_path and checkpoint_path:
        spec = replace(
            spec,
            checkpoint=str(Path(checkpoint_path).expanduser().resolve()),
            checkpoint_format=checkpoint_format,
        )
    adapter = create_adapter(spec, resolved_device).load()
    dataset = _EmbeddingDataset(frame, adapter.transform())
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    feature_batches: list[np.ndarray] = []
    labels: list[str] = []
    paths: list[str] = []
    splits: list[str] = []
    sets: list[str] = []
    for images, batch_labels, batch_paths, batch_splits, batch_sets in loader:
        feature_batches.append(adapter.encode(images).numpy().astype(np.float32))
        labels.extend(batch_labels)
        paths.extend(batch_paths)
        splits.extend(batch_splits)
        sets.extend(batch_sets)
    features = np.concatenate(feature_batches, axis=0)
    metadata = {
        "model": adapter.metadata(),
        "model_name": spec.name,
        "model_spec_path": str(Path(model_spec_path).expanduser().resolve())
        if model_spec_path
        else None,
        "manifest_path": str(Path(manifest_path).expanduser().resolve()),
        "device": resolved_device,
        "diagnostics": _embedding_diagnostics(features),
    }
    table = EmbeddingTable(
        features=features,
        labels=np.asarray(labels, dtype=str),
        paths=np.asarray(paths, dtype=str),
        splits=np.asarray(splits, dtype=str),
        sets=np.asarray(sets, dtype=str),
        metadata=metadata,
    )
    return table.save(output)


def load_labels(path: str | Path) -> tuple[list[str], list[str], list[str]]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    all_labels = [str(value) for value in payload["all"]]
    closed = [str(value) for value in payload["closed"]]
    opened = [str(value) for value in payload["open"]]
    validate_partition(all_labels, closed, opened, strict=False)
    return all_labels, closed, opened


def require_coverage(table: EmbeddingTable, labels: Iterable[str], name: str) -> None:
    expected = set(labels)
    present = set(table.labels.tolist())
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    if missing or extra:
        raise ValueError(
            f"{name} label coverage mismatch; missing={missing}, extra={extra}"
        )


def _ss_indices(
    indices: np.ndarray,
    label: str,
    seed: int,
    count: int | None,
    fraction: float | None,
) -> np.ndarray:
    if count is not None and count <= 0:
        raise ValueError("ss_count must be positive")
    if fraction is not None and not 0.0 < fraction <= 1.0:
        raise ValueError("ss_fraction must be in (0,1]")
    if count is None:
        size = max(
            1,
            int(math.ceil(len(indices) * (fraction if fraction is not None else 0.25))),
        )
    else:
        size = min(count, len(indices))
    rng = np.random.default_rng(np.random.SeedSequence([seed, *label.encode("utf-8")]))
    return np.sort(rng.choice(indices, size=size, replace=False))


def build_prototypes(
    table: EmbeddingTable,
    labels: Sequence[str],
    method: str,
    *,
    seed: int = 2026,
    ss_count: int | None = 5,
    ss_fraction: float | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    require_coverage(table, labels, f"{method} prototype source")
    prototypes: list[np.ndarray] = []
    memberships: list[dict[str, Any]] = []
    for label in labels:
        indices = np.flatnonzero(table.labels == label)
        if method == "ss":
            selected = _ss_indices(indices, label, seed, ss_count, ss_fraction)
        elif method in {"mean", "gt"}:
            selected = indices
        else:
            raise ValueError("prototype method must be gt, mean or ss")
        prototypes.append(table.features[selected].mean(axis=0))
        for index in selected:
            memberships.append(
                {
                    "prototype": method,
                    "label": label,
                    "path": table.paths[index],
                    "source_index": int(index),
                }
            )
    return np.stack(prototypes).astype(np.float32), pd.DataFrame(memberships)


def predict_42way(
    query_features: np.ndarray,
    prototypes: np.ndarray,
    labels: Sequence[str],
    distance: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(labels) < 2:
        raise ValueError("prediction requires at least two gallery labels")
    if prototypes.shape[0] != len(labels):
        raise ValueError("one prototype is required for every gallery label")
    if query_features.shape[1] != prototypes.shape[1]:
        raise ValueError("query and prototype embedding dimensions differ")
    label_array = np.asarray(labels, dtype=str)
    if distance == "euclidean":
        matrix = np.sqrt(
            np.maximum(
                ((query_features[:, None, :] - prototypes[None, :, :]) ** 2).sum(
                    axis=2
                ),
                0.0,
            )
        )
        order = np.argsort(matrix, axis=1)
        best = order[:, 0]
        decision = matrix[np.arange(len(matrix)), best]
        margin = matrix[np.arange(len(matrix)), order[:, 1]] - decision
    elif distance == "cosine":
        query_norm = query_features / np.maximum(
            np.linalg.norm(query_features, axis=1, keepdims=True), 1e-12
        )
        prototype_norm = prototypes / np.maximum(
            np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12
        )
        matrix = query_norm @ prototype_norm.T
        order = np.argsort(-matrix, axis=1)
        best = order[:, 0]
        decision = matrix[np.arange(len(matrix)), best]
        margin = decision - matrix[np.arange(len(matrix)), order[:, 1]]
    else:
        raise ValueError("distance must be euclidean or cosine")
    return label_array[best], decision.astype(float), margin.astype(float)


def classification_metrics(
    truth: np.ndarray, predictions: np.ndarray, group_labels: Sequence[str]
) -> dict[str, float | int]:
    from sklearn.metrics import precision_recall_fscore_support

    mask = np.isin(truth, np.asarray(group_labels, dtype=str))
    selected_truth = truth[mask]
    selected_predictions = predictions[mask]
    if len(selected_truth) == 0:
        return {
            "n_samples": 0,
            "n_classes_present": 0,
            "accuracy": float("nan"),
            "macro_precision": float("nan"),
            "macro_recall": float("nan"),
            "macro_f1": float("nan"),
        }
    precision, recall, f1, _ = precision_recall_fscore_support(
        selected_truth,
        selected_predictions,
        labels=list(group_labels),
        average="macro",
        zero_division=0,
    )
    return {
        "n_samples": int(len(selected_truth)),
        "n_classes_present": int(len(set(selected_truth.tolist()))),
        "accuracy": float((selected_truth == selected_predictions).mean()),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def prototype_geometry(
    table: EmbeddingTable, prototypes: np.ndarray, labels: Sequence[str]
) -> dict[str, float | int]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    own = np.asarray([label_to_index[label] for label in table.labels])
    intra = np.linalg.norm(table.features - prototypes[own], axis=1)
    if len(prototypes) > 1:
        pairwise = np.linalg.norm(
            prototypes[:, None, :] - prototypes[None, :, :], axis=2
        )
        upper = pairwise[np.triu_indices(len(prototypes), k=1)]
        normalized = prototypes / np.maximum(
            np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12
        )
        cosine = normalized @ normalized.T
        cosine_upper = cosine[np.triu_indices(len(prototypes), k=1)]
    else:
        upper = np.asarray([np.nan])
        cosine_upper = np.asarray([np.nan])
    norms = np.linalg.norm(table.features, axis=1)
    dimension_std = table.features.std(axis=0)
    return {
        "mean_intra_class_euclidean": float(intra.mean()),
        "mean_inter_prototype_euclidean": float(upper.mean()),
        "mean_inter_prototype_cosine": float(cosine_upper.mean()),
        "separation_ratio": float(upper.mean() / max(intra.mean(), 1e-12)),
        "embedding_norm_mean": float(norms.mean()),
        "embedding_norm_std": float(norms.std()),
        "mean_dimension_std": float(dimension_std.mean()),
        "near_zero_dimension_count": int((dimension_std < 1e-8).sum()),
    }


def evaluate(args: argparse.Namespace) -> Path:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_labels, closed, opened = load_labels(args.labels_json)
    if not args.allow_noncanonical_labels and (
        len(all_labels),
        len(closed),
        len(opened),
    ) != (42, 27, 15):
        raise ValueError(
            f"canonical evaluation requires 42=27+15 labels, got {len(all_labels)}={len(closed)}+{len(opened)}"
        )
    support = EmbeddingTable.load(args.support_embeddings)
    query = EmbeddingTable.load(args.query_embeddings)
    require_coverage(support, all_labels, "support")
    if not args.allow_incomplete_query:
        require_coverage(query, all_labels, "query")
    elif set(query.labels) - set(all_labels):
        raise ValueError("query contains labels outside labels.json")
    gt = EmbeddingTable.load(args.gt_embeddings) if args.gt_embeddings else None

    methods = [value.strip() for value in args.prototypes.split(",") if value.strip()]
    distances = [value.strip() for value in args.distances.split(",") if value.strip()]
    invalid_methods = sorted(set(methods) - {"gt", "mean", "ss"})
    invalid_distances = sorted(set(distances) - {"euclidean", "cosine"})
    if invalid_methods or invalid_distances or not methods or not distances:
        raise ValueError(
            f"invalid evaluation axes; prototypes={invalid_methods}, distances={invalid_distances}"
        )
    if "gt" in methods and gt is None:
        raise ValueError("--gt-embeddings is required when prototypes include gt")
    if gt is not None:
        require_coverage(gt, all_labels, "GT")
        insufficient = pd.Series(gt.labels).value_counts()
        insufficient = insufficient[insufficient < 2].to_dict()
        if insufficient:
            raise ValueError(
                "GT prototype requires at least two independent images per label: "
                f"{insufficient}"
            )
        overlap = sorted(
            set(gt.paths.tolist())
            & (set(support.paths.tolist()) | set(query.paths.tolist()))
        )
        if overlap:
            raise ValueError(
                "GT prototype images must be independent from support/query data: "
                f"{overlap[:10]}"
            )

    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    per_class_rows: list[pd.DataFrame] = []
    membership_rows: list[pd.DataFrame] = []
    geometry_rows: list[dict[str, Any]] = []
    effective_ss_count = (
        args.ss_count
        if args.ss_count is not None
        else (None if args.ss_fraction else 5)
    )
    for method in methods:
        source = gt if method == "gt" else support
        assert source is not None
        prototypes, memberships = build_prototypes(
            source,
            all_labels,
            method,
            seed=args.seed,
            ss_count=effective_ss_count,
            ss_fraction=args.ss_fraction,
        )
        memberships.insert(0, "model", args.model_name)
        membership_rows.append(memberships)
        geometry_rows.append(
            {
                "model": args.model_name,
                "prototype": method,
                **prototype_geometry(source, prototypes, all_labels),
            }
        )
        np.save(output_dir / f"prototypes_{method}.npy", prototypes)
        for distance in distances:
            predictions, values, margins = predict_42way(
                query.features, prototypes, all_labels, distance
            )
            prediction_frame = pd.DataFrame(
                {
                    "model": args.model_name,
                    "query": args.query_name,
                    "prototype": method,
                    "distance": distance,
                    "path": query.paths,
                    "true_label": query.labels,
                    "predicted_label": predictions,
                    "true_set": [
                        "closed" if label in set(closed) else "open"
                        for label in query.labels
                    ],
                    "correct": query.labels == predictions,
                    "decision_value": values,
                    "top2_margin": margins,
                }
            )
            prediction_rows.append(prediction_frame)
            for group, group_labels in (
                ("all", all_labels),
                ("closed", closed),
                ("open", opened),
            ):
                row = {
                    "model": args.model_name,
                    "query": args.query_name,
                    "prototype": method,
                    "distance": distance,
                    "group": group,
                    "gallery_classes": len(all_labels),
                    **classification_metrics(query.labels, predictions, group_labels),
                }
                metrics_rows.append(row)
            precision, recall, f1, support_counts = precision_recall_fscore_support(
                query.labels,
                predictions,
                labels=all_labels,
                average=None,
                zero_division=0,
            )
            per_class_rows.append(
                pd.DataFrame(
                    {
                        "model": args.model_name,
                        "query": args.query_name,
                        "prototype": method,
                        "distance": distance,
                        "label": all_labels,
                        "set": [
                            "closed" if label in set(closed) else "open"
                            for label in all_labels
                        ],
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "support": support_counts,
                    }
                )
            )
            matrix = confusion_matrix(query.labels, predictions, labels=all_labels)
            pd.DataFrame(matrix, index=all_labels, columns=all_labels).to_csv(
                output_dir / f"confusion_{args.query_name}_{method}_{distance}.csv"
            )

    metrics_path = output_dir / "metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        output_dir / "predictions.csv.gz", index=False, compression="gzip"
    )
    pd.concat(per_class_rows, ignore_index=True).to_csv(
        output_dir / "per_class_metrics.csv", index=False
    )
    pd.concat(membership_rows, ignore_index=True).to_csv(
        output_dir / "prototype_membership.csv", index=False
    )
    pd.DataFrame(geometry_rows).to_csv(output_dir / "geometry.csv", index=False)
    metadata = {
        "model": args.model_name,
        "query": args.query_name,
        "labels_json": str(Path(args.labels_json).expanduser().resolve()),
        "support_embeddings": str(Path(args.support_embeddings).expanduser().resolve()),
        "gt_embeddings": str(Path(args.gt_embeddings).expanduser().resolve())
        if args.gt_embeddings
        else None,
        "query_embeddings": str(Path(args.query_embeddings).expanduser().resolve()),
        "gallery_size": len(all_labels),
        "closed_labels": len(closed),
        "open_labels": len(opened),
        "query_coverage": sorted(set(query.labels.tolist())),
        "prototypes": methods,
        "distances": distances,
        "seed": args.seed,
        "ss_count": effective_ss_count,
        "ss_fraction": args.ss_fraction,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics_path


def _add_embed_parser(subparsers) -> None:
    parser = subparsers.add_parser("embed", help="extract one manifest with one model")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-spec")
    source.add_argument(
        "--model", help="timm model architecture, for example vit_base_patch16_224"
    )
    parser.add_argument("--model-name")
    parser.add_argument(
        "--adapter",
        default="timm_pretrained",
        help="timm_pretrained, classifier_checkpoint, or supcon_checkpoint",
    )
    parser.add_argument(
        "--pretrained", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--embedding-source", choices=["backbone", "projection"], default="backbone"
    )
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--projection-hidden-dim", type=int, default=512)
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--checkpoint-format",
        choices=["auto", "ckpt", "safetensors"],
        default="auto",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")


def _add_evaluate_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "evaluate", help="evaluate GT/mean/SS × euclidean/cosine"
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--support-embeddings", required=True)
    parser.add_argument("--gt-embeddings")
    parser.add_argument("--query-embeddings", required=True)
    parser.add_argument("--labels-json", required=True)
    parser.add_argument(
        "--query-name", default="test", choices=["valid", "test", "train", "custom"]
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prototypes", default="gt,mean,ss")
    parser.add_argument("--distances", default="euclidean,cosine")
    ss = parser.add_mutually_exclusive_group()
    ss.add_argument("--ss-count", type=int)
    ss.add_argument("--ss-fraction", type=float)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--allow-incomplete-query", action="store_true")
    parser.add_argument(
        "--allow-noncanonical-labels",
        action="store_true",
        help="allow a gallery other than the canonical 42=27+15 labels",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_embed_parser(subparsers)
    _add_evaluate_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "embed":
        output = extract_embeddings(
            model_spec_path=args.model_spec,
            model_name=args.model_name,
            model=args.model,
            adapter=args.adapter,
            pretrained=args.pretrained,
            embedding_source=args.embedding_source,
            input_size=args.input_size,
            projection_dim=args.projection_dim,
            projection_hidden_dim=args.projection_hidden_dim,
            num_classes=args.num_classes,
            manifest_path=args.manifest,
            output_path=args.output,
            checkpoint_path=args.checkpoint,
            checkpoint_format=args.checkpoint_format,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            overwrite=args.overwrite,
        )
    else:
        output = evaluate(args)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
