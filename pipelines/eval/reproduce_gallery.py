"""Re-run gallery and few-shot report protocols with explicit support membership."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from tools.improve_unseen_mahalanobis import candidate_scorers, class_means, normalize_rows, sha256

SPLITS = ("train", "valid", "test")
SEEDS = (0, 1, 2, 3, 4, 42)


def validate_cache(cache):
    names = cache["labels"].astype(str)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("Cache needs at least two distinct class labels")
    dimensions = set()
    for split in SPLITS:
        x, y, paths = (cache[f"{split}_{key}"] for key in ("x", "y", "paths"))
        if x.ndim != 2 or not len(x) or len(x) != len(y) or len(x) != len(paths):
            raise ValueError(f"Invalid {split} feature/label/path dimensions")
        if y.ndim != 1 or y.dtype.kind not in "iu" or y.min() < 0 or y.max() >= len(names):
            raise ValueError(f"Invalid {split} class indices")
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite {split} embeddings")
        if len(set(paths.astype(str))) != len(paths):
            raise ValueError(f"Duplicate {split} paths")
        dimensions.add(x.shape[1])
    if len(dimensions) != 1 or 0 in dimensions:
        raise ValueError("Feature dimensions differ across splits")
    if set(cache["train_y"]) != set(range(len(names))):
        raise ValueError("Train gallery does not cover every class")


def extract_cache(checkpoint, manifest, embedding, model_name, batch_size, device_name, normalization):
    import torch
    from pipelines.eval.evaluate_3label_metric_from_7label import ImagePathDataset, load_stage1_metric_model

    frame = pd.read_csv(manifest)
    if not {"path", "label", "split"} <= set(frame) or frame[["path", "label", "split"]].isna().any().any():
        raise ValueError("Manifest requires nonempty path, label and split columns")
    if set(frame.split) != set(SPLITS):
        raise ValueError("Manifest must contain exactly train, valid, test")
    names = sorted(frame.label.astype(str).unique())
    device = torch.device(device_name if device_name != "auto" else
                          "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args = SimpleNamespace(model_name=model_name, image_size=None, proj_hidden_dim=None,
                           proj_output_dim=None, backbone_only=embedding == "backbone")
    model, size, metadata = load_stage1_metric_model(Path(checkpoint), args, device)
    missing_backbone = [key for key in metadata["missing_keys"] if key.startswith("backbone.")]
    if missing_backbone or (embedding == "projection" and metadata["missing_keys"]):
        raise ValueError(f"Checkpoint is incomplete for {embedding}: {metadata['missing_keys']}")
    if embedding == "projection" and not metadata["use_projection"]:
        raise ValueError("Checkpoint has no trained projection head")
    cache = {"labels": np.asarray(names)}
    for split in SPLITS:
        dataset = ImagePathDataset(Path(manifest), split, {v: i for i, v in enumerate(names)}, size)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        features, labels, paths = [], [], []
        with torch.no_grad():
            for images, targets, filenames in loader:
                images = images.to(device)
                values = model.backbone(images)
                if embedding == "projection":
                    values = model.projector(values) if hasattr(model, "projector") else model.projection(values)
                features.append(values.cpu().numpy())
                labels.extend(targets.tolist())
                paths.extend(filenames)
        if not features:
            raise ValueError(f"Empty {split} split")
        cache.update({f"{split}_x": np.concatenate(features), f"{split}_y": np.asarray(labels),
                      f"{split}_paths": np.asarray(paths)})
    return cache, {"checkpoint_sha256": sha256(checkpoint), "manifest_sha256": sha256(manifest),
                   "model": metadata, "normalization": normalization}


def select_support(labels, paths, shots, seed, strict, order="row"):
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        if strict and len(indices) < shots:
            raise ValueError(f"Class {label} has fewer than {shots} support images")
        if order == "path":
            indices = indices[np.argsort(paths[indices].astype(str))]
        selected.extend(rng.choice(indices, min(shots, len(indices)), replace=False).tolist())
    return np.asarray(selected, dtype=int)


def metrics(truth, prediction):
    return {"n": len(truth), "accuracy": float(accuracy_score(truth, prediction)),
            "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, prediction))}


def evaluate(cache, output, *, protocol="gallery", policies=("mean",), shots=5, seeds=(42,),
             methods=("euclidean", "manhattan", "cosine", "pooled_ledoit_wolf"),
             covariance="train", select_on_valid=False, normalization="l2", sampling_order="row", exact_shots=False):
    validate_cache(cache)
    if shots < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Positive shots and distinct seeds required")
    if protocol in ("strict", "episodic") and (covariance != "support" or select_on_valid or tuple(policies) != ("ss",)):
        raise ValueError("Strict protocols require SS, support-only covariance, and fixed methods")
    if select_on_valid and "gt" in policies:
        raise ValueError("Validation cannot select a method when it also supplies prototypes")
    if normalization not in ("l2", "none"):
        raise ValueError("Unknown normalization")
    cache = dict(cache)
    for split in SPLITS:
        if normalization == "l2":
            cache[f"{split}_x"] = normalize_rows(cache[f"{split}_x"])
    names = cache["labels"].astype(str)
    all_paths = np.concatenate([cache[f"{split}_paths"] for split in SPLITS]).astype(str)
    if len(set(all_paths)) != len(all_paths):
        raise ValueError("Repeated image path across splits")
    # Image content, not just filenames, defines leakage at the support/query boundary.
    hashes = {path: sha256(path) for path in all_paths}
    if protocol == "episodic" and len(set(hashes.values())) != len(hashes):
        raise ValueError("Episodic pool contains exact duplicate images")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, predictions, memberships, selections, per_class = [], [], [], [], []
    for seed in seeds:
        for policy in policies:
            support_split = "valid" if policy == "gt" else "train"
            sx, sy, sp = (cache[f"{support_split}_{key}"] for key in ("x", "y", "paths"))
            query_splits = ("test",) if policy == "gt" or protocol != "gallery" else ("valid", "test")
            queries = {s: tuple(cache[f"{s}_{key}"] for key in ("x", "y", "paths")) for s in query_splits}
            if protocol == "episodic":
                sx, sy, sp = (np.concatenate([cache[f"{split}_{key}"] for split in SPLITS]) for key in ("x", "y", "paths"))
            selected = select_support(sy, sp, shots, seed, protocol != "gallery" or exact_shots, sampling_order) if policy == "ss" else np.arange(len(sy))
            if protocol == "episodic":
                mask = np.ones(len(sy), dtype=bool)
                mask[selected] = False
                queries = {"query": (sx[mask], sy[mask], sp[mask])}
            px, py, pp = (sx[selected], sy[selected], sp[selected]) if policy == "ss" else (sx, sy, sp)
            if set(py) != set(range(len(names))):
                raise ValueError("Prototype support does not cover every class")
            cx, cy = (px, py) if covariance == "support" else (cache["train_x"], cache["train_y"])
            fit_paths = pp if covariance == "support" else np.concatenate([pp, cache["train_paths"]])
            fit_hashes = {hashes[str(path)] for path in fit_paths}
            centers = class_means(px, py, len(names))
            candidates = candidate_scorers(cx, cy, len(names), seed, px, py,
                                          include_experimental=any("fractional" in m or "angular" in m for m in methods))
            scorers = {name: fn for name, _, fn in candidates}
            scorers["euclidean"] = lambda q: ((q[:, None] - centers[None]) ** 2).sum(2)
            scorers["manhattan"] = lambda q: np.abs(q[:, None] - centers[None]).sum(2)
            scorers["cosine"] = lambda q: 1 - normalize_rows(q) @ normalize_rows(centers).T
            selected_methods = list(methods)
            if select_on_valid:
                ranking = []
                for name, _, scorer in candidates:
                    value = metrics(queries["valid"][1], scorer(queries["valid"][0]).argmin(1))
                    ranking.append({"seed": seed, "prototype": policy, "method": name, **value})
                ranked = sorted(ranking, key=lambda v: (-v["macro_f1"], -v["balanced_accuracy"], -v["accuracy"], v["method"]))
                selections.extend(ranking)
                selected_methods = list(dict.fromkeys(["pooled_ledoit_wolf", ranked[0]["method"]]))
            if set(selected_methods) - set(scorers):
                raise ValueError(f"Unknown methods: {set(selected_methods) - set(scorers)}")
            memberships.extend({"seed": seed, "prototype": policy, "role": "support", "path": str(path),
                                "label": names[label], "sha256": hashes[str(path)]} for path, label in zip(pp, py))
            for split, (qx, qy, qp) in queries.items():
                if not len(qy):
                    raise ValueError("No query images remain")
                clean = np.asarray([hashes[str(path)] not in fit_hashes for path in qp])
                if protocol != "gallery" and not clean.all():
                    raise ValueError("Strict support/query contains exact image overlap")
                memberships.extend({"seed": seed, "prototype": policy, "role": split, "path": str(path),
                                    "label": names[label], "sha256": hashes[str(path)]} for path, label in zip(qp, qy))
                for method in selected_methods:
                    scores = scorers[method](qx)
                    if not np.isfinite(scores).all():
                        raise ValueError(f"Non-finite scores: {method}")
                    pred = scores.argmin(1)
                    base = {"seed": seed, "prototype": policy, "method": method}
                    rows.append({**base, "split": split, "excluded_duplicates": 0, **metrics(qy, pred)})
                    if split == "test" and not clean.all() and clean.any():
                        rows.append({**base, "split": "test_duplicate_free", "excluded_duplicates": int((~clean).sum()),
                                     **metrics(qy[clean], pred[clean])})
                    predictions.extend({**base, "split": split, "path": str(path), "true_label": names[y],
                                        "predicted_label": names[p]} for path, y, p in zip(qp, qy, pred))
                    precision, recall, f1, support = precision_recall_fscore_support(
                        qy, pred, labels=np.arange(len(names)), zero_division=0)
                    per_class.extend({**base, "split": split, "label": label, "precision": precision[i],
                                      "recall": recall[i], "f1": f1[i], "support": int(support[i])}
                                     for i, label in enumerate(names))
                    pd.DataFrame(confusion_matrix(qy, pred, labels=np.arange(len(names))), index=names, columns=names).to_csv(
                        output / f"confusion_seed{seed}_{policy}_{split}_{method}.csv")
    result = pd.DataFrame(rows)
    result.to_csv(output / "results.csv", index=False)
    result.groupby(["prototype", "split", "method"])[["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"]].agg(
        ["mean", "std", "min", "max", "count"]).to_csv(output / "summary.csv")
    pd.DataFrame(predictions).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(memberships).to_csv(output / "membership.csv", index=False)
    pd.DataFrame(per_class).to_csv(output / "per_class.csv", index=False)
    if selections:
        pd.DataFrame(selections).to_csv(output / "validation_candidates.csv", index=False)
    averages = result.groupby(["prototype", "split", "method"])[["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"]].mean()
    report = ["# 실험 재실행 결과", "", "현재 입력으로 계산한 값입니다. 과거 보고서 수치와의 일치는 별도 확인이 필요합니다.", "",
              f"- 평가: {protocol}; 공분산: {covariance}; seed: {list(seeds)}; 정규화: {normalization}",
              f"- 방법 선택: {'validation Macro-F1' if select_on_valid else '실행 전에 고정'}", "",
              "| Prototype | Query | Method | Accuracy | Macro-F1 | Weighted-F1 | Balanced accuracy |",
              "|---|---|---|---:|---:|---:|---:|"]
    for key, row in averages.iterrows():
        report.append("| " + " | ".join([*key, *[f"{100 * value:.4f}%" for value in row]]) + " |")
    (output / "REPORT.md").write_text("\n".join(report) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cache", type=Path)
    source.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--expected-classes", type=int)
    parser.add_argument("--model-name")
    parser.add_argument("--embedding", choices=("backbone", "projection"), default="backbone")
    parser.add_argument("--normalization", choices=("none", "l2"), required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--protocol", choices=("gallery", "strict", "episodic"), default="gallery")
    parser.add_argument("--policies", nargs="+", choices=("mean", "gt", "ss"), default=["mean"])
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--sampling-order", choices=("row", "path"), default="row")
    parser.add_argument("--exact-shots", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--methods", nargs="+", default=["euclidean", "manhattan", "cosine", "pooled_ledoit_wolf"])
    parser.add_argument("--covariance", choices=("train", "support"), default="train")
    parser.add_argument("--select-on-valid", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.cache:
        with np.load(args.cache, allow_pickle=False) as archive:
            cache = dict(archive)
        provenance = {"cache_sha256": sha256(args.cache), "checkpoint_identity": "not verified from cache alone"}
    else:
        if not args.manifest:
            parser.error("--manifest is required with --checkpoint")
        cache, provenance = extract_cache(args.checkpoint, args.manifest, args.embedding, args.model_name,
                                          args.batch_size, args.device, args.normalization)
    if args.expected_classes is not None and len(cache["labels"]) != args.expected_classes:
        raise ValueError(f"Expected {args.expected_classes} classes, got {len(cache['labels'])}")
    if args.source_manifest:
        source_frame = pd.read_csv(args.source_manifest)
        overlap = set(source_frame.label.astype(str).str.casefold()) & set(np.char.lower(cache["labels"].astype(str)))
        if overlap:
            raise ValueError(f"Source/target class overlap: {sorted(overlap)}")
        source_hashes = {sha256(path) for path in source_frame.path.astype(str)}
        target_hashes = {sha256(path) for split in SPLITS for path in cache[f"{split}_paths"]}
        if source_hashes & target_hashes:
            raise ValueError("Source/target manifests contain exact duplicate images")
        provenance["source_manifest_sha256"] = sha256(args.source_manifest)
    evaluate(cache, args.output_dir, protocol=args.protocol, policies=args.policies, shots=args.shots,
             seeds=args.seeds, methods=args.methods, covariance=args.covariance,
             select_on_valid=args.select_on_valid, normalization=args.normalization,
             sampling_order=args.sampling_order, exact_shots=args.exact_shots)
    np.savez_compressed(args.output_dir / "embedding_cache.npz", **cache)
    (args.output_dir / "protocol.json").write_text(json.dumps({
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "provenance": provenance, "patient_independence": "not verified",
        "historical_numeric_parity": "not verified", "test_used_for_selection": False,
    }, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
