#!/usr/bin/env python3
"""Build deterministic cached evidence for the experimental spectral scorers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from tools.improve_unseen_mahalanobis import candidate_scorers, normalize_rows


ROOT = Path(__file__).resolve().parents[1]
ANCHOR = Path(__file__).resolve().parents[2]
SEEDS = (0, 1, 2, 3, 4, 42)
METHODS = (
    "pooled_ledoit_wolf",
    "fractional_0.5",
    "fractional_0.75",
    "centered_angular_0.5",
    "centered_angular_0.75",
)
THREE_LABEL = {
    name: ROOT / "docs/reports/mahalanobis_improvement_20260813" / name / "embedding_cache.npz"
    for name in ("fresh_backbone", "fresh_projection", "legacy_backbone", "legacy_projection")
}
STRICT_ROOT = ROOT / "docs/reports/vit6supcon_to_5label_strict10_20260814"
STRICT_CACHE = STRICT_ROOT / "strict10/backbone_seed42/embedding_cache.npz"
STRICT_EPISODES = STRICT_ROOT / "episodic_all_target_5shot"
DEFAULT_FINAL = ROOT / "docs/reports/mahalanobis_spectrum_validation_20260816"
CONTEXT = ROOT / ".omx/context/mahalanobis-spectrum-validation-20260816T173955Z.md"
SOURCE = ROOT / "tools/improve_unseen_mahalanobis.py"
FILES = ("metrics.csv", "memberships.csv", "provenance.json", "VALIDATION.md", "SUCCESS.json")
METRICS = ("accuracy", "macro_f1", "balanced_accuracy")
METRIC_COLUMNS = (
    "condition", "seed", "method", "evaluation_view", "excluded_exact_duplicates", "query_count",
    "accuracy", "macro_f1", "balanced_accuracy", "delta_accuracy", "delta_macro_f1",
    "delta_balanced_accuracy", "regressed_accuracy", "regressed_macro_f1",
    "regressed_balanced_accuracy", "selected_status", "best_in_condition_seed_view", "cache_sha256",
    "condition_protocol_sha256", "manifest_sha256", "episode_protocol_sha256",
    "cache_generation_protocol_sha256", "membership_sha256", "membership_set_sha256",
    "source_pre_sha256", "source_post_sha256",
)
MEMBERSHIP_COLUMNS = ("condition", "seed", "position", "path", "label", "membership_sha256")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def json_hash(value: object) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def safe_path(value: str | Path, *, must_exist: bool = True) -> Path:
    path = Path(value).expanduser().resolve(strict=must_exist)
    try:
        path.relative_to(ANCHOR)
    except ValueError as error:
        raise ValueError(f"path escapes validation anchor: {path}") from error
    return path


def relative(path: str | Path) -> str:
    return safe_path(path).relative_to(ANCHOR).as_posix()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_cache(path: Path) -> dict[str, np.ndarray]:
    required = {
        "train_x", "train_y", "train_paths",
        "valid_x", "valid_y", "valid_paths",
        "test_x", "test_y", "test_paths", "labels",
    }
    with np.load(path, allow_pickle=False) as archive:
        if not required <= set(archive.files):
            raise ValueError(f"malformed embedding cache: {path}")
        cache = {key: archive[key] for key in required}
    labels = cache["labels"]
    if labels.ndim != 1 or not len(labels):
        raise ValueError(f"malformed cache labels: {path}")
    dimensions = set()
    for split in ("train", "valid", "test"):
        features = cache[f"{split}_x"]
        targets = cache[f"{split}_y"]
        paths = cache[f"{split}_paths"]
        if not (
            features.ndim == 2
            and targets.ndim == 1
            and paths.ndim == 1
            and len(features) == len(targets) == len(paths)
        ):
            raise ValueError(f"malformed cache split: {path} {split}")
        if not np.issubdtype(features.dtype, np.number) or not np.isfinite(features).all():
            raise ValueError(f"non-finite cache features: {path} {split}")
        if not np.issubdtype(targets.dtype, np.integer) or ((targets < 0) | (targets >= len(labels))).any():
            raise ValueError(f"invalid cache targets: {path} {split}")
        dimensions.add(features.shape[1])
    if len(dimensions) != 1 or not next(iter(dimensions)):
        raise ValueError(f"cache feature dimensions differ: {path}")
    return cache


def validate_cache(path: Path, protocol_path: Path, expected_embedding: str) -> tuple[dict, dict, Path, str]:
    path = safe_path(path)
    protocol_path = safe_path(protocol_path)
    protocol = load_json(protocol_path)
    manifest = safe_path(protocol["manifest"])
    if sha256(manifest) != protocol.get("manifest_sha256"):
        raise ValueError(f"manifest hash mismatch: {manifest}")
    if protocol.get("embedding") != expected_embedding:
        raise ValueError(f"cache/protocol embedding mismatch: {path}")

    cache = load_cache(path)
    frame = pd.read_csv(manifest)
    if not {"path", "label", "split"} <= set(frame.columns):
        raise ValueError(f"manifest columns are incomplete: {manifest}")
    labels = sorted(frame.label.astype(str).unique())
    if cache["labels"].astype(str).tolist() != labels:
        raise ValueError(f"cache labels differ from manifest: {path}")
    label_to_index = {label: index for index, label in enumerate(labels)}
    for split in ("train", "valid", "test"):
        rows = frame.loc[frame.split.astype(str) == split]
        manifest_paths = [relative(value) for value in rows.path.astype(str)]
        cache_paths = [relative(value) for value in cache[f"{split}_paths"].astype(str)]
        if cache_paths != manifest_paths:
            raise ValueError(f"cache {split} paths differ from manifest: {path}")
        expected_y = [label_to_index[value] for value in rows.label.astype(str)]
        if cache[f"{split}_y"].astype(int).tolist() != expected_y:
            raise ValueError(f"cache {split} labels differ from manifest: {path}")
    return cache, protocol, manifest, sha256(protocol_path)


def membership_bytes(rows: list[tuple[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("path", "label"))
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
    }


def selected_scorers(train_x: np.ndarray, train_y: np.ndarray, classes: int, seed: int) -> list[tuple[str, dict, object]]:
    available = candidate_scorers(train_x, train_y, classes, seed, include_experimental=True)
    by_name = {name: (name, config, scorer) for name, config, scorer in available}
    if not set(METHODS) <= set(by_name):
        raise ValueError("production scorer factory is missing required methods")
    return [by_name[name] for name in METHODS]


def scored_rows(
    *,
    condition: str,
    seed: int,
    evaluation_view: str,
    truth: np.ndarray,
    query_x: np.ndarray,
    scorers: list[tuple[str, dict, object]],
    excluded: int,
    hashes: dict[str, str],
) -> list[dict]:
    values = []
    for name, _, scorer in scorers:
        scores = np.asarray(scorer(query_x), dtype=np.float64)
        if scores.shape != (len(query_x), len(np.unique(truth))):
            raise ValueError(f"unexpected score shape for {condition}/{seed}/{name}: {scores.shape}")
        if not np.isfinite(scores).all():
            raise ValueError(f"non-finite scores for {condition}/{seed}/{name}")
        values.append((name, metrics(truth, scores.argmin(axis=1))))

    baseline = values[0][1]
    best_accuracy = max(value["accuracy"] for _, value in values)
    rows = []
    for name, value in values:
        deltas = {metric: value[metric] - baseline[metric] for metric in METRICS}
        rows.append(
            {
                "condition": condition,
                "seed": seed,
                "method": name,
                "evaluation_view": evaluation_view,
                "excluded_exact_duplicates": excluded,
                "query_count": len(truth),
                **value,
                **{f"delta_{metric}": deltas[metric] for metric in METRICS},
                **{f"regressed_{metric}": deltas[metric] < -1e-12 for metric in METRICS},
                "selected_status": "baseline" if name == METHODS[0] else "experimental_fixed_only",
                "best_in_condition_seed_view": value["accuracy"] >= best_accuracy - 1e-12
                and not any(
                    prior[1]["accuracy"] >= best_accuracy - 1e-12
                    for prior in values[: METHODS.index(name)]
                ),
                **hashes,
            }
        )
    return rows


def evaluate_three_label(condition: str, cache_path: Path, image_hashes: dict[Path, str]) -> tuple[list[dict], list[dict], dict]:
    embedding = condition.rsplit("_", 1)[1]
    protocol_path = cache_path.parent / "protocol.json"
    cache, protocol, manifest, protocol_hash = validate_cache(cache_path, protocol_path, embedding)
    labels = cache["labels"].astype(str).tolist()
    train_paths = [safe_path(value) for value in cache["train_paths"].astype(str)]
    test_paths = [safe_path(value) for value in cache["test_paths"].astype(str)]

    def image_hash(path: Path) -> str:
        if path not in image_hashes:
            image_hashes[path] = sha256(path)
        return image_hashes[path]

    train_content = {image_hash(path) for path in train_paths}
    duplicate_free = np.asarray([image_hash(path) not in train_content for path in test_paths])
    excluded = int((~duplicate_free).sum())
    if excluded != int(protocol["excluded_exact_duplicates"]):
        raise ValueError(f"duplicate count differs from protocol: {condition}")

    episode_protocol = {
        "support_pool": "cache train split",
        "query_pool": "complete cache test split",
        "support_shots_per_class": 10,
        "prototype_and_covariance_membership_identical": True,
        "seeds": list(SEEDS),
    }
    episode_hash = json_hash(episode_protocol)
    cache_hash = sha256(safe_path(cache_path))
    manifest_hash = sha256(manifest)
    membership_rows = []
    episodes = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        support = np.concatenate(
            [rng.choice(np.flatnonzero(cache["train_y"] == index), 10, replace=False) for index in range(len(labels))]
        )
        canonical = [(relative(train_paths[index]), labels[int(cache["train_y"][index])]) for index in support]
        membership_hash = hashlib.sha256(membership_bytes(canonical)).hexdigest()
        membership_rows.extend(
            {
                "condition": condition,
                "seed": seed,
                "position": position,
                "path": path,
                "label": label,
                "membership_sha256": membership_hash,
            }
            for position, (path, label) in enumerate(canonical)
        )
        episodes.append((seed, support, membership_hash))

    membership_set_hash = hashlib.sha256(
        "".join(f"{seed}:{membership_hash}\n" for seed, _, membership_hash in sorted(episodes)).encode()
    ).hexdigest()
    rows = []
    for seed, support, membership_hash in episodes:
        scorers = selected_scorers(cache["train_x"][support], cache["train_y"][support], len(labels), seed)
        common_hashes = {
            "cache_sha256": cache_hash,
            "condition_protocol_sha256": protocol_hash,
            "manifest_sha256": manifest_hash,
            "episode_protocol_sha256": episode_hash,
            "cache_generation_protocol_sha256": protocol_hash,
            "membership_sha256": membership_hash,
            "membership_set_sha256": membership_set_hash,
        }
        rows.extend(
            scored_rows(
                condition=condition,
                seed=seed,
                evaluation_view="all_rows",
                truth=cache["test_y"],
                query_x=cache["test_x"],
                scorers=scorers,
                excluded=0,
                hashes=common_hashes,
            )
        )
        rows.extend(
            scored_rows(
                condition=condition,
                seed=seed,
                evaluation_view="duplicate_free",
                truth=cache["test_y"][duplicate_free],
                query_x=cache["test_x"][duplicate_free],
                scorers=scorers,
                excluded=excluded,
                hashes=common_hashes,
            )
        )
    return rows, membership_rows, {
        "cache": relative(cache_path),
        "cache_sha256": cache_hash,
        "protocol": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "manifest": relative(manifest),
        "manifest_sha256": manifest_hash,
        "episode_protocol": episode_protocol,
        "episode_protocol_sha256": episode_hash,
        "membership_set_sha256": membership_set_hash,
        "excluded_exact_duplicates": excluded,
    }


def evaluate_strict(cache_path: Path, episodes_root: Path) -> tuple[list[dict], list[dict], dict]:
    cache_path = safe_path(cache_path)
    episodes_root = safe_path(episodes_root)
    cache_protocol_path = safe_path(cache_path.parent / "protocol.json")
    episode_protocol_path = safe_path(episodes_root / "protocol.json")
    cache, cache_protocol, manifest, cache_protocol_hash = validate_cache(
        cache_path, cache_protocol_path, "backbone"
    )
    episode_protocol = load_json(episode_protocol_path)
    labels = cache["labels"].astype(str).tolist()
    if episode_protocol.get("labels") != labels or episode_protocol.get("seeds") != list(SEEDS):
        raise ValueError("strict episode labels or seeds differ from protocol")
    if (
        episode_protocol.get("pooled_target_images") != 152
        or episode_protocol.get("support_total") != 25
        or episode_protocol.get("query_total") != 127
        or episode_protocol.get("support_shots_per_class") != 5
    ):
        raise ValueError("strict episode counts differ from protocol")

    features = np.concatenate([cache[f"{split}_x"] for split in ("train", "valid", "test")])
    targets = np.concatenate([cache[f"{split}_y"] for split in ("train", "valid", "test")])
    paths = [safe_path(value) for split in ("train", "valid", "test") for value in cache[f"{split}_paths"].astype(str)]
    canonical_paths = [relative(path) for path in paths]
    if len(set(canonical_paths)) != len(canonical_paths):
        raise ValueError("strict cache paths must be unique")
    if len(paths) != 152:
        raise ValueError("strict cache must contain 152 pooled rows")

    membership_rows = []
    episodes = []
    for seed in SEEDS:
        membership_path = safe_path(episodes_root / f"seed{seed}/support_membership.csv")
        membership = pd.read_csv(membership_path)
        if list(membership.columns) != ["path", "label"] or len(membership) != 25:
            raise ValueError(f"malformed strict membership: {membership_path}")
        stored = [(relative(path), str(label)) for path, label in membership.itertuples(index=False, name=None)]
        rng = np.random.default_rng(seed)
        support = np.concatenate(
            [rng.choice(np.flatnonzero(targets == index), 5, replace=False) for index in range(len(labels))]
        )
        regenerated = [(canonical_paths[index], labels[int(targets[index])]) for index in support]
        if stored != regenerated:
            raise ValueError(f"strict support membership mismatch for seed {seed}")
        membership_hash = hashlib.sha256(membership_bytes(stored)).hexdigest()
        membership_rows.extend(
            {
                "condition": "strict_5label_5shot",
                "seed": seed,
                "position": position,
                "path": path,
                "label": label,
                "membership_sha256": membership_hash,
            }
            for position, (path, label) in enumerate(stored)
        )
        episodes.append((seed, support, membership_hash))

    membership_set_hash = hashlib.sha256(
        "".join(f"{seed}:{membership_hash}\n" for seed, _, membership_hash in sorted(episodes)).encode()
    ).hexdigest()
    episode_protocol_hash = sha256(episode_protocol_path)
    cache_hash = sha256(cache_path)
    manifest_hash = sha256(manifest)
    rows = []
    for seed, support, membership_hash in episodes:
        query = np.ones(len(targets), dtype=bool)
        query[support] = False
        support_x = normalize_rows(features[support])
        query_x = normalize_rows(features[query])
        query_y = targets[query]
        observed_counts = {labels[index]: int((query_y == index).sum()) for index in range(len(labels))}
        if observed_counts != episode_protocol.get("query_counts"):
            raise ValueError(f"strict query counts differ for seed {seed}")
        scorers = selected_scorers(support_x, targets[support], len(labels), seed)
        rows.extend(
            scored_rows(
                condition="strict_5label_5shot",
                seed=seed,
                evaluation_view="all_rows",
                truth=query_y,
                query_x=query_x,
                scorers=scorers,
                excluded=0,
                hashes={
                    "cache_sha256": cache_hash,
                    "condition_protocol_sha256": episode_protocol_hash,
                    "manifest_sha256": manifest_hash,
                    "episode_protocol_sha256": episode_protocol_hash,
                    "cache_generation_protocol_sha256": cache_protocol_hash,
                    "membership_sha256": membership_hash,
                    "membership_set_sha256": membership_set_hash,
                },
            )
        )
    return rows, membership_rows, {
        "cache": relative(cache_path),
        "cache_sha256": cache_hash,
        "episode_protocol": relative(episode_protocol_path),
        "episode_protocol_sha256": episode_protocol_hash,
        "cache_generation_protocol": relative(cache_protocol_path),
        "cache_generation_protocol_sha256": cache_protocol_hash,
        "manifest": relative(manifest),
        "manifest_sha256": manifest_hash,
        "membership_set_sha256": membership_set_hash,
        "total": 152,
        "support": 25,
        "query": 127,
    }


def markdown_table(frame: pd.DataFrame) -> str:
    columns = ("condition", "evaluation_view", "method", "accuracy", "delta_accuracy", "macro_f1", "delta_macro_f1")
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        rendered = list(row[:3]) + [f"{100 * float(value):.2f}%" for value in row[3:]]
        lines.append("| " + " | ".join(map(str, rendered)) + " |")
    return "\n".join(lines)


def render_report(frame: pd.DataFrame) -> str:
    summary = (
        frame.groupby(["condition", "evaluation_view", "method"], sort=False)[
            ["accuracy", "delta_accuracy", "macro_f1", "delta_macro_f1"]
        ]
        .mean()
        .reset_index()
    )
    strict = summary.query("condition == 'strict_5label_5shot'").set_index("method")
    observed = max(float(strict.loc[name, "accuracy"]) for name in METHODS[-2:])
    difference = 100 * (observed - 0.6706)
    fresh = summary[summary.condition.str.startswith("fresh_")]
    legacy = summary[summary.condition.str.startswith("legacy_")]
    strict_rows = summary[summary.condition == "strict_5label_5shot"]
    return f"""# Mahalanobis spectrum validation

## Scope

This report evaluates four fixed experimental scorers against pooled Ledoit-Wolf on cached embeddings. The candidates were proposed after these caches had already been inspected, so every result below is exploratory post-selection evidence. It is not a fresh confirmatory estimate and it does not change automatic method selection.

The 3-label conditions use ten train supports per class for both prototypes and covariance, with the complete test split as query. Fresh-cache duplicate-free rows exclude exact train/test content duplicates and are shown beside the all-row results. The strict condition uses the stored five-support-per-class episodes, 25 supports and 127 queries for each seed.

## Fresh 3-label results

{markdown_table(fresh)}

## Legacy 3-label results

{markdown_table(legacy)}

## Strict 5-label 5-shot results

{markdown_table(strict_rows)}

## 67.06% statement

The captured source wording is: "the best angular candidate reaches 67.06% accuracy." The source does not define its aggregation formula. For this validation only, the operational interpretation is the larger of the two per-method six-seed mean strict accuracies. The observed value is {100 * observed:.2f}%, which differs from 67.06% by {difference:+.2f} percentage points. This is not the mean of per-seed maxima.

## Reproducibility boundary

All rows come from the production scorer factory with experimental enumeration explicitly enabled. Existing callers keep experimental enumeration disabled. The machine-readable evidence records cache, protocol, manifest, membership, source, and artifact hashes. A new locked dataset or nested development protocol is still required for confirmatory selection.
"""


def validate_matrix(frame: pd.DataFrame) -> None:
    key = ["condition", "seed", "method", "evaluation_view"]
    if tuple(frame.columns) != METRIC_COLUMNS or len(frame) != 270 or frame.duplicated(key).any():
        raise ValueError("metrics matrix must contain 270 unique rows")
    if len(frame.query("evaluation_view == 'all_rows'")) != 150:
        raise ValueError("all_rows matrix must contain 150 rows")
    if len(frame.query("evaluation_view == 'duplicate_free'")) != 120:
        raise ValueError("duplicate_free matrix must contain 120 rows")
    expected_conditions = set(THREE_LABEL) | {"strict_5label_5shot"}
    if set(frame.condition) != expected_conditions or set(frame.seed) != set(SEEDS) or set(frame.method) != set(METHODS):
        raise ValueError("metrics matrix has missing conditions, seeds, or methods")
    numeric = [*METRICS, *(f"delta_{metric}" for metric in METRICS)]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("metrics matrix contains non-finite values")
    boolean_columns = [*(f"regressed_{metric}" for metric in METRICS), "best_in_condition_seed_view"]
    if any(not pd.api.types.is_bool_dtype(frame[column]) for column in boolean_columns):
        raise ValueError("metrics matrix contains non-boolean flags")
    baseline = frame[frame.method == METHODS[0]]
    if not (baseline[[f"delta_{metric}" for metric in METRICS]].to_numpy() == 0).all():
        raise ValueError("baseline deltas must be exactly zero")
    for _, group in frame.groupby(["condition", "seed", "evaluation_view"], sort=False):
        by_method = group.set_index("method")
        baseline_values = by_method.loc[METHODS[0]]
        best_accuracy = max(float(by_method.loc[name, "accuracy"]) for name in METHODS)
        best_method = next(
            name for name in METHODS if float(by_method.loc[name, "accuracy"]) >= best_accuracy - 1e-12
        )
        for name in METHODS:
            row = by_method.loc[name]
            for metric in METRICS:
                expected_delta = float(row[metric]) - float(baseline_values[metric])
                if not np.isclose(float(row[f"delta_{metric}"]), expected_delta, rtol=0, atol=1e-15):
                    raise ValueError(f"metric delta mismatch: {name} {metric}")
                if bool(row[f"regressed_{metric}"]) != (expected_delta < -1e-12):
                    raise ValueError(f"metric regression flag mismatch: {name} {metric}")
            if row["selected_status"] != ("baseline" if name == METHODS[0] else "experimental_fixed_only"):
                raise ValueError(f"selected status mismatch: {name}")
            if bool(row["best_in_condition_seed_view"]) != (name == best_method):
                raise ValueError(f"best method flag mismatch: {name}")


def validate_memberships(memberships: pd.DataFrame, metric_frame: pd.DataFrame, provenance: dict) -> None:
    if (
        tuple(memberships.columns) != MEMBERSHIP_COLUMNS
        or len(memberships) != 870
        or memberships.duplicated(["condition", "seed", "position"]).any()
    ):
        raise ValueError("membership matrix schema mismatch")
    for condition in list(THREE_LABEL) + ["strict_5label_5shot"]:
        metadata = provenance["inputs"][condition]
        if condition == "strict_5label_5shot":
            episode_root = safe_path(ANCHOR / metadata["episode_protocol"]).parent
            expected_by_seed = {}
            for seed in SEEDS:
                source = pd.read_csv(safe_path(episode_root / f"seed{seed}/support_membership.csv"))
                expected_by_seed[seed] = [
                    (relative(path), str(label)) for path, label in source.itertuples(index=False, name=None)
                ]
        else:
            cache = load_cache(safe_path(ANCHOR / metadata["cache"]))
            labels = cache["labels"].astype(str).tolist()
            paths = [safe_path(value) for value in cache["train_paths"].astype(str)]
            expected_by_seed = {}
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                support = np.concatenate(
                    [
                        rng.choice(np.flatnonzero(cache["train_y"] == index), 10, replace=False)
                        for index in range(len(labels))
                    ]
                )
                expected_by_seed[seed] = [
                    (relative(paths[index]), labels[int(cache["train_y"][index])]) for index in support
                ]

        membership_hashes = {}
        for seed, expected in expected_by_seed.items():
            group = memberships.query("condition == @condition and seed == @seed").sort_values("position")
            if group.position.astype(int).tolist() != list(range(len(expected))):
                raise ValueError(f"membership positions differ: {condition} seed {seed}")
            observed = list(group[["path", "label"]].itertuples(index=False, name=None))
            if observed != expected:
                raise ValueError(f"membership rows differ from source: {condition} seed {seed}")
            membership_hash = hashlib.sha256(membership_bytes(expected)).hexdigest()
            if set(group.membership_sha256.astype(str)) != {membership_hash}:
                raise ValueError(f"membership hash mismatch: {condition} seed {seed}")
            metric_hashes = set(
                metric_frame.query("condition == @condition and seed == @seed").membership_sha256.astype(str)
            )
            if metric_hashes != {membership_hash}:
                raise ValueError(f"metric membership hash mismatch: {condition} seed {seed}")
            membership_hashes[seed] = membership_hash
        membership_set_hash = hashlib.sha256(
            "".join(f"{seed}:{membership_hashes[seed]}\n" for seed in sorted(membership_hashes)).encode()
        ).hexdigest()
        if membership_set_hash != metadata["membership_set_sha256"]:
            raise ValueError(f"membership set hash mismatch: {condition}")
        if set(metric_frame.query("condition == @condition").membership_set_sha256.astype(str)) != {
            membership_set_hash
        }:
            raise ValueError(f"metric membership set hash mismatch: {condition}")


def write_run(output: Path, args: argparse.Namespace) -> dict:
    image_hashes: dict[Path, str] = {}
    rows, memberships, inputs = [], [], {}
    overrides = dict(item.split("=", 1) for item in args.cache)
    if set(overrides) - set(THREE_LABEL):
        raise ValueError("unknown --cache condition")
    for condition, default_path in THREE_LABEL.items():
        condition_rows, condition_memberships, metadata = evaluate_three_label(
            condition, safe_path(overrides.get(condition, default_path)), image_hashes
        )
        rows.extend(condition_rows)
        memberships.extend(condition_memberships)
        inputs[condition] = metadata
    strict_rows, strict_memberships, strict_metadata = evaluate_strict(
        safe_path(args.strict_cache), safe_path(args.strict_episodes)
    )
    rows.extend(strict_rows)
    memberships.extend(strict_memberships)
    inputs["strict_5label_5shot"] = strict_metadata

    source_pre = Path(args.source_pre).expanduser().resolve(strict=True)
    source_pre_hash = sha256(source_pre)
    source_post_hash = sha256(safe_path(SOURCE))
    context_hash = sha256(safe_path(CONTEXT))
    parity_path = safe_path(args.automatic_parity)
    parity = load_json(parity_path)
    if parity.get("passed") is not True:
        raise ValueError("automatic-selection parity evidence did not pass")
    parity_hash = sha256(parity_path)
    scorer_records = [{"name": name, "config": ({ } if name == METHODS[0] else {
        "spectral_alpha": float(name.rsplit("_", 1)[1]), "experimental_fixed_only": True
    })} for name in METHODS]
    scorer_record_hash = json_hash(scorer_records)

    for row in rows:
        row["source_pre_sha256"] = source_pre_hash
        row["source_post_sha256"] = source_post_hash
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["condition", "seed", "evaluation_view", "method"],
        key=lambda column: column.map({name: index for index, name in enumerate(METHODS)}) if column.name == "method" else column,
    ).reset_index(drop=True)
    validate_matrix(frame)
    membership_frame = pd.DataFrame(memberships).sort_values(["condition", "seed", "position"]).reset_index(drop=True)
    if len(membership_frame) != 870:
        raise ValueError("memberships matrix must contain 870 rows")

    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "metrics.csv", index=False, float_format="%.17g", lineterminator="\n")
    membership_frame.to_csv(output / "memberships.csv", index=False, lineterminator="\n")
    provenance = {
        "schema_version": 1,
        "anchor": ".",
        "conditions": list(THREE_LABEL) + ["strict_5label_5shot"],
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "inputs": inputs,
        "source_pre_sha256": source_pre_hash,
        "source_post": relative(SOURCE),
        "source_post_sha256": source_post_hash,
        "harness": relative(Path(__file__)),
        "harness_sha256": sha256(Path(__file__)),
        "automatic_selection_parity": relative(parity_path),
        "automatic_selection_parity_sha256": parity_hash,
        "experimental_scorer_records": scorer_records,
        "experimental_scorer_records_sha256": scorer_record_hash,
        "claim": {
            "source_locator": f"{relative(CONTEXT)}:49",
            "context_sha256": context_hash,
            "captured_text": "the best angular candidate reaches 67.06% accuracy.",
            "aggregation": "operational interpretation: maximum of the two per-method six-seed mean strict accuracies",
        },
        "disclosure": "exploratory post-selection evidence; not confirmatory",
    }
    (output / "provenance.json").write_bytes(json_bytes(provenance))
    (output / "VALIDATION.md").write_text(render_report(frame), encoding="utf-8", newline="\n")
    artifacts = {name: sha256(output / name) for name in FILES[:-1]}
    manifest = {
        "schema_version": 1,
        "row_counts": {"all_rows": 150, "duplicate_free": 120, "total": 270, "memberships": 870},
        "conditions": provenance["conditions"],
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "input_hashes": {
            condition: {key: value for key, value in metadata.items() if key.endswith("sha256")}
            for condition, metadata in inputs.items()
        },
        "source_pre_sha256": source_pre_hash,
        "source_post_sha256": source_post_hash,
        "context_sha256": context_hash,
        "harness_sha256": provenance["harness_sha256"],
        "automatic_selection_parity_sha256": parity_hash,
        "experimental_scorer_records_sha256": scorer_record_hash,
        "artifacts": artifacts,
    }
    success = {**manifest, "manifest_sha256": json_hash(manifest), "status": "passed"}
    (output / "SUCCESS.json").write_bytes(json_bytes(success))
    return success


def validate_run(path: Path) -> dict:
    if not path.is_dir() or not all((path / name).is_file() for name in FILES):
        raise ValueError(f"incomplete validation output: {path}")
    success = load_json(path / "SUCCESS.json")
    if success.get("status") != "passed":
        raise ValueError(f"validation output is not successful: {path}")
    expected_counts = {"all_rows": 150, "duplicate_free": 120, "total": 270, "memberships": 870}
    if (
        success.get("row_counts") != expected_counts
        or success.get("conditions") != list(THREE_LABEL) + ["strict_5label_5shot"]
        or success.get("seeds") != list(SEEDS)
        or success.get("methods") != list(METHODS)
    ):
        raise ValueError(f"success manifest schema mismatch: {path}")
    for name, expected in success.get("artifacts", {}).items():
        if name not in FILES[:-1] or sha256(path / name) != expected:
            raise ValueError(f"artifact hash mismatch: {path / name}")
    if set(success.get("artifacts", {})) != set(FILES[:-1]):
        raise ValueError(f"success artifact list mismatch: {path}")
    metric_frame = pd.read_csv(path / "metrics.csv")
    validate_matrix(metric_frame)
    memberships = pd.read_csv(path / "memberships.csv")
    provenance = load_json(path / "provenance.json")
    if (
        provenance.get("schema_version") != 1
        or provenance.get("conditions") != success["conditions"]
        or provenance.get("seeds") != success["seeds"]
        or provenance.get("methods") != success["methods"]
        or provenance.get("source_pre_sha256") != success["source_pre_sha256"]
        or provenance.get("source_post_sha256") != success["source_post_sha256"]
    ):
        raise ValueError(f"provenance schema mismatch: {path}")
    validate_memberships(memberships, metric_frame, provenance)
    manifest = {key: value for key, value in success.items() if key not in {"manifest_sha256", "status"}}
    if json_hash(manifest) != success.get("manifest_sha256"):
        raise ValueError(f"success manifest hash mismatch: {path}")
    return success


def publish(args: argparse.Namespace) -> dict:
    final = safe_path(args.final_dir, must_exist=False)
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists() and not (final / "SUCCESS.json").is_file():
        raise ValueError(f"pre-existing final directory is incomplete: {final}")
    stages = [Path(tempfile.mkdtemp(prefix=".mahalanobis_spectrum_validation_20260816.staging-", dir=final.parent)) for _ in range(2)]
    promoted = None
    try:
        for stage in stages:
            stage.rmdir()
            write_run(stage, args)
            validate_run(stage)
        if any((stages[0] / name).read_bytes() != (stages[1] / name).read_bytes() for name in FILES):
            raise ValueError("independent validation runs are not byte-identical")
        if final.exists():
            validate_run(final)
            if any((final / name).read_bytes() != (stages[0] / name).read_bytes() for name in FILES):
                raise ValueError("pre-existing final directory differs; left untouched")
            result = {"status": "reused", "output": relative(final), **validate_run(final)}
        else:
            os.replace(stages[0], final)
            promoted = final
            result = {"status": "published", "output": relative(final), **validate_run(final)}
        return result
    finally:
        for stage in stages:
            if stage.exists() and stage != promoted:
                shutil.rmtree(stage)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--source-pre", type=Path, required=True)
    parser.add_argument("--automatic-parity", type=Path, required=True)
    parser.add_argument("--cache", action="append", default=[], metavar="CONDITION=PATH")
    parser.add_argument("--strict-cache", type=Path, default=STRICT_CACHE)
    parser.add_argument("--strict-episodes", type=Path, default=STRICT_EPISODES)
    args = parser.parse_args()
    try:
        result = publish(args)
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
