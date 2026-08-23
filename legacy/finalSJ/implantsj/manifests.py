"""Build deterministic, auditable closed/open experiment manifests."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .contracts import (
    CLOSED_LABELS,
    clean_label,
    label_key,
    read_label_list,
    validate_partition,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_aliases(path: Path | None) -> dict[str, str]:
    aliases: dict[str, str] = {}
    if path is None:
        return aliases
    frame = pd.read_csv(path)
    required = {"source_label", "canonical_label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"alias CSV must contain {sorted(required)}")
    for row in frame.itertuples(index=False):
        source = label_key(row.source_label)
        target = clean_label(row.canonical_label)
        if not source or not target:
            raise ValueError("alias source_label and canonical_label must be non-empty")
        if source in aliases and aliases[source] != target:
            raise ValueError(f"conflicting aliases for {row.source_label!r}")
        aliases[source] = target
    return aliases


def _canonicalize_labels(
    frame: pd.DataFrame,
    aliases: dict[str, str],
    closed_labels: Sequence[str] = CLOSED_LABELS,
) -> pd.DataFrame:
    closed_by_key = {label_key(label): label for label in closed_labels}

    def canonical(value: object) -> str:
        cleaned = clean_label(value)
        aliased = aliases.get(label_key(cleaned), cleaned)
        return closed_by_key.get(label_key(aliased), clean_label(aliased))

    result = frame.copy()
    result["label"] = result["label"].map(canonical)
    if (result["label"] == "").any():
        raise ValueError("empty labels are not allowed")
    collisions: dict[str, set[str]] = {}
    for label in result["label"].unique():
        collisions.setdefault(label_key(label), set()).add(label)
    bad = {key: values for key, values in collisions.items() if len(values) > 1}
    if bad:
        raise ValueError(f"case/spacing label collisions remain: {bad}")
    return result


def _resolve_manifest_paths(frame: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
    result = frame.copy()

    def resolve(value: object) -> str:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = manifest_path.parent / path
        return str(path.resolve())

    result["path"] = result["path"].map(resolve)
    return result


def read_source_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"path", "label", "split"}
    if not required.issubset(frame.columns):
        raise ValueError(f"source manifest must contain {sorted(required)}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("source manifest path,label,split values must be non-empty")
    if any(frame[column].astype(str).str.strip().eq("").any() for column in required):
        raise ValueError("source manifest path,label,split values must be non-empty")
    frame = _resolve_manifest_paths(frame, path)
    frame["split"] = frame["split"].astype(str).str.strip().str.lower()
    allowed = {"train", "valid", "val", "validation", "test"}
    invalid = sorted(set(frame["split"]) - allowed)
    if invalid:
        raise ValueError(f"unsupported source split values: {invalid}")
    frame["origin_split"] = frame["split"].replace(
        {"val": "train", "valid": "train", "validation": "train"}
    )
    return frame


def scan_imagefolder(root: Path, split: str) -> pd.DataFrame:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"image root does not exist: {root}")
    rows: list[dict[str, str]] = []
    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    for class_dir in class_dirs:
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                rows.append(
                    {
                        "path": str(path.resolve()),
                        "label": class_dir.name,
                        "split": split,
                        "origin_split": split,
                    }
                )
    if not rows:
        raise ValueError(f"no images found below {root}; expected root/<label>/<image>")
    return pd.DataFrame(rows)


def read_gt_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if not {"path", "label"}.issubset(frame.columns):
        raise ValueError("GT manifest must contain path,label")
    if frame.empty or frame[["path", "label"]].isna().any().any():
        raise ValueError("GT manifest path,label values must be non-empty")
    if any(
        frame[column].astype(str).str.strip().eq("").any()
        for column in ("path", "label")
    ):
        raise ValueError("GT manifest path,label values must be non-empty")
    frame = _resolve_manifest_paths(frame, path)
    frame["split"] = "gt"
    frame["origin_split"] = "gt"
    return frame


def _manual_stratified_split(
    frame: pd.DataFrame, valid_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    train_indices: list[int] = []
    valid_indices: list[int] = []
    for label, group in frame.groupby("label", sort=True):
        indices = list(group.index)
        rng.shuffle(indices)
        if len(indices) < 2:
            raise ValueError(
                f"label {label!r} has only {len(indices)} training sample; "
                "at least two are required to create train and validation"
            )
        n_valid = max(1, int(round(len(indices) * valid_fraction)))
        n_valid = min(n_valid, len(indices) - 1)
        valid_indices.extend(indices[:n_valid])
        train_indices.extend(indices[n_valid:])
    return frame.loc[sorted(train_indices)].copy(), frame.loc[
        sorted(valid_indices)
    ].copy()


def _group_split(
    frame: pd.DataFrame, group_column: str, valid_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Try deterministic group splits and keep the best label-balanced valid set."""
    try:
        from sklearn.model_selection import GroupShuffleSplit
    except ImportError as exc:  # pragma: no cover - dependency is documented
        raise RuntimeError(
            "scikit-learn is required for group-aware splitting"
        ) from exc

    labels = set(frame["label"])
    best: tuple[float, pd.DataFrame, pd.DataFrame] | None = None
    for offset in range(128):
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=valid_fraction, random_state=seed + offset
        )
        train_pos, valid_pos = next(splitter.split(frame, groups=frame[group_column]))
        train = frame.iloc[train_pos]
        valid = frame.iloc[valid_pos]
        if set(train["label"]) != labels or set(valid["label"]) != labels:
            continue
        total_counts = frame["label"].value_counts(normalize=True)
        valid_counts = valid["label"].value_counts(normalize=True)
        imbalance = float((total_counts - valid_counts).abs().sum())
        size_error = abs(len(valid) / len(frame) - valid_fraction)
        score = imbalance + size_error
        if best is None or score < best[0]:
            best = (score, train.copy(), valid.copy())
    if best is None:
        raise ValueError(
            f"could not form a group-disjoint split with every label in both sets using {group_column!r}; "
            "add more groups per class or omit --group-column"
        )
    return best[1], best[2]


def split_train_valid(
    train_pool: pd.DataFrame,
    valid_fraction: float,
    seed: int,
    group_column: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("valid_fraction must be between 0 and 1")
    if group_column and group_column in train_pool.columns:
        groups = train_pool[group_column].fillna("").astype(str).str.strip()
        if groups.ne("").all() and groups.nunique() > 1:
            train_pool = train_pool.copy()
            train_pool[group_column] = groups
            train, valid = _group_split(train_pool, group_column, valid_fraction, seed)
            return train, valid, f"group:{group_column}"
    train, valid = _manual_stratified_split(train_pool, valid_fraction, seed)
    return train, valid, "label_stratified"


def _counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(key): int(value) for key, value in sorted(Counter(frame["label"]).items())
    }


def _validate_label_contract(
    labels: Sequence[str],
    allow_incomplete: bool,
    closed_labels: Sequence[str] = CLOSED_LABELS,
) -> tuple[list[str], list[str]]:
    unique = sorted(set(labels), key=label_key)
    by_key = {label_key(label): label for label in unique}
    missing_closed = [
        label for label in closed_labels if label_key(label) not in by_key
    ]
    if missing_closed and not allow_incomplete:
        raise ValueError(f"data is missing closed labels: {missing_closed}")
    closed = [label for label in closed_labels if label_key(label) in by_key]
    closed_keys = {label_key(label) for label in closed_labels}
    opened = [label for label in unique if label_key(label) not in closed_keys]
    if (
        tuple(closed_labels) == CLOSED_LABELS
        and not allow_incomplete
        and (len(unique) != 42 or len(closed) != 27 or len(opened) != 15)
    ):
        raise ValueError(
            f"expected 42 total labels = 27 closed + 15 open, got "
            f"{len(unique)} = {len(closed)} + {len(opened)}; open labels={opened}"
        )
    validate_partition(
        unique,
        closed,
        opened,
        strict=tuple(closed_labels) == CLOSED_LABELS and not allow_incomplete,
    )
    return closed, opened


def _ensure_split_coverage(
    frame: pd.DataFrame, expected: Iterable[str], name: str
) -> None:
    missing = sorted(set(expected) - set(frame["label"]), key=label_key)
    if missing:
        raise ValueError(f"{name} is missing labels: {missing}")


def _ordered_columns(frame: pd.DataFrame) -> list[str]:
    preferred = ["path", "label", "split", "set", "origin_split"]
    return [column for column in preferred if column in frame.columns] + [
        column for column in frame.columns if column not in preferred
    ]


def build_manifests(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    aliases = read_aliases(
        Path(args.aliases).expanduser().resolve() if args.aliases else None
    )
    closed_labels = read_label_list(
        Path(args.closed_labels).expanduser().resolve()
        if getattr(args, "closed_labels", None)
        else None
    )

    if args.source_manifest:
        if args.train_root or args.test_root:
            raise ValueError(
                "use either --source-manifest or --train-root/--test-root, not both"
            )
        source = read_source_manifest(Path(args.source_manifest).expanduser().resolve())
    else:
        if not args.train_root or not args.test_root:
            raise ValueError(
                "provide --source-manifest or both --train-root and --test-root"
            )
        source = pd.concat(
            [
                scan_imagefolder(Path(args.train_root), "train"),
                scan_imagefolder(Path(args.test_root), "test"),
            ],
            ignore_index=True,
        )

    alias_applied_count = sum(
        label_key(value) in aliases and clean_label(value) != aliases[label_key(value)]
        for value in source["label"]
    )
    source = _canonicalize_labels(source, aliases, closed_labels)
    duplicated_paths = source[source["path"].duplicated(keep=False)][
        ["path", "label", "split"]
    ]
    if not duplicated_paths.empty:
        raise ValueError(
            "source manifest contains duplicate image paths; each image must occur once to prevent split leakage: "
            f"{duplicated_paths.head(10).to_dict('records')}"
        )
    closed, opened = _validate_label_contract(
        source["label"].tolist(), args.allow_incomplete, closed_labels
    )
    all_labels = closed + opened
    set_by_key = {label_key(label): "closed" for label in closed}
    set_by_key.update({label_key(label): "open" for label in opened})
    source["set"] = source["label"].map(lambda value: set_by_key[label_key(value)])

    train_pool = source[source["origin_split"] == "train"].copy()
    test = source[source["origin_split"] == "test"].copy()
    if train_pool.empty or test.empty:
        raise ValueError(
            "source data must contain both original train and original test rows"
        )
    if not args.allow_incomplete:
        _ensure_split_coverage(train_pool, all_labels, "original train")
        _ensure_split_coverage(test, all_labels, "original test")
    train, valid, split_method = split_train_valid(
        train_pool, args.valid_fraction, args.seed, args.group_column
    )
    train["split"] = "train"
    valid["split"] = "valid"
    test["split"] = "test"

    original_test_paths = set(source.loc[source["origin_split"] == "test", "path"])
    if set(test["path"]) != original_test_paths:
        raise AssertionError("internal error: original test membership changed")
    if original_test_paths & (set(train["path"]) | set(valid["path"])):
        raise AssertionError("test leakage detected in train/validation")

    if not args.allow_incomplete:
        for name, frame in (("train", train), ("valid", valid), ("test", test)):
            _ensure_split_coverage(frame, all_labels, name)

    full = pd.concat([train, valid, test], ignore_index=True)
    support = train.copy()
    outputs: dict[str, Path] = {}
    for name, frame in (
        ("train", train),
        ("valid", valid),
        ("test", test),
        ("support", support),
        ("full", full),
    ):
        path = output_dir / f"{name}.csv"
        frame.loc[:, _ordered_columns(frame)].sort_values(["label", "path"]).to_csv(
            path, index=False
        )
        outputs[name] = path

    gt: pd.DataFrame | None = None
    gt_data_overlap: list[str] = []
    if args.gt_manifest and args.gt_root:
        raise ValueError("use either --gt-manifest or --gt-root")
    if args.gt_manifest:
        gt = read_gt_manifest(Path(args.gt_manifest).expanduser().resolve())
    elif args.gt_root:
        gt = scan_imagefolder(Path(args.gt_root), "gt")
    if gt is not None:
        gt = _canonicalize_labels(gt, aliases, closed_labels)
        if gt["path"].duplicated().any():
            raise ValueError("GT manifest contains duplicate image paths")
        unknown_gt = sorted(set(gt["label"]) - set(all_labels), key=label_key)
        if unknown_gt:
            raise ValueError(
                f"GT manifest contains labels not present in the gallery data: {unknown_gt}"
            )
        gt["set"] = gt["label"].map(lambda value: set_by_key[label_key(value)])
        if not args.allow_incomplete:
            _ensure_split_coverage(gt, all_labels, "GT prototype")
        insufficient = gt["label"].value_counts()
        insufficient = insufficient[insufficient < 2].to_dict()
        if insufficient:
            raise ValueError(
                "GT prototype requires at least two independent images per label: "
                f"{insufficient}"
            )
        gt_data_overlap = sorted(set(gt["path"]) & set(full["path"]))
        if gt_data_overlap:
            raise ValueError(
                "GT prototype images must be independent from train/valid/test data: "
                f"{gt_data_overlap[:10]}"
            )
        gt_path = output_dir / "gt.csv"
        gt.loc[:, _ordered_columns(gt)].sort_values(["label", "path"]).to_csv(
            gt_path, index=False
        )
        outputs["gt"] = gt_path

    labels_payload = {
        "all": all_labels,
        "closed": closed,
        "open": opened,
        "contract": {
            "total": len(all_labels),
            "closed": len(closed),
            "open": len(opened),
        },
    }
    labels_path = output_dir / "labels.json"
    labels_path.write_text(
        json.dumps(labels_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    outputs["labels"] = labels_path

    audit = {
        "seed": args.seed,
        "valid_fraction": args.valid_fraction,
        "split_method": split_method,
        "group_column_used": (
            args.group_column if split_method.startswith("group:") else None
        ),
        "alias_applied_count": int(alias_applied_count),
        "strict_42_class_contract": (
            tuple(closed_labels) == CLOSED_LABELS and not args.allow_incomplete
        ),
        "label_counts": {
            "total": len(all_labels),
            "closed": len(closed),
            "open": len(opened),
        },
        "sample_counts": {
            name: len(frame)
            for name, frame in (("train", train), ("valid", valid), ("test", test))
        },
        "per_class_counts": {
            name: _counts(frame)
            for name, frame in (("train", train), ("valid", valid), ("test", test))
        },
        "original_test_preserved": True,
        "test_leakage_count": 0,
        "gt_available": gt is not None,
        "gt_data_path_overlap_count": len(gt_data_overlap),
    }
    audit_path = output_dir / "audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    outputs["audit"] = audit_path

    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_argument_group("data source")
    source.add_argument("--source-manifest", help="CSV with path,label,split")
    source.add_argument(
        "--train-root", help="ImageFolder root for original training data"
    )
    source.add_argument(
        "--test-root", help="ImageFolder root for immutable original test data"
    )
    source.add_argument(
        "--gt-manifest", help="optional GT prototype CSV with path,label"
    )
    source.add_argument("--gt-root", help="optional GT ImageFolder root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--aliases", help="optional CSV: source_label,canonical_label")
    parser.add_argument(
        "--closed-labels",
        help="optional newline-delimited closed-set labels; defaults to the canonical 27",
    )
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--group-column", default="case_id")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="development only: relax exact 42-label and per-split coverage checks",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = build_manifests(args)
    print(
        json.dumps(
            {key: str(value) for key, value in outputs.items()},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
