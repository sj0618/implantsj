from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .data import IMAGE_EXTENSIONS


def _find_first_key(obj: Any, target_keys: Sequence[str]) -> Optional[Any]:
    if isinstance(obj, dict):
        for key in target_keys:
            if key in obj:
                return obj[key]
        for value in obj.values():
            found = _find_first_key(value, target_keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_first_key(item, target_keys)
            if found is not None:
                return found
    return None


def _read_json_text(path: Path) -> str:
    """Read a label JSON tolerating the encodings Korean annotation tools emit.

    Closed-network label files are frequently UTF-8 with a BOM or CP949 rather
    than plain UTF-8, and json.loads() rejects a leading BOM outright.
    """
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def normalize_label(text: str) -> str:
    """Collapse whitespace so 'TS  III' and 'TS III' do not become two classes."""
    return " ".join(str(text).split()).strip()


def load_label_from_json(json_path: str | Path, label_keys: Sequence[str] = ("모델명",)) -> str:
    path = Path(json_path)
    data = json.loads(_read_json_text(path))
    value = _find_first_key(data, tuple(label_keys))
    label = "" if value is None else normalize_label(value)
    if not label:
        raise RuntimeError(f"Could not find label key {list(label_keys)} in {path}")
    return label


def scan_split_pairs(
    image_root: str | Path,
    json_root: str | Path,
    split_name: str,
    label_keys: Sequence[str] = ("모델명",),
    allow_missing: bool = False,
    progress_every: int = 2000,
    report: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    """Pair every image under image_root with its JSON label under json_root.

    With allow_missing=True, images whose JSON is absent or unreadable are
    skipped and recorded in `report` instead of aborting the whole scan; a
    single stray file should not cost a multi-hour dataset build.
    """
    image_root = Path(image_root)
    json_root = Path(json_root)
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing image root: {image_root}")
    if not json_root.is_dir():
        raise FileNotFoundError(f"Missing json root: {json_root}")

    rows: List[Dict[str, str]] = []
    missing_json: List[str] = []
    unreadable_json: List[str] = []
    scanned = 0
    print(f"[scan:{split_name}] images={image_root}", flush=True)
    print(f"[scan:{split_name}] labels={json_root}", flush=True)
    for image_path in sorted(image_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        scanned += 1
        if progress_every and scanned % progress_every == 0:
            print(f"[scan:{split_name}] {scanned} images, {len(rows)} paired", flush=True)
        rel = image_path.relative_to(image_root)
        json_path = (json_root / rel).with_suffix(".json")
        if not json_path.is_file():
            missing_json.append(str(rel))
            continue
        try:
            label = load_label_from_json(json_path, label_keys=label_keys)
        except Exception as exc:  # unparseable JSON or absent label key
            unreadable_json.append(f"{rel}: {exc}")
            continue
        rel_parts = rel.parts
        hospital = rel_parts[0] if len(rel_parts) >= 1 else ""
        case_id = rel_parts[1] if len(rel_parts) >= 2 else ""
        rows.append(
            {
                "path": str(image_path.resolve()),
                "label": label,
                "split": split_name,
                "origin_split": split_name,
                "hospital": hospital,
                "case_id": case_id,
                "group_id": f"{hospital}/{case_id}",
                "relative_path": str(rel),
                "json_path": str(json_path.resolve()),
            }
        )

    skipped = missing_json + unreadable_json
    if report is not None:
        report.setdefault("missing_json", []).extend(missing_json)
        report.setdefault("unreadable_json", []).extend(unreadable_json)
    if skipped and not allow_missing:
        raise RuntimeError(
            f"{len(skipped)} images under {image_root} have no usable JSON label "
            f"(missing={len(missing_json)}, unreadable={len(unreadable_json)}). "
            f"Sample: {skipped[:10]}. "
            f"Pass --allow-missing-json to skip them instead."
        )
    if skipped:
        print(
            f"[scan:{split_name}] WARNING skipped {len(skipped)} images "
            f"(missing={len(missing_json)}, unreadable={len(unreadable_json)})",
            flush=True,
        )
    if not rows:
        raise RuntimeError(f"No image/json pairs found under image_root={image_root}, json_root={json_root}")
    print(f"[scan:{split_name}] done: {len(rows)} paired from {scanned} images", flush=True)
    return pd.DataFrame(rows)


def build_closed_network_dataframe(
    train_image_root: str | Path,
    train_json_root: str | Path,
    test_image_root: str | Path,
    test_json_root: str | Path,
    label_keys: Sequence[str] = ("모델명",),
    allow_missing: bool = False,
    report: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    train_df = scan_split_pairs(
        train_image_root, train_json_root, split_name="train",
        label_keys=label_keys, allow_missing=allow_missing, report=report,
    )
    test_df = scan_split_pairs(
        test_image_root, test_json_root, split_name="test",
        label_keys=label_keys, allow_missing=allow_missing, report=report,
    )
    df = pd.concat([train_df, test_df], ignore_index=True)
    df["label"] = df["label"].astype(str)
    return df.reset_index(drop=True)


def _validated_group_ids(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.Series:
    missing = [column for column in group_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"group-aware split requires columns {list(group_columns)}; missing={missing}"
        )
    values = []
    for column in group_columns:
        part = frame[column].fillna("").astype(str).str.strip()
        if part.eq("").any():
            sample = frame.loc[part.eq(""), "path"].astype(str).head(5).tolist()
            raise ValueError(
                f"group column {column!r} contains blank values; sample paths={sample}"
            )
        values.append(part)
    group_ids = values[0]
    for value in values[1:]:
        group_ids = group_ids.str.cat(value, sep="/")
    return group_ids


def split_train_valid(
    df: pd.DataFrame,
    valid_ratio: float = 0.1,
    seed: int = 42,
    group_columns: Sequence[str] = ("hospital", "case_id"),
    required_valid_labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Split original training data without allowing a case into both splits.

    Validation groups are sampled independently per label. Labels with only one
    source group stay in training unless they are listed in
    ``required_valid_labels``, in which case the build fails with an actionable
    error instead of silently reporting a validation score for an absent class.
    """
    if not 0.0 <= valid_ratio < 1.0:
        raise ValueError(f"valid_ratio must be in [0,1). Got {valid_ratio}")

    work = df.copy().reset_index(drop=True)
    work["group_id"] = _validated_group_ids(work, group_columns)
    train_mask = work["origin_split"].astype(str) == "train"
    source_train_groups = set(work.loc[train_mask, "group_id"])
    source_test_groups = set(work.loc[~train_mask, "group_id"])
    source_overlap = sorted(source_train_groups & source_test_groups)
    if source_overlap:
        raise ValueError(
            "original train/test data reuse case groups; "
            f"overlap sample={source_overlap[:10]}"
        )
    if valid_ratio <= 0.0:
        if required_valid_labels:
            raise ValueError("required validation labels cannot be satisfied when valid_ratio is 0")
        work.loc[train_mask, "split"] = "train"
        work.loc[~train_mask, "split"] = "test"
        return work

    train_df = work[train_mask].copy()
    test_df = work[~train_mask].copy()
    train_df["split"] = "train"
    group_label_counts = train_df.groupby("group_id")["label"].nunique()
    ambiguous_groups = group_label_counts[group_label_counts > 1].index.astype(str).tolist()
    if ambiguous_groups:
        raise ValueError(
            "a case group contains multiple labels; "
            f"ambiguous groups={ambiguous_groups[:10]}"
        )
    rng = random.Random(seed)
    required = (
        set(train_df["label"].astype(str))
        if required_valid_labels is None
        else {str(label) for label in required_valid_labels}
    )
    unknown_required = sorted(required - set(train_df["label"].astype(str)))
    if unknown_required:
        raise ValueError(f"required validation labels are absent from train data: {unknown_required}")

    valid_groups: set[str] = set()
    for label, group in train_df.groupby("label", sort=True):
        grouped = group.groupby("group_id", sort=True).size()
        group_ids = grouped.index.astype(str).tolist()
        if len(group_ids) <= 1:
            if str(label) in required:
                raise ValueError(
                    f"label {label!r} has only {len(group_ids)} case group; "
                    "at least two disjoint groups are required for train/valid"
                )
            continue
        rng.shuffle(group_ids)
        n_valid = max(1, int(round(len(group_ids) * valid_ratio)))
        n_valid = min(n_valid, len(group_ids) - 1)
        valid_groups.update(group_ids[:n_valid])

    train_df.loc[train_df["group_id"].isin(valid_groups), "split"] = "valid"
    train_groups = set(train_df.loc[train_df["split"] == "train", "group_id"])
    validation_groups = set(train_df.loc[train_df["split"] == "valid", "group_id"])
    overlap = sorted(train_groups & validation_groups)
    if overlap:
        raise AssertionError(f"group-aware split produced overlap: {overlap[:10]}")
    valid_labels = set(train_df.loc[train_df["split"] == "valid", "label"].astype(str))
    missing_required = sorted(required - valid_labels)
    if missing_required:
        raise ValueError(f"validation split is missing required labels: {missing_required}")
    out = pd.concat([train_df, test_df], ignore_index=True)
    out["label"] = out["label"].astype(str)
    return out.reset_index(drop=True)


def label_counts(df: pd.DataFrame, split: Optional[str] = None) -> pd.Series:
    work = df
    if split is not None:
        work = work[work["split"].astype(str) == split]
    return work["label"].astype(str).value_counts().sort_index()


def infer_seen_unseen_labels(
    df: pd.DataFrame,
    seen_label_count: Optional[int] = None,
    unseen_label_count: Optional[int] = 15,
    min_train_count: Optional[int] = None,
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """Split labels into seen (many train images) and unseen (fewest) groups.

    The closed-network label set is not guaranteed to be exactly 42, so the
    counts are derived from the data instead of asserted. Give either
    `unseen_label_count` (take the N rarest labels as unseen) or
    `min_train_count` (any label below the threshold becomes unseen). If both
    seen and unseen counts are given they must add up to the labels actually
    present, and a mismatch is reported with the real total.
    """
    train_side = df[df["origin_split"].astype(str) == "train"].copy()
    counts = train_side["label"].astype(str).value_counts()
    labels = sorted(counts.index.tolist())
    total = len(labels)

    stats = pd.DataFrame(
        [{"label": str(label), "train_count": int(counts.get(label, 0))} for label in labels]
    ).sort_values(["train_count", "label"], ascending=[True, True], ignore_index=True)

    if min_train_count is not None:
        unseen = stats[stats["train_count"] < min_train_count]["label"].astype(str).tolist()
    else:
        if seen_label_count is not None and unseen_label_count is not None:
            if seen_label_count + unseen_label_count != total:
                raise RuntimeError(
                    f"seen({seen_label_count}) + unseen({unseen_label_count}) = "
                    f"{seen_label_count + unseen_label_count}, but the data has {total} labels. "
                    f"Pass only --unseen-label-count (or --min-train-count) and let seen be the rest."
                )
        elif seen_label_count is not None:
            unseen_label_count = total - seen_label_count
        if unseen_label_count is None:
            unseen_label_count = 0
        unseen_label_count = max(0, min(int(unseen_label_count), total))
        unseen = stats.head(unseen_label_count)["label"].astype(str).tolist()

    seen = sorted(set(labels) - set(unseen))
    if not seen:
        raise RuntimeError(
            f"Every one of the {total} labels was classified as unseen; nothing left to train on. "
            f"Lower --unseen-label-count or --min-train-count."
        )
    print(
        f"[labels] total={total} seen={len(seen)} unseen={len(unseen)}",
        flush=True,
    )
    return seen, sorted(unseen), stats


def build_seen27_manifest(df: pd.DataFrame, seen_labels: Sequence[str]) -> pd.DataFrame:
    seen_set = {str(x) for x in seen_labels}
    out = df[df["label"].astype(str).isin(seen_set)].copy().reset_index(drop=True)
    return out


def build_unseen15_manifest(df: pd.DataFrame, unseen_labels: Sequence[str]) -> pd.DataFrame:
    unseen_set = {str(x) for x in unseen_labels}
    out = df[df["label"].astype(str).isin(unseen_set)].copy().reset_index(drop=True)
    out["split"] = "test"
    return out


def sanitize_folder_name(name: str) -> str:
    text = str(name).strip()
    for bad in [os.sep, "/", "\\", "\0"]:
        text = text.replace(bad, "_")
    return text if text else "unknown"


def build_class_folder_map(labels: Sequence[str], indexed: bool = False) -> Dict[str, str]:
    labels = sorted(str(x) for x in labels)
    if not indexed:
        return {label: sanitize_folder_name(label) for label in labels}
    width = max(1, len(str(len(labels))))
    return {
        label: f"{idx:0{width}d}.{sanitize_folder_name(label)}"
        for idx, label in enumerate(labels, start=1)
    }


def write_manifest(df: pd.DataFrame, out_csv: str | Path) -> Path:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["path", "label", "split", "origin_split", "hospital", "case_id", "group_id", "relative_path", "json_path"]
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            df[col] = ""
    df[cols].to_csv(out_csv, index=False)
    return out_csv


def write_label_list(labels: Sequence[str], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(str(x) for x in labels) + "\n", encoding="utf-8")
    return out_path


def materialize_imagefolder(
    df: pd.DataFrame,
    out_root: str | Path,
    class_folder_map: Dict[str, str],
    split_name_map: Optional[Dict[str, str]] = None,
    link_mode: str = "hardlink",
) -> Path:
    """Lay images out as <out_root>/<split>/<class>/... for ImageFolder readers.

    link_mode choices, in order of robustness inside a container:
      hardlink - no extra disk, survives a changed mount path, same filesystem only
      copy     - always works, costs a full second copy of the dataset
      symlink  - cheapest, but the links are relative and break if the image tree
                 and this output tree stop being reachable from one another

    Absolute symlinks were the previous default and silently broke whenever the
    dataset was mounted at a different path than when it was materialized; the
    resulting dangling links are then skipped by the image scanner, so training
    quietly ran on a fraction of the data. hardlink avoids that failure mode.
    """
    out_root = Path(out_root)
    split_name_map = split_name_map or {"train": "train", "valid": "val", "test": "test"}
    if link_mode not in {"symlink", "copy", "hardlink"}:
        raise ValueError(f"Unsupported link_mode: {link_mode}")

    linked = 0
    skipped_existing = 0
    missing_src: List[str] = []
    for row in df.itertuples(index=False):
        label = str(row.label)
        split = str(row.split)
        if split not in split_name_map:
            continue
        class_dir = class_folder_map[label]
        rel = Path(str(row.relative_path))
        dest = out_root / split_name_map[split] / class_dir / rel
        src = Path(str(row.path))
        if not src.is_file():
            missing_src.append(str(src))
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        # exists() follows symlinks, so a dangling link reads as absent while
        # still occupying the name; lstat-based checks catch that case.
        if dest.is_symlink() or dest.exists():
            if dest.is_symlink() and not dest.exists():
                dest.unlink()  # replace a link left dangling by an earlier run
            else:
                skipped_existing += 1
                continue
        if link_mode == "copy":
            shutil.copy2(src, dest)
        elif link_mode == "hardlink":
            try:
                os.link(src, dest)
            except OSError:
                shutil.copy2(src, dest)  # cross-device or unsupported: fall back
        else:
            os.symlink(os.path.relpath(src, dest.parent), dest)
        linked += 1

    if missing_src:
        print(
            f"[materialize] WARNING {len(missing_src)} source images were missing. "
            f"Sample: {missing_src[:5]}",
            flush=True,
        )
    print(
        f"[materialize] {out_root}: {linked} new, {skipped_existing} already present ({link_mode})",
        flush=True,
    )
    return out_root

