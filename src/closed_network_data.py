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


def load_label_from_json(json_path: str | Path, label_keys: Sequence[str] = ("모델명",)) -> str:
    path = Path(json_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    value = _find_first_key(data, tuple(label_keys))
    label = "" if value is None else str(value).strip()
    if not label:
        raise RuntimeError(f"Could not find label key {list(label_keys)} in {path}")
    return label


def scan_split_pairs(
    image_root: str | Path,
    json_root: str | Path,
    split_name: str,
    label_keys: Sequence[str] = ("모델명",),
) -> pd.DataFrame:
    image_root = Path(image_root)
    json_root = Path(json_root)
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing image root: {image_root}")
    if not json_root.is_dir():
        raise FileNotFoundError(f"Missing json root: {json_root}")

    rows: List[Dict[str, str]] = []
    missing_json: List[str] = []
    for image_path in sorted(image_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = image_path.relative_to(image_root)
        json_path = (json_root / rel).with_suffix(".json")
        if not json_path.is_file():
            missing_json.append(str(rel))
            continue
        label = load_label_from_json(json_path, label_keys=label_keys)
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
                "relative_path": str(rel),
                "json_path": str(json_path.resolve()),
            }
        )
    if missing_json:
        sample = missing_json[:10]
        raise RuntimeError(
            f"Missing matching JSON files under {json_root} for {len(missing_json)} images. Sample: {sample}"
        )
    if not rows:
        raise RuntimeError(f"No image/json pairs found under image_root={image_root}, json_root={json_root}")
    return pd.DataFrame(rows)


def build_closed_network_dataframe(
    train_image_root: str | Path,
    train_json_root: str | Path,
    test_image_root: str | Path,
    test_json_root: str | Path,
    label_keys: Sequence[str] = ("모델명",),
) -> pd.DataFrame:
    train_df = scan_split_pairs(train_image_root, train_json_root, split_name="train", label_keys=label_keys)
    test_df = scan_split_pairs(test_image_root, test_json_root, split_name="test", label_keys=label_keys)
    df = pd.concat([train_df, test_df], ignore_index=True)
    df["label"] = df["label"].astype(str)
    return df.reset_index(drop=True)


def split_train_valid(df: pd.DataFrame, valid_ratio: float = 0.1, seed: int = 42) -> pd.DataFrame:
    if not 0.0 <= valid_ratio < 1.0:
        raise ValueError(f"valid_ratio must be in [0,1). Got {valid_ratio}")

    work = df.copy().reset_index(drop=True)
    train_mask = work["origin_split"].astype(str) == "train"
    if valid_ratio <= 0.0:
        work.loc[train_mask, "split"] = "train"
        work.loc[~train_mask, "split"] = "test"
        return work

    train_df = work[train_mask].copy()
    test_df = work[~train_mask].copy()
    train_df["split"] = "train"
    rng = random.Random(seed)

    valid_indices: List[int] = []
    for _, group in train_df.groupby("label", sort=True):
        indices = group.index.tolist()
        if len(indices) <= 1:
            continue
        rng.shuffle(indices)
        n_valid = max(1, int(round(len(indices) * valid_ratio)))
        n_valid = min(n_valid, len(indices) - 1)
        valid_indices.extend(indices[:n_valid])

    train_df.loc[valid_indices, "split"] = "valid"
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
    seen_label_count: int = 27,
    unseen_label_count: int = 15,
) -> Tuple[List[str], List[str], pd.DataFrame]:
    train_side = df[df["origin_split"].astype(str) == "train"].copy()
    counts = train_side["label"].astype(str).value_counts()
    labels = sorted(counts.index.tolist())
    total = len(labels)
    if seen_label_count + unseen_label_count != total:
        raise RuntimeError(
            f"seen_label_count + unseen_label_count must equal total labels. "
            f"got seen={seen_label_count}, unseen={unseen_label_count}, total={total}"
        )

    stats = pd.DataFrame(
        [{"label": str(label), "train_count": int(counts.get(label, 0))} for label in labels]
    ).sort_values(["train_count", "label"], ascending=[True, True], ignore_index=True)

    unseen = stats.head(unseen_label_count)["label"].astype(str).tolist()
    seen = sorted(set(labels) - set(unseen))
    if len(seen) != seen_label_count:
        raise RuntimeError(f"Expected {seen_label_count} seen labels, got {len(seen)}")
    return seen, unseen, stats


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
    cols = ["path", "label", "split", "origin_split", "hospital", "case_id", "relative_path", "json_path"]
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
    link_mode: str = "symlink",
) -> Path:
    out_root = Path(out_root)
    split_name_map = split_name_map or {"train": "train", "valid": "val", "test": "test"}
    if link_mode not in {"symlink", "copy"}:
        raise ValueError(f"Unsupported link_mode: {link_mode}")

    for row in df.itertuples(index=False):
        label = str(row.label)
        split = str(row.split)
        if split not in split_name_map:
            continue
        class_dir = class_folder_map[label]
        rel = Path(str(row.relative_path))
        dest = out_root / split_name_map[split] / class_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = Path(str(row.path))
        if dest.exists():
            continue
        if link_mode == "copy":
            shutil.copy2(src, dest)
        else:
            dest.symlink_to(src)
    return out_root

