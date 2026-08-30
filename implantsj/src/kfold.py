from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd


def build_stratified_kfold_manifests(
    df: pd.DataFrame,
    out_dir: str | Path,
    base_name: str,
    k: int = 5,
    seed: int = 42,
    label_col: str = "label",
) -> Tuple[List[Path], pd.DataFrame]:
    """Write one train/valid manifest per stratified KFold split.

    The input dataframe must contain at least path,label columns. If it already has a
    split column, the original values are preserved in source_split and split is
    replaced by each fold's train/valid assignment.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2 for KFold, got {k}")
    if "path" not in df.columns or label_col not in df.columns:
        raise ValueError(f"KFold dataframe must contain path and {label_col} columns")

    work = df.copy().reset_index(drop=True)
    work[label_col] = work[label_col].astype(str)
    if "split" in work.columns:
        work["source_split"] = work["split"].astype(str)

    counts = work[label_col].value_counts().sort_index()
    too_small = counts[counts < k]
    if not too_small.empty:
        raise ValueError(
            f"Every label needs at least k={k} samples for stratified KFold. "
            f"Too small: {too_small.to_dict()}"
        )

    try:
        from sklearn.model_selection import StratifiedKFold
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("scikit-learn is required for --kfold") from exc

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    labels = work[label_col].to_numpy()
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    manifest_paths: List[Path] = []
    summary_rows: list[dict[str, object]] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(work, labels), start=1):
        fold_df = work.copy()
        fold_df["fold"] = fold_idx
        fold_df["split"] = "train"
        fold_df.loc[valid_idx, "split"] = "valid"

        path = out / f"{base_name}_fold{fold_idx:02d}.csv"
        fold_df.to_csv(path, index=False)
        manifest_paths.append(path)

        counts_table = fold_df.groupby(["split", label_col]).size().reset_index(name="count")
        for row in counts_table.to_dict("records"):
            summary_rows.append({"fold": fold_idx, **row})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / f"{base_name}_summary.csv", index=False)
    return manifest_paths, summary
