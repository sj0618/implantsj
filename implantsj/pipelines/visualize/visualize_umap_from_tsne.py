#!/usr/bin/env python3
"""Generate UMAP visualizations beside existing t-SNE artifacts.

This script scans model output folders for `embeddings/tsne_*.npz` files,
loads the stored embedding features and metadata, computes a 2D UMAP
projection, and saves the result next to the existing t-SNE files as:

* `umap_<suffix>.png`
* `umap_<suffix>.csv`
* `umap_<suffix>.npz`

It is designed to be run once over a collection of model output folders and
then re-run incrementally when new t-SNE artifacts appear.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

try:
    import umap
except ImportError as exc:  # pragma: no cover - dependency check
    raise SystemExit(
        "The 'umap-learn' package is required. Install it with: pip install umap-learn"
    ) from exc


def detect_class_names(df: pd.DataFrame) -> list[str]:
    if "target" in df.columns:
        ordered = (
            df[["target", "label"]]
            .dropna()
            .drop_duplicates(subset=["target"])
            .sort_values("target")
        )
        names = ordered["label"].astype(str).tolist()
        if names:
            return names
    if "label" in df.columns:
        return sorted(df["label"].astype(str).unique().tolist())
    return []


def infer_split_values(df: pd.DataFrame) -> list[str]:
    if "split" not in df.columns:
        return ["all"]
    splits = df["split"].astype(str).dropna().unique().tolist()
    return sorted(splits) if splits else ["all"]


def compute_umap(features: np.ndarray, seed: int, pca_dim: int, n_neighbors: int, min_dist: float) -> np.ndarray:
    if features.shape[0] < 2:
        raise ValueError("UMAP needs at least 2 samples.")

    x = features
    max_pca = min(pca_dim, x.shape[1], max(1, x.shape[0] - 1))
    if pca_dim > 0 and x.shape[1] > max_pca:
        x = PCA(n_components=max_pca, random_state=seed).fit_transform(x)

    if x.shape[0] < 4:
        # UMAP becomes unstable with tiny sample counts; keep the pipeline robust.
        fallback = PCA(n_components=min(2, x.shape[1], x.shape[0]), random_state=seed).fit_transform(x)
        if fallback.shape[1] == 1:
            fallback = np.concatenate([fallback, np.zeros((fallback.shape[0], 1), dtype=fallback.dtype)], axis=1)
        return fallback

    safe_neighbors = max(2, min(int(n_neighbors), x.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=safe_neighbors,
        min_dist=float(min_dist),
        metric="euclidean",
        random_state=seed,
    )
    return reducer.fit_transform(x)


def save_plot(df: pd.DataFrame, class_names: list[str], out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    split_values = infer_split_values(df)

    for cls_idx, cls_name in enumerate(class_names):
        cls_df = df[df["target"] == cls_idx] if "target" in df.columns else df[df["label"] == cls_name]
        if cls_df.empty:
            continue
        for split_idx, split_name in enumerate(split_values):
            if "split" in cls_df.columns and split_name != "all":
                part = cls_df[cls_df["split"].astype(str) == split_name]
            else:
                part = cls_df
            if part.empty:
                continue
            ax.scatter(
                part["umap_x"],
                part["umap_y"],
                s=22,
                alpha=0.82,
                c=[cmap(cls_idx % 10)],
                marker=markers[split_idx % len(markers)],
                linewidths=0.25,
                edgecolors="white",
                label=cls_name if split_name == "all" else f"{cls_name} / {split_name}",
            )

    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.22)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def process_artifact(tsne_npz: Path, seed: int, pca_dim: int, n_neighbors: int, min_dist: float, overwrite: bool) -> Path | None:
    suffix = tsne_npz.stem.replace("tsne_", "", 1)
    out_dir = tsne_npz.parent
    out_png = out_dir / f"umap_{suffix}.png"
    out_csv = out_dir / f"umap_{suffix}.csv"
    out_npz = out_dir / f"umap_{suffix}.npz"

    if out_png.exists() and not overwrite:
        return out_png

    tsne_csv = tsne_npz.with_suffix(".csv")
    if not tsne_csv.exists():
        raise FileNotFoundError(f"Missing metadata CSV for {tsne_npz}")

    with np.load(tsne_npz, allow_pickle=True) as payload:
        if "features" not in payload:
            raise KeyError(f"'features' not found in {tsne_npz}")
        features = np.asarray(payload["features"])
    coords = compute_umap(features, seed=seed, pca_dim=pca_dim, n_neighbors=n_neighbors, min_dist=min_dist)

    rows = pd.read_csv(tsne_csv)
    if len(rows) != len(coords):
        raise ValueError(f"Row count mismatch for {tsne_npz}: csv={len(rows)} coords={len(coords)}")

    rows = rows.copy()
    rows["umap_x"] = coords[:, 0]
    rows["umap_y"] = coords[:, 1]

    class_names = detect_class_names(rows)
    if not class_names:
        class_names = [str(v) for v in sorted(rows["target"].dropna().unique().tolist())] if "target" in rows.columns else []

    rows.to_csv(out_csv, index=False)
    np.savez_compressed(
        out_npz,
        features=features,
        targets=np.asarray(rows["target"], dtype=np.int64) if "target" in rows.columns else np.array([], dtype=np.int64),
        coords=coords,
        paths=np.asarray(rows["path"], dtype=object) if "path" in rows.columns else np.array([], dtype=object),
    )
    save_plot(rows, class_names, out_png, f"UMAP embeddings ({suffix}, {len(class_names)} labels)")
    return out_png


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate UMAP plots beside existing t-SNE artifacts.")
    parser.add_argument("--outputs-root", default="outputs", help="Root folder to scan recursively for embeddings/tsne_*.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root).resolve()
    if not outputs_root.exists():
        raise FileNotFoundError(f"Outputs root does not exist: {outputs_root}")

    artifacts = sorted(outputs_root.rglob("embeddings/tsne_*.npz"))
    if not artifacts:
        print(f"No t-SNE artifacts found under {outputs_root}")
        return

    written: list[Path] = []
    for tsne_npz in artifacts:
        out_png = process_artifact(
            tsne_npz=tsne_npz,
            seed=args.seed,
            pca_dim=args.pca_dim,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            overwrite=args.overwrite,
        )
        if out_png is not None:
            written.append(out_png)
            print(out_png)

    print(f"Generated/updated {len(written)} UMAP plots under {outputs_root}")


if __name__ == "__main__":
    main()
