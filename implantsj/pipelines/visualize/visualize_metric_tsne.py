#!/usr/bin/env python3
"""t-SNE visualization for ViT metric-learning embeddings.

This script loads a MetricModel checkpoint, extracts normalized embeddings, and
saves a t-SNE plot plus a CSV table with coordinates/prediction metadata.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import ImageClassificationDataset, build_transforms, label_mapping, load_manifest
from src.models import MetricModel


def checkpoint_state(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ["model_state_dict", "model_state", "state_dict", "model"]:
        state = ckpt.get(key)
        if isinstance(state, dict):
            return state
    return ckpt


def clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state.items():
        new_key = key
        for prefix in ["module.", "model."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def infer_embedding_dim(state: dict[str, torch.Tensor], fallback: int) -> int:
    for key in ["projector.3.weight", "projector.2.weight"]:
        weight = state.get(key)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            return int(weight.shape[0])
    return fallback


def checkpoint_run_dir(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path).resolve()
    if path.parent.name == "checkpoints":
        return path.parent.parent
    return path.parent


def build_loader(
    manifest: str,
    split: str,
    class_to_idx: dict[str, int],
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    if split == "all":
        df = load_manifest(manifest, split=None)
    else:
        df = load_manifest(manifest, split=split)
    ds = ImageClassificationDataset(
        manifest_csv=manifest,
        split=None if split == "all" else split,
        class_to_idx=class_to_idx,
        transform=build_transforms(image_size=image_size, train=False),
        return_path=True,
    )
    # Keep the full dataframe for split labels when split=all.
    ds.df = df.reset_index(drop=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def extract_embeddings(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    model.eval()
    zs, ys, paths, labels, splits = [], [], [], [], []
    for x, y, p in loader:
        x = x.to(device, non_blocking=True)
        z = model(x)
        z = torch.nn.functional.normalize(z, dim=1)
        zs.append(z.cpu().numpy())
        ys.extend([int(v) for v in y.numpy().tolist()])
        paths.extend(list(p))
        batch_df = loader.dataset.df.iloc[len(labels):len(labels) + len(y)]
        labels.extend(batch_df["label"].astype(str).tolist())
        splits.extend(batch_df["split"].astype(str).tolist() if "split" in batch_df.columns else ["unknown"] * len(y))
    return np.concatenate(zs, axis=0), np.asarray(ys, dtype=np.int64), labels, paths, splits


def stratified_subsample(y: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or len(y) <= max_samples:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = max(1, math.ceil(max_samples / len(classes)))
    selected = []
    for cls in classes:
        idx = np.flatnonzero(y == cls)
        selected.append(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
    out = np.concatenate(selected)
    if len(out) > max_samples:
        out = rng.choice(out, size=max_samples, replace=False)
    return np.sort(out)


def compute_tsne(features: np.ndarray, perplexity: float, pca_dim: int, seed: int) -> np.ndarray:
    if len(features) < 3:
        raise ValueError("t-SNE needs at least 3 samples.")
    x = features
    if pca_dim > 0 and x.shape[1] > pca_dim:
        x = PCA(n_components=min(pca_dim, len(x) - 1), random_state=seed).fit_transform(x)
    safe_perplexity = min(perplexity, max(1.0, (len(x) - 1) / 3.0))
    return TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        init="pca",
        learning_rate="auto",
        metric="euclidean",
        random_state=seed,
    ).fit_transform(x)


def save_plot(df: pd.DataFrame, class_names: list[str], out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.get_cmap("tab10")
    markers = {"train": "o", "valid": "s", "val": "s", "test": "^", "unknown": "x"}
    for class_idx, class_name in enumerate(class_names):
        class_df = df[df["target"] == class_idx]
        if class_df.empty:
            continue
        for split in sorted(class_df["split"].astype(str).unique()):
            part = class_df[class_df["split"].astype(str) == split]
            ax.scatter(
                part["tsne_x"],
                part["tsne_y"],
                s=24,
                alpha=0.82,
                c=[colors(class_idx % 10)],
                marker=markers.get(split, "o"),
                edgecolors="white",
                linewidths=0.25,
                label=f"{class_name}/{split}",
            )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Draw t-SNE for ViT metric-learning embeddings.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", default="all", help="train, valid, test, or all")
    p.add_argument("--tag", default=None, help="Output name suffix, e.g. 7label or 7to3.")
    p.add_argument("--output-root", default=None, help="Deprecated; outputs are saved beside the checkpoint run folder.")
    p.add_argument("--experiment-name", default=None, help="Deprecated; outputs are saved beside the checkpoint run folder.")
    p.add_argument("--model-name", default=None)
    p.add_argument("--embedding-dim", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=2000)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--pca-dim", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA is not available. Pass --allow-cpu only for debugging.")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = clean_state_dict(checkpoint_state(ckpt))
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    full_df = load_manifest(args.manifest, split=None)
    class_to_idx, _ = label_mapping(full_df)
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    model_name = args.model_name or ckpt_args.get("model_name") or "vit_base_patch16_224"
    embedding_dim = args.embedding_dim or int(ckpt_args.get("embedding_dim") or infer_embedding_dim(state, 128))
    image_size = args.image_size or int(ckpt_args.get("image_size", 224))

    model = MetricModel(
        model_name=model_name,
        embedding_dim=embedding_dim,
        pretrained=False,
        dropout=float(ckpt_args.get("dropout", 0.0)),
        drop_path_rate=float(ckpt_args.get("drop_path_rate", 0.0)),
    )
    model.load_state_dict(state, strict=True)
    model.to(device)

    loader = build_loader(args.manifest, args.split, class_to_idx, image_size, args.batch_size, args.num_workers)
    features, targets, labels, paths, splits = extract_embeddings(model, loader, device)

    keep = stratified_subsample(targets, args.max_samples, args.seed)
    features = features[keep]
    targets = targets[keep]
    labels = np.asarray(labels, dtype=object)[keep]
    paths = np.asarray(paths, dtype=object)[keep]
    splits = np.asarray(splits, dtype=object)[keep]

    coords = compute_tsne(features, args.perplexity, args.pca_dim, args.seed)
    run_dir = checkpoint_run_dir(args.checkpoint)
    out_dir = run_dir / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.DataFrame({
        "path": paths,
        "label": labels,
        "target": targets,
        "split": splits,
        "tsne_x": coords[:, 0],
        "tsne_y": coords[:, 1],
    })
    suffix = args.tag or args.split
    rows.to_csv(out_dir / f"tsne_{suffix}.csv", index=False)
    np.savez_compressed(out_dir / f"tsne_{suffix}.npz", features=features, targets=targets, coords=coords, paths=paths)
    save_plot(rows, class_names, out_dir / f"tsne_{suffix}.png", f"Metric ViT t-SNE ({suffix}, cosine-trained embeddings)")

    print("TSNE_RUN_DIR=", run_dir)
    print("PNG=", out_dir / f"tsne_{suffix}.png")
    print("CSV=", out_dir / f"tsne_{suffix}.csv")


if __name__ == "__main__":
    main()
