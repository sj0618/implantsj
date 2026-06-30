#!/usr/bin/env python3
"""t-SNE visualization for supervised timm ViT classifier features."""
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import create_loader, label_mapping, load_manifest
from src.models import create_classifier


def clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state.items():
        new_key = key
        for prefix in ["module.", "model."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def class_names_from_mapping(class_to_idx: dict[str, int]) -> list[str]:
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}
    return [idx_to_class[i] for i in range(len(idx_to_class))]


def checkpoint_run_dir(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path).resolve()
    if path.parent.name == "checkpoints":
        return path.parent.parent
    return path.parent


def classifier_features(model: torch.nn.Module, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    if hasattr(model, "forward_features"):
        feats = model.forward_features(x)
        if hasattr(model, "forward_head"):
            try:
                feats = model.forward_head(feats, pre_logits=True)
            except TypeError:
                feats = model.forward_head(feats)
        elif feats.ndim == 3:
            feats = feats[:, 0]
    else:
        feats = model(x)

    if isinstance(feats, dict):
        feats = feats.get("features", feats.get("pre_logits", feats.get("logits")))
    if feats.ndim > 2:
        feats = torch.nn.functional.adaptive_avg_pool2d(feats, output_size=1).flatten(1)
    if normalize:
        feats = torch.nn.functional.normalize(feats, dim=1)
    return feats


@torch.no_grad()
def extract_features(model, loader, device, class_names: list[str]) -> dict[str, Any]:
    model.eval()
    features, targets, labels, paths, splits = [], [], [], [], []
    preds, confidences = [], []

    seen = 0
    for batch in loader:
        if len(batch) == 3:
            x, y, p = batch
        else:
            x, y = batch
            p = [""] * len(y)
        x = x.to(device, non_blocking=True)
        z = classifier_features(model, x, normalize=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)

        features.append(z.cpu().numpy())
        targets.extend([int(v) for v in y.numpy().tolist()])
        preds.extend([int(v) for v in pred.cpu().numpy().tolist()])
        confidences.extend([float(v) for v in probs.max(dim=1).values.cpu().numpy().tolist()])
        paths.extend(list(p))

        batch_df = loader.dataset.df.iloc[seen:seen + len(y)]
        labels.extend(batch_df["label"].astype(str).tolist())
        splits.extend(batch_df["split"].astype(str).tolist() if "split" in batch_df.columns else ["unknown"] * len(y))
        seen += len(y)

    return {
        "features": np.concatenate(features, axis=0),
        "targets": np.asarray(targets, dtype=np.int64),
        "labels": np.asarray(labels, dtype=object),
        "paths": np.asarray(paths, dtype=object),
        "splits": np.asarray(splits, dtype=object),
        "preds": np.asarray(preds, dtype=np.int64),
        "pred_labels": np.asarray([class_names[i] if i < len(class_names) else str(i) for i in preds], dtype=object),
        "confidences": np.asarray(confidences, dtype=np.float32),
    }


def stratified_subsample(y: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or len(y) <= max_samples:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    selected = []
    classes = np.unique(y)
    per_class = max(1, math.ceil(max_samples / len(classes)))
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Draw t-SNE for supervised ViT classifier features.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", default="all", help="train, valid, test, or all")
    p.add_argument("--output-root", default=None, help="Deprecated; outputs are saved beside the checkpoint run folder.")
    p.add_argument("--experiment-name", default=None, help="Deprecated; outputs are saved beside the checkpoint run folder.")
    p.add_argument("--model-name", default=None)
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
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    full_df = load_manifest(args.manifest, split=None)
    class_to_idx = {str(k): int(v) for k, v in ckpt.get("class_to_idx", {}).items()} if isinstance(ckpt, dict) else {}
    if not class_to_idx:
        class_to_idx, _ = label_mapping(full_df)
    class_names = class_names_from_mapping(class_to_idx)

    model_name = args.model_name or ckpt_args.get("model_name") or "vit_base_patch16_224"
    image_size = args.image_size or int(ckpt_args.get("image_size", 224))
    model = create_classifier(model_name, len(class_names), pretrained=False)
    model.load_state_dict(clean_state_dict(ckpt["model_state_dict"]), strict=True)
    model.to(device)

    split_for_loader = "train" if args.split == "all" else args.split
    loader = create_loader(
        args.manifest,
        split_for_loader,
        class_to_idx,
        image_size,
        args.batch_size,
        args.num_workers,
        train=False,
        return_path=True,
    )
    if args.split == "all":
        loader.dataset.df = full_df.reset_index(drop=True)

    data = extract_features(model, loader, device, class_names)
    keep = stratified_subsample(data["targets"], args.max_samples, args.seed)
    features = data["features"][keep]
    coords = compute_tsne(features, args.perplexity, args.pca_dim, args.seed)

    rows = pd.DataFrame({
        "path": data["paths"][keep],
        "label": data["labels"][keep],
        "target": data["targets"][keep],
        "split": data["splits"][keep],
        "pred": data["preds"][keep],
        "pred_label": data["pred_labels"][keep],
        "confidence": data["confidences"][keep],
        "correct": data["preds"][keep] == data["targets"][keep],
        "tsne_x": coords[:, 0],
        "tsne_y": coords[:, 1],
    })

    run_dir = checkpoint_run_dir(args.checkpoint)
    out_dir = run_dir / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_dir / f"tsne_{args.split}.csv", index=False)
    np.savez_compressed(out_dir / f"tsne_{args.split}.npz", features=features, targets=data["targets"][keep], coords=coords, paths=data["paths"][keep])
    save_plot(rows, class_names, out_dir / f"tsne_{args.split}.png", f"Classifier ViT t-SNE ({args.split}, {len(class_names)} labels)")

    print("TSNE_RUN_DIR=", run_dir)
    print("PNG=", out_dir / f"tsne_{args.split}.png")
    print("CSV=", out_dir / f"tsne_{args.split}.csv")


if __name__ == "__main__":
    main()
