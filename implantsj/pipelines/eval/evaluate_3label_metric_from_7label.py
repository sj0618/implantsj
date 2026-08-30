#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import IMAGE_EXTENSIONS, build_transforms, label_mapping, load_manifest
from src.metrics import compute_metrics, save_confusion_matrix, save_metrics_json, save_predictions
from src.models import load_checkpoint


CANONICAL_3LABELS = {
    "bego": "Bego",
    "bicon": "Bicon",
    "iti": "ITI",
}


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Stage1MetricModel(nn.Module):
    def __init__(self, model_name: str, projection_hidden_dim: int, projection_dim: int) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        feature_dim = int(getattr(self.backbone, "num_features"))
        self.projection = ProjectionHead(feature_dim, projection_hidden_dim, projection_dim)

    def forward(self, x: torch.Tensor, use_projection: bool = True) -> torch.Tensor:
        z = self.backbone(x)
        if use_projection:
            z = self.projection(z)
        return F.normalize(z, dim=1)


class ProjectorMetricModel(nn.Module):
    def __init__(self, model_name: str, embedding_dim: int, dropout: float = 0.0, drop_path_rate: float = 0.0) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            drop_rate=dropout,
            drop_path_rate=drop_path_rate,
        )
        feature_dim = int(getattr(self.backbone, "num_features"))
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor, use_projection: bool = True) -> torch.Tensor:
        z = self.backbone(x)
        if use_projection:
            z = self.projector(z)
        return F.normalize(z, dim=1)


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, manifest: Path, split: str, class_to_idx: Dict[str, int], image_size: int) -> None:
        self.df = load_manifest(manifest, split=split)
        self.class_to_idx = class_to_idx
        self.transform = build_transforms(image_size=image_size, train=False)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = Path(row["path"])
        img = Image.open(path).convert("RGB")
        return self.transform(img), self.class_to_idx[str(row["label"])], str(path)


def canonical_label(label: str) -> str:
    key = label.strip().casefold()
    if key not in CANONICAL_3LABELS:
        raise ValueError(f"Unexpected 3-label class folder: {label}")
    return CANONICAL_3LABELS[key]


def build_3label_manifest(data_root: Path, out_csv: Path) -> pd.DataFrame:
    rows: List[dict] = []
    for split in ["train", "valid", "test"]:
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
            label = canonical_label(class_dir.name)
            for img in sorted(class_dir.rglob("*")):
                if img.is_file() and img.suffix.lower() in IMAGE_EXTENSIONS:
                    rows.append({"path": str(img.resolve()), "label": label, "split": split})
    if not rows:
        raise RuntimeError(f"No images found under {data_root}")
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def checkpoint_state(ckpt: dict) -> dict:
    for key in ["model_state", "model_state_dict", "state_dict", "model"]:
        state = ckpt.get(key)
        if isinstance(state, dict):
            return state
    return ckpt


def normalize_state_dict(state: dict) -> dict:
    normalized = {}
    for key, value in state.items():
        clean_key = str(key)
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module."):]
        if clean_key.startswith("model."):
            clean_key = clean_key[len("model."):]
        normalized[clean_key] = value
    return normalized


def infer_model_args(ckpt: dict, args) -> Tuple[str, int, int, int]:
    cfg = ckpt.get("config") or {}
    ckpt_args = ckpt.get("args") or {}
    model_name = args.model_name or cfg.get("model_name") or ckpt_args.get("model_name") or "vit_small_patch16_224.augreg_in21k_ft_in1k"
    image_size = args.image_size or int(cfg.get("image_size") or ckpt_args.get("image_size") or 224)
    hidden_dim = args.proj_hidden_dim or int(cfg.get("proj_hidden_dim") or 128)
    projection_dim = args.proj_output_dim or int(cfg.get("proj_output_dim") or ckpt_args.get("embedding_dim") or 64)
    return model_name, image_size, hidden_dim, projection_dim


def load_stage1_metric_model(checkpoint: Path, args, device: torch.device):
    ckpt = load_checkpoint(checkpoint, map_location="cpu")
    state = normalize_state_dict(checkpoint_state(ckpt))
    model_name, image_size, hidden_dim, projection_dim = infer_model_args(ckpt, args)
    uses_projector = any(key.startswith("projector.") for key in state)
    uses_projection = any(key.startswith("projection.") for key in state)
    if uses_projector:
        model = ProjectorMetricModel(model_name, projection_dim)
        model_kind = "metric_model_projector"
    else:
        model = Stage1MetricModel(model_name, hidden_dim, projection_dim)
        model_kind = "stage1_projection" if uses_projection else "backbone_only"
    target = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state.items():
        if key in target and tuple(target[key].shape) == tuple(value.shape):
            filtered[key] = value
        elif f"backbone.{key}" in target and tuple(target[f"backbone.{key}"].shape) == tuple(value.shape):
            filtered[f"backbone.{key}"] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    model.to(device).eval()
    projection_loaded = any(k.startswith("projection.") or k.startswith("projector.") for k in filtered)
    metadata = {
        "checkpoint": str(checkpoint),
        "model_kind": model_kind,
        "model_name": model_name,
        "image_size": image_size,
        "projection_hidden_dim": hidden_dim,
        "projection_dim": projection_dim,
        "loaded_keys": len(filtered),
        "skipped_keys": len(skipped),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "use_projection": projection_loaded and not getattr(args, "backbone_only", False),
    }
    return model, image_size, metadata


@torch.no_grad()
def extract_embeddings(model, loader, device: torch.device, use_projection: bool):
    zs, ys, paths = [], [], []
    for x, y, p in loader:
        x = x.to(device, non_blocking=True)
        z = model(x, use_projection=use_projection).cpu().numpy()
        zs.append(z)
        ys.extend(y.numpy().tolist())
        paths.extend(list(p))
    return np.concatenate(zs, axis=0), np.asarray(ys), paths


def prototype_predict(train_z: np.ndarray, train_y: np.ndarray, query_z: np.ndarray):
    classes = sorted(set(train_y.tolist()))
    protos = []
    for cls in classes:
        proto = train_z[train_y == cls].mean(axis=0)
        proto = proto / max(1e-12, np.linalg.norm(proto))
        protos.append(proto)
    protos = np.stack(protos, axis=0)
    q = query_z / np.maximum(1e-12, np.linalg.norm(query_z, axis=1, keepdims=True))
    sims = q @ protos.T
    pos = sims.argmax(axis=1)
    pred = np.asarray([classes[i] for i in pos])
    conf = sims.max(axis=1)
    return pred, conf


def save_projection_distribution(embedding_cache: dict, class_to_idx: Dict[str, int], out_dir: Path) -> tuple[Path, Path, Path]:
    import plotly.express as px
    import plotly.graph_objects as go
    from umap import UMAP
    from pipelines.visualize.visualize_metric_tsne import compute_tsne, save_plot

    class_names = [label for label, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    if set(class_names) != set(CANONICAL_3LABELS.values()):
        raise ValueError(f"Projection plot requires Bego, Bicon, and ITI; got {class_names}")
    features = np.concatenate([embedding_cache[split][0] for split in ["train", "valid", "test"]])
    targets = np.concatenate([embedding_cache[split][1] for split in ["train", "valid", "test"]])
    paths = sum((embedding_cache[split][2] for split in ["train", "valid", "test"]), [])
    splits = np.concatenate([
        np.full(len(embedding_cache[split][1]), split, dtype=object)
        for split in ["train", "valid", "test"]
    ])
    coordinates = compute_tsne(features, perplexity=30.0, pca_dim=50, seed=42)
    frame = pd.DataFrame({
        "path": paths,
        "label": [class_names[int(target)] for target in targets],
        "target": targets,
        "split": splits,
        "tsne_x": coordinates[:, 0],
        "tsne_y": coordinates[:, 1],
    })
    plot_dir = out_dir / "embeddings"
    plot_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(plot_dir / "projection_tsne.csv", index=False)
    np.savez_compressed(
        plot_dir / "projection_embeddings.npz",
        features=features,
        targets=targets,
        splits=splits,
        paths=np.asarray(paths, dtype=object),
    )

    output = plot_dir / "projection_tsne.png"
    save_plot(frame, class_names, output, "7→3 metric projection distribution (t-SNE)")

    coordinates_3d = UMAP(n_components=3, random_state=42).fit_transform(features)
    frame_3d = frame[["path", "label", "target", "split"]].copy()
    frame_3d[["umap_x", "umap_y", "umap_z"]] = coordinates_3d
    frame_3d.to_csv(plot_dir / "projection_umap_3d.csv", index=False)
    figure_3d = px.scatter_3d(
        frame_3d,
        x="umap_x",
        y="umap_y",
        z="umap_z",
        color="label",
        symbol="split",
        hover_data=["path", "label", "split"],
        color_discrete_map={"Bego": "#1f77b4", "Bicon": "#ff7f0e", "ITI": "#2ca02c"},
        category_orders={"label": class_names, "split": ["train", "valid", "test"]},
        title="7→3 metric projection distribution (3D UMAP)",
    )
    figure_3d.update_traces(marker={"size": 3, "opacity": 0.75})
    html = plot_dir / "projection_umap_3d.html"
    figure_3d.write_html(html, include_plotlyjs=True, full_html=True)

    order = np.lexsort((splits.astype(str), targets))
    ordered = frame.iloc[order].reset_index(drop=True)
    ordered.insert(0, "heatmap_row", np.arange(len(ordered)))
    ordered[["heatmap_row", "path", "label", "split"]].to_csv(plot_dir / "projection_heatmap_rows.csv", index=False)
    centers, offset = [], 0
    for label in class_names:
        count = int((ordered["label"] == label).sum())
        centers.append(offset + (count - 1) / 2)
        offset += count
    heatmap = go.Figure(go.Heatmap(
        z=features[order],
        x=[f"d{i + 1}" for i in range(features.shape[1])],
        y=np.arange(len(features)),
        colorscale="RdBu",
        zmid=0,
        colorbar={"title": "value"},
        hovertemplate="row=%{y}<br>dimension=%{x}<br>value=%{z:.5f}<extra></extra>",
    ))
    heatmap.update_layout(
        title=f"7→3 metric raw projection ({features.shape[1]} dimensions)",
        xaxis_title="projection dimension",
        yaxis={"title": "samples grouped by label", "tickmode": "array", "tickvals": centers, "ticktext": class_names, "autorange": "reversed"},
        height=900,
    )
    heatmap_html = plot_dir / f"projection_{features.shape[1]}d_heatmap.html"
    heatmap.write_html(heatmap_html, include_plotlyjs=True, full_html=True)
    return output, html, heatmap_html


def evaluate_split(model, manifest: Path, class_to_idx: Dict[str, int], image_size: int, batch_size: int, num_workers: int, device: torch.device, split: str, out_dir: Path, use_projection: bool, embedding_cache: dict | None = None):
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    embedding_cache = embedding_cache if embedding_cache is not None else {}
    for source_split in ["train", split]:
        if source_split not in embedding_cache:
            dataset = ImagePathDataset(manifest, source_split, class_to_idx, image_size)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
            embedding_cache[source_split] = extract_embeddings(model, loader, device, use_projection)
    train_z, train_y, _ = embedding_cache["train"]
    query_z, query_y, paths = embedding_cache[split]
    pred, conf = prototype_predict(train_z, train_y, query_z)
    metrics = compute_metrics(query_y, pred, class_names)
    save_metrics_json(out_dir / "metrics" / f"{split}_prototype_metrics.json", metrics)
    save_confusion_matrix(out_dir / "confusion_matrices" / f"{split}_prototype.csv", out_dir / "confusion_matrices" / f"{split}_prototype.png", query_y, pred, class_names)
    rows = []
    for p, t, pr, cf in zip(paths, query_y, pred, conf):
        rows.append({
            "path": p,
            "true_idx": int(t),
            "true_label": idx_to_class[int(t)],
            "pred_idx": int(pr),
            "pred_label": idx_to_class[int(pr)],
            "similarity": float(cf),
            "correct": bool(int(t) == int(pr)),
        })
    save_predictions(out_dir / "metrics" / f"{split}_prototype_predictions.csv", rows)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained embedding model on a canonical 3-label prototype task.")
    parser.add_argument("--checkpoint", default="/workspace/data_1_2/outputs_two_stage_metric/stage1/best_stage1.pth")
    parser.add_argument("--data-root", default="/workspace/implant_python_only_final/data/3label")
    parser.add_argument("--manifest", default="/workspace/implant_python_only_final/data/manifests/3label.csv")
    parser.add_argument("--output-root", default="/workspace/implant_python_only_final/outputs")
    parser.add_argument("--experiment-name", default="metric_supcon_to_3label")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--proj-hidden-dim", type=int, default=None)
    parser.add_argument("--proj-output-dim", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--backbone-only", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    if device.type == "cpu" and not args.allow_cpu:
        raise RuntimeError("CUDA/MPS is not available. Pass --allow-cpu only for debugging.")

    manifest = Path(args.manifest)
    df = (
        load_manifest(manifest, split=None)
        if manifest.is_file()
        else build_3label_manifest(Path(args.data_root), manifest)
    )
    class_to_idx, _ = label_mapping(df)
    run_dir = Path(args.output_root) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.experiment_name}"
    for sub in ["metrics", "confusion_matrices"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    model, image_size, metadata = load_stage1_metric_model(Path(args.checkpoint), args, device)
    counts_df = df.groupby(["split", "label"]).size().reset_index(name="count")
    metadata.update({
        "device": str(device),
        "manifest": str(manifest),
        "class_to_idx": class_to_idx,
        "sample_counts": [
            {"split": str(row.split), "label": str(row.label), "count": int(row.count)}
            for row in counts_df.itertuples(index=False)
        ],
    })
    (run_dir / "args.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    results = {}
    embedding_cache = {}
    for split in ["valid", "test"]:
        results[split] = evaluate_split(
            model=model,
            manifest=manifest,
            class_to_idx=class_to_idx,
            image_size=image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            split=split,
            out_dir=run_dir,
            use_projection=bool(metadata["use_projection"]),
            embedding_cache=embedding_cache,
        )
    if metadata["use_projection"]:
        projection_tsne, projection_umap, projection_heatmap = save_projection_distribution(embedding_cache, class_to_idx, run_dir)
        print(f"PROJECTION_TSNE={projection_tsne}")
        print(f"PROJECTION_UMAP_3D={projection_umap}")
        print(f"PROJECTION_HEATMAP={projection_heatmap}")

    print(f"RUN_DIR={run_dir}")
    for split, metrics in results.items():
        print(f"{split}: accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} weighted_f1={metrics['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
