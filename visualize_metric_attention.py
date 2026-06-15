#!/usr/bin/env python3
"""Attention rollout overlays for ViT metric-learning checkpoints.

Prediction labels are computed by nearest train-set prototype in the embedding
space; attention heatmaps are ViT attention rollout maps.
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
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import build_transforms, label_mapping, load_manifest
from src.models import MetricModel


def checkpoint_run_dir(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path).resolve()
    if path.parent.name == "checkpoints":
        return path.parent.parent
    return path.parent


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
    weight = state.get("projector.3.weight")
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    return fallback


def disable_fused_attention(model: torch.nn.Module) -> None:
    backbone = getattr(model, "backbone", model)
    for block in getattr(backbone, "blocks", []):
        attn = getattr(block, "attn", None)
        if attn is not None and hasattr(attn, "fused_attn"):
            attn.fused_attn = False


def collect_attention(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    backbone = getattr(model, "backbone", model)
    attentions: list[torch.Tensor] = []
    handles = []

    def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if isinstance(output, torch.Tensor) and output.ndim == 4:
            attentions.append(output.detach())

    for block in getattr(backbone, "blocks", []):
        attn_drop = getattr(getattr(block, "attn", None), "attn_drop", None)
        if attn_drop is not None:
            handles.append(attn_drop.register_forward_hook(hook))
    try:
        _ = model(x)
    finally:
        for handle in handles:
            handle.remove()
    if not attentions:
        raise RuntimeError("No attention tensors captured. Use a timm ViT metric model.")
    return torch.stack(attentions, dim=0)


def attention_rollout(attentions: torch.Tensor, discard_ratio: float = 0.0) -> torch.Tensor:
    attn = attentions.mean(dim=2)
    eye = torch.eye(attn.size(-1), device=attn.device).view(1, 1, attn.size(-1), attn.size(-1))
    attn = attn + eye
    attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    if discard_ratio > 0:
        flat = attn.flatten(2)
        n_drop = int(flat.size(-1) * discard_ratio)
        if n_drop > 0:
            _, indices = flat.topk(n_drop, dim=-1, largest=False)
            flat.scatter_(dim=-1, index=indices, value=0)
            attn = flat.view_as(attn)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    joint = attn[0]
    for layer_idx in range(1, attn.size(0)):
        joint = attn[layer_idx].bmm(joint)
    return joint[:, 0, 1:]


def tokens_to_heatmap(tokens: torch.Tensor, image_size: int) -> np.ndarray:
    token_count = int(tokens.numel())
    side = int(math.sqrt(token_count))
    if side * side != token_count:
        raise RuntimeError(f"Patch token count is not square: {token_count}")
    heat = tokens.reshape(1, 1, side, side)
    heat = torch.nn.functional.interpolate(heat, size=(image_size, image_size), mode="bilinear", align_corners=False)
    heat = heat[0, 0]
    heat = heat - heat.min()
    heat = heat / heat.max().clamp_min(1e-12)
    return heat.detach().cpu().numpy()


def denormalize(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device)[:, None, None]
    return (x.detach() * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def save_overlay(img_np: np.ndarray, heat: np.ndarray, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.imshow(img_np)
    ax.imshow(heat, cmap="jet", alpha=0.45)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


@torch.no_grad()
def extract_embeddings(model: torch.nn.Module, df: pd.DataFrame, tfm, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    zs, ys = [], []
    for start in range(0, len(df), batch_size):
        part = df.iloc[start:start + batch_size]
        xs = []
        for raw_path in part["path"].astype(str).tolist():
            xs.append(tfm(Image.open(raw_path).convert("RGB")))
        x = torch.stack(xs, dim=0).to(device)
        z = torch.nn.functional.normalize(model(x), dim=1)
        zs.append(z.cpu().numpy())
        ys.extend(part["_target"].astype(int).tolist())
    return np.concatenate(zs, axis=0), np.asarray(ys, dtype=np.int64)


def build_prototypes(train_z: np.ndarray, train_y: np.ndarray) -> tuple[list[int], np.ndarray]:
    classes = sorted(set(train_y.tolist()))
    protos = []
    for cls in classes:
        proto = train_z[train_y == cls].mean(axis=0)
        proto = proto / max(1e-12, np.linalg.norm(proto))
        protos.append(proto)
    return classes, np.stack(protos, axis=0)


def main() -> None:
    p = argparse.ArgumentParser(description="Save metric ViT attention rollout overlays.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--tag", default=None)
    p.add_argument("--model-name", default=None)
    p.add_argument("--embedding-dim", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-samples", type=int, default=32)
    p.add_argument("--discard-ratio", type=float, default=0.0)
    p.add_argument("--wrong-only", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA is not available. Pass --allow-cpu only for debugging.")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = clean_state_dict(checkpoint_state(ckpt))
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    model_name = args.model_name or ckpt_args.get("model_name") or "vit_base_patch16_224"
    embedding_dim = args.embedding_dim or int(ckpt_args.get("embedding_dim") or infer_embedding_dim(state, 128))
    image_size = args.image_size or int(ckpt_args.get("image_size", 224))

    full_df = load_manifest(args.manifest, split=None)
    class_to_idx, idx_to_class = label_mapping(full_df)
    full_df = full_df.copy()
    full_df["_target"] = full_df["label"].astype(str).map(class_to_idx)
    train_df = full_df[full_df["split"].astype(str) == "train"].reset_index(drop=True)
    query_df = full_df[full_df["split"].astype(str) == args.split].reset_index(drop=True)
    if args.max_samples > 0:
        query_df = query_df.head(args.max_samples)

    model = MetricModel(model_name=model_name, embedding_dim=embedding_dim, pretrained=False, dropout=float(ckpt_args.get("dropout", 0.0)), drop_path_rate=float(ckpt_args.get("drop_path_rate", 0.0)))
    model.load_state_dict(state, strict=True)
    disable_fused_attention(model)
    model.to(device).eval()
    tfm = build_transforms(image_size=image_size, train=False)

    train_z, train_y = extract_embeddings(model, train_df, tfm, device, args.batch_size)
    classes, protos = build_prototypes(train_z, train_y)
    protos_t = torch.as_tensor(protos, dtype=torch.float32, device=device)

    run_dir = checkpoint_run_dir(args.checkpoint)
    tag = args.tag or f"metric_{Path(args.manifest).stem}_{args.split}"
    out_dir = run_dir / "attention_maps" / tag
    metrics_dir = run_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    saved = 0
    for _, row in query_df.iterrows():
        path = Path(str(row["path"]))
        true_idx = int(row["_target"])
        true_label = str(row["label"])
        x = tfm(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            z = torch.nn.functional.normalize(model(x), dim=1)
            sims = z @ protos_t.T
            pos = int(sims.argmax(dim=1).item())
            pred = int(classes[pos])
            conf = float(sims[0, pos].item())
            if args.wrong_only and pred == true_idx:
                continue
            attns = collect_attention(model, x)
            rollout = attention_rollout(attns, discard_ratio=args.discard_ratio)[0]

        pred_label = idx_to_class[pred]
        heat = tokens_to_heatmap(rollout, image_size)
        safe_name = f"{saved:04d}_true-{true_label}_pred-{pred_label}_{path.stem}.png".replace("/", "_")
        save_overlay(denormalize(x[0]), heat, out_dir / safe_name, f"true={true_label} | pred={pred_label} ({conf:.3f})")
        rows.append({"image": str(path), "true_label": true_label, "pred_label": pred_label, "similarity": conf, "file": str(out_dir / safe_name)})
        saved += 1

    pd.DataFrame(rows).to_csv(metrics_dir / f"metric_attention_samples_{tag}.csv", index=False)
    print("ATTENTION_RUN_DIR=", run_dir)
    print("ATTENTION_DIR=", out_dir)
    print("saved=", saved)


if __name__ == "__main__":
    main()
