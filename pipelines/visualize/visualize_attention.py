#!/usr/bin/env python3
"""Attention rollout visualization for timm ViT classifiers.

Use `--image` for one image or `--manifest --split test` for many images.
This captures ViT self-attention from `model.blocks[*].attn.attn_drop`.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import build_transforms, label_mapping, load_manifest
from src.models import create_classifier


def unwrap_state_dict(ckpt: Any) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "model_state", "state_dict", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        return ckpt
    raise TypeError("Checkpoint must be a dict.")


def strip_prefixes(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state.items():
        new_key = key
        for prefix in ["module.", "model."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def infer_num_classes(ckpt: dict[str, Any] | None, state: dict[str, torch.Tensor] | None, fallback: int) -> int:
    if isinstance(ckpt, dict) and isinstance(ckpt.get("class_to_idx"), dict):
        return len(ckpt["class_to_idx"])
    if state:
        for key in ["head.weight", "classifier.weight", "fc.weight"]:
            weight = state.get(key)
            if isinstance(weight, torch.Tensor) and weight.ndim == 2:
                return int(weight.shape[0])
    return fallback


def class_names_from_checkpoint(ckpt: dict[str, Any] | None, class_to_idx: dict[str, int], num_classes: int) -> list[str]:
    if class_to_idx:
        inv = {idx: label for label, idx in class_to_idx.items()}
        return [inv.get(i, str(i)) for i in range(num_classes)]
    if isinstance(ckpt, dict) and isinstance(ckpt.get("class_to_idx"), dict):
        inv = {int(v): str(k) for k, v in ckpt["class_to_idx"].items()}
        return [inv.get(i, str(i)) for i in range(num_classes)]
    return [str(i) for i in range(num_classes)]


def checkpoint_run_dir(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path).resolve()
    if path.parent.name == "checkpoints":
        return path.parent.parent
    return path.parent


def disable_fused_attention(model: torch.nn.Module) -> None:
    for block in getattr(model, "blocks", []):
        attn = getattr(block, "attn", None)
        if attn is not None and hasattr(attn, "fused_attn"):
            attn.fused_attn = False


def collect_attention(model: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    attentions: list[torch.Tensor] = []
    handles = []

    def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if isinstance(output, torch.Tensor) and output.ndim == 4:
            attentions.append(output.detach())

    for block in getattr(model, "blocks", []):
        attn_drop = getattr(getattr(block, "attn", None), "attn_drop", None)
        if attn_drop is not None:
            handles.append(attn_drop.register_forward_hook(hook))

    try:
        logits = model(x)
    finally:
        for handle in handles:
            handle.remove()

    if not attentions:
        raise RuntimeError("No attention tensors captured. Use a timm ViT model with blocks[*].attn.attn_drop.")
    return logits, torch.stack(attentions, dim=0)


def attention_rollout(attentions: torch.Tensor, discard_ratio: float = 0.0) -> torch.Tensor:
    # [layers, batch, heads, tokens, tokens] -> [batch, patch_tokens]
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


def denormalize(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device)[:, None, None]
    img = x.detach() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()


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


def save_overlay(img_np: np.ndarray, heat: np.ndarray, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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


def build_rows(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    if args.image:
        rows.append({"path": args.image, "label": ""})
    if args.manifest:
        df = load_manifest(args.manifest, split=args.split)
        rows.extend(df.to_dict("records"))
    if not rows:
        raise ValueError("Pass --image or --manifest.")
    limit = args.max_samples if args.max_samples > 0 else len(rows)
    return pd.DataFrame(rows).head(limit)


def main() -> None:
    p = argparse.ArgumentParser(description="Save ViT attention rollout overlays for a trained classifier.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--image", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--output-root", default=None, help="Deprecated; outputs are saved beside the checkpoint run folder.")
    p.add_argument("--experiment-name", default=None, help="Deprecated; outputs are saved beside the checkpoint run folder.")
    p.add_argument("--model-name", default=None)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=64)
    p.add_argument("--discard-ratio", type=float, default=0.0)
    p.add_argument("--wrong-only", action="store_true")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA is not available. Pass --allow-cpu only for debugging.")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = strip_prefixes(unwrap_state_dict(ckpt))
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    class_to_idx = {str(k): int(v) for k, v in ckpt.get("class_to_idx", {}).items()} if isinstance(ckpt, dict) else {}
    if not class_to_idx and args.manifest:
        class_to_idx, _ = label_mapping(load_manifest(args.manifest, split=None))

    model_name = args.model_name or ckpt_args.get("model_name") or "vit_base_patch16_224"
    image_size = args.image_size or int(ckpt_args.get("image_size", 224))
    num_classes = infer_num_classes(ckpt, state, len(class_to_idx) or args.num_classes)
    class_names = class_names_from_checkpoint(ckpt, class_to_idx, num_classes)

    model = create_classifier(
        model_name,
        num_classes,
        pretrained=args.pretrained and ckpt is None,
        dropout=float(ckpt_args.get("dropout", 0.0)),
        drop_path_rate=float(ckpt_args.get("drop_path_rate", 0.0)),
    )
    disable_fused_attention(model)
    if state is not None:
        model.load_state_dict(state, strict=True)
    model.to(device).eval()

    tfm = build_transforms(image_size=image_size, train=False)
    run_dir = checkpoint_run_dir(args.checkpoint)
    out_dir = run_dir / "attention_maps"
    metrics_dir = run_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    out_rows = []
    for _, row in build_rows(args).iterrows():
        path = Path(str(row["path"]))
        true_label = str(row.get("label", ""))
        y_true = class_to_idx.get(true_label)

        pil = Image.open(path).convert("RGB")
        x = tfm(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, attns = collect_attention(model, x)
            probs = torch.softmax(logits, dim=1)
            pred = int(probs.argmax(dim=1).item())
            if args.wrong_only and y_true is not None and pred == y_true:
                continue
            rollout = attention_rollout(attns, discard_ratio=args.discard_ratio)[0]

        pred_label = class_names[pred] if pred < len(class_names) else str(pred)
        heat = tokens_to_heatmap(rollout, image_size=image_size)
        img_np = denormalize(x[0])
        title = f"pred={pred_label} ({float(probs[0, pred]):.3f})"
        if true_label:
            title = f"true={true_label} | {title}"
        safe_name = f"{saved:04d}_true-{true_label}_pred-{pred_label}_{path.stem}.png".replace("/", "_")
        save_overlay(img_np, heat, out_dir / safe_name, title=title)
        out_rows.append({
            "image": str(path),
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": float(probs[0, pred].item()),
            "file": str(out_dir / safe_name),
        })
        saved += 1

    sample_name = f"attention_samples_{args.split or 'image'}.csv"
    pd.DataFrame(out_rows).to_csv(metrics_dir / sample_name, index=False)
    print("ATTENTION_RUN_DIR=", run_dir)
    print("ATTENTION_DIR=", out_dir)
    print("saved=", saved)


if __name__ == "__main__":
    main()
