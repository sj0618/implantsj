#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import create_loader, label_mapping, load_manifest
from src.models import create_classifier


def load_model(checkpoint: Path, model_name: str | None, image_size: int | None, manifest: Path, device):
    ckpt = torch.load(checkpoint, map_location="cpu")
    class_to_idx = {str(k): int(v) for k, v in ckpt.get("class_to_idx", {}).items()}
    if not class_to_idx:
        class_to_idx, _ = label_mapping(load_manifest(manifest, split=None))
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    ckpt_args = ckpt.get("args", {}) or {}
    resolved_model = model_name or ckpt_args.get("model_name") or "vit_base_patch16_224"
    resolved_size = int(image_size or ckpt_args.get("image_size", 224))

    model = create_classifier(resolved_model, num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, class_to_idx, idx_to_class, resolved_size


def predict_rows(model, loader, idx_to_class, device):
    rows = []
    with torch.no_grad():
        for x, y, paths in loader:
            x = x.to(device, non_blocking=True)
            probs = torch.softmax(model(x), dim=1).cpu()
            conf, pred = probs.max(dim=1)
            for path, true_idx, pred_idx, score in zip(paths, y.tolist(), pred.tolist(), conf.tolist()):
                rows.append(
                    {
                        "path": path,
                        "true_label": idx_to_class[int(true_idx)],
                        "pred_label": idx_to_class[int(pred_idx)],
                        "confidence": float(score),
                        "correct": idx_to_class[int(true_idx)] == idx_to_class[int(pred_idx)],
                    }
                )
    return rows


def main():
    p = argparse.ArgumentParser(description="Predict labels with a trained ViT checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    p.add_argument("--output", default="outputs/predictions.csv")
    p.add_argument("--model-name", default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--allow-cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA is not available. Pass --allow-cpu only for debugging.")

    checkpoint = Path(args.checkpoint)
    manifest = Path(args.manifest)
    model, class_to_idx, idx_to_class, image_size = load_model(
        checkpoint, args.model_name, args.image_size, manifest, device
    )
    loader = create_loader(
        manifest, args.split, class_to_idx, image_size, args.batch_size, args.num_workers,
        train=False, strong_aug=False, return_path=True
    )
    rows = predict_rows(model, loader, idx_to_class, device)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "true_label", "pred_label", "confidence", "correct"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} predictions to {out}")


if __name__ == "__main__":
    main()
