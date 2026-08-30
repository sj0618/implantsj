#!/usr/bin/env python3
"""패치-feature ViT checkpoint 를 평가한다.

    python3 tools/run.py patcheval \
      --checkpoint /workspace/outputs/<run>/checkpoints/best.pt \
      --manifest   /workspace/outputs/manifests/7label.csv \
      --vlm-features /workspace/data/aihub_all_13feature_21reason_vlm.jsonl \
      --split test

학습 때 쓴 설정(model-name, image-size, feature-mode, 라벨 순서)은
checkpoint 안에 저장돼 있으므로 다시 입력하지 않는다.

--feature-mode 를 주면 checkpoint 의 값을 덮어쓴다. 같은 가중치로
feature 를 넣었을 때와 뺐을 때(none/zero/shuffle)를 비교하는 ablation 용이다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import build_transforms
from src.metrics import (compute_metrics, save_confusion_matrix, save_metrics_json,
                         save_predictions)
from src.models import load_checkpoint
from src.patch_feature import (FEATURE_MODES, PatchFeatureDataset, ViTWithFeaturePatches,
                               load_feature_table)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a patch-feature ViT checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--vlm-features", default="")
    p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    p.add_argument("--output-dir", default="", help="생략하면 checkpoint 옆 eval/ 에 쓴다")
    p.add_argument("--feature-mode", choices=FEATURE_MODES, default=None,
                   help="ablation 용. 생략하면 checkpoint 에 저장된 값을 쓴다")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def resolve_device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if not allow_cpu:
        raise RuntimeError("CUDA is not available. Pass --allow-cpu only for debugging.")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    args = parse_args()
    device = resolve_device(args.allow_cpu)

    ck_path = Path(args.checkpoint)
    ck = load_checkpoint(ck_path, map_location="cpu")
    train_args = ck.get("args", {})
    class_to_idx = ck["class_to_idx"]
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    # 학습 때 설정을 그대로 따른다. feature_mode 만 필요하면 덮어쓴다.
    model_name = train_args.get("model_name", "vit_base_patch16_224")
    image_size = int(train_args.get("image_size", 224))
    mode = args.feature_mode or train_args.get("feature_mode", "patch")
    trained_mode = train_args.get("feature_mode", "patch")
    if mode != trained_mode:
        print(f"[ablation] 학습은 feature_mode={trained_mode} 였고 평가는 {mode} 로 한다.")

    table = None
    if mode != "none":
        feats = args.vlm_features or train_args.get("vlm_features", "")
        if not feats:
            raise SystemExit("--vlm-features 가 필요하다 (checkpoint 에도 없음).")
        table = load_feature_table(feats)

    ds = PatchFeatureDataset(
        args.manifest, args.split, class_to_idx, table,
        transform=build_transforms(image_size, train=False),
        feature_mode=mode, shuffle_seed=int(train_args.get("seed", 42)),
    )
    if len(ds) == 0:
        raise SystemExit(f"{args.manifest}: '{args.split}' split 이 비어 있다.")
    if mode != "none":
        n, tot = ds.coverage()
        print(f"[features] {args.split}: {n}/{tot} ({100.0*n/tot:.1f}%) 에 VLM feature 가 붙음")

    model = ViTWithFeaturePatches(
        model_name=model_name, num_classes=len(class_to_idx),
        num_features=ds.num_features, dims_per_feature=ds.dims_per_feature,
        pretrained=False, feature_mode=mode,
    ).to(device)
    # feature_mode 를 바꾸면 tokenizer 유무가 달라져 키가 안 맞을 수 있다.
    missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
    if missing:
        print(f"[warn] checkpoint 에 없는 파라미터 {len(missing)}개는 초기값을 쓴다 (예: {missing[:3]})")
    if unexpected:
        print(f"[warn] 이 모드에서 쓰이지 않는 파라미터 {len(unexpected)}개는 무시한다 (예: {unexpected[:3]})")
    model.eval()

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")

    ys, ps, rows = [], [], []
    with torch.no_grad():
        for img, y, feat, has, paths in loader:
            logits = model(img.to(device), feat.to(device), has.to(device))
            pred = logits.argmax(1).cpu()
            prob = logits.softmax(1).cpu()
            ys.extend(y.tolist()); ps.extend(pred.tolist())
            for i, path in enumerate(paths):
                rows.append({
                    "path": path,
                    "true": idx_to_class[int(y[i])],
                    "pred": idx_to_class[int(pred[i])],
                    "confidence": float(prob[i].max()),
                    "has_feature": bool(has[i].item()),
                })

    names = [idx_to_class[i] for i in range(len(idx_to_class))]
    m = compute_metrics(ys, ps, names)

    out = Path(args.output_dir) if args.output_dir else ck_path.parent.parent / "eval" / f"{args.split}_{mode}"
    out.mkdir(parents=True, exist_ok=True)
    save_metrics_json(out / "metrics.json", m)
    save_predictions(out / "predictions.csv", rows)
    save_confusion_matrix(out / "confusion_matrix.csv", out / "confusion_matrix.png", ys, ps, names)
    (out / "eval_args.json").write_text(
        json.dumps({"checkpoint": str(ck_path), "manifest": args.manifest,
                    "split": args.split, "feature_mode": mode,
                    "trained_feature_mode": trained_mode}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    # feature 유무별로 나눠 보면 feature 가 실제로 기여하는지 보인다.
    with_f = [(t, p) for (t, p), r in zip(zip(ys, ps), rows) if r["has_feature"]]
    without_f = [(t, p) for (t, p), r in zip(zip(ys, ps), rows) if not r["has_feature"]]
    print(f"[{args.split}] accuracy={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}  (n={len(ys)})")
    for tag, sub in (("feature 있음", with_f), ("feature 없음", without_f)):
        if sub:
            acc = sum(1 for t, p in sub if t == p) / len(sub)
            print(f"   {tag}: n={len(sub)}  accuracy={acc:.4f}")
    print(f"OUTPUT_DIR= {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
