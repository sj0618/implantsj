#!/usr/bin/env python3
"""VLM feature 를 패치 토큰으로 붙인 ViT 를 학습한다.

    python3 tools/run.py patchtrain \
      --manifest /workspace/outputs/manifests/7label.csv \
      --vlm-features /workspace/data/aihub_all_13feature_21reason_vlm.jsonl \
      --model-name vit_base_patch16_224.augreg2_in21k_ft_in1k \
      --output-root /workspace/outputs

통제 실험용 대조군은 --feature-mode 로 바꾼다:
    patch    feature 를 패치 토큰으로 주입 (본 실험)
    none     이미지만 (baseline)
    zero     토큰은 붙이되 입력 0 (늘어난 파라미터 효과만)
    shuffle  feature 를 남의 것과 섞음 (진짜 쓰는지 검증)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import build_transforms, label_mapping, load_manifest
from src.metrics import (compute_metrics, save_confusion_matrix, save_metrics_json,
                         save_predictions)
from src.patch_feature import (FEATURE_MODES, PatchFeatureDataset, ViTWithFeaturePatches,
                               assert_finite_state_dict, load_feature_table)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a ViT with VLM features injected as patch tokens.")
    p.add_argument("--manifest", required=True, help="path,label,split 컬럼을 가진 CSV")
    p.add_argument("--vlm-features", default="", help="feature jsonl. --feature-mode none 이면 생략 가능")
    p.add_argument("--output-root", default="/workspace/outputs")
    p.add_argument("--experiment-name", default="vit_patch_feature")
    p.add_argument("--model-name", default="vit_base_patch16_224.augreg2_in21k_ft_in1k")
    p.add_argument("--feature-mode", choices=FEATURE_MODES, default="patch")
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--drop-path-rate", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--monitor", default="macro_f1", choices=["macro_f1", "accuracy"])
    p.add_argument("--no-strong-aug", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--require-feature-coverage", type=float, default=0.0,
                   help="train split 의 feature 커버리지가 이 값보다 낮으면 시작 전에 실패한다 (0~1)")
    return p.parse_args()


def resolve_device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if not allow_cpu:
        raise RuntimeError("CUDA is not available. Pass --allow-cpu only for debugging.")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def run_eval(model, loader, device, idx_to_class, split_name, out_dir: Path | None = None):
    model.eval()
    ys, ps, rows = [], [], []
    for img, y, feat, has, paths in loader:
        img, feat, has = img.to(device), feat.to(device), has.to(device)
        logits = model(img, feat, has)
        pred = logits.argmax(1).cpu()
        ys.extend(y.tolist())
        ps.extend(pred.tolist())
        if out_dir is not None:
            prob = logits.softmax(1).cpu()
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
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_metrics_json(out_dir / f"{split_name}_metrics.json", m)
        save_predictions(out_dir / f"{split_name}_predictions.csv", rows)
        cm = out_dir.parent / "confusion_matrices"
        cm.mkdir(parents=True, exist_ok=True)
        save_confusion_matrix(cm / f"{split_name}.csv", cm / f"{split_name}.png", ys, ps, names)
    return m


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.allow_cpu)

    # --- 라벨 공간 -----------------------------------------------------
    full = load_manifest(args.manifest, split=None)
    class_to_idx, idx_to_class = label_mapping(full)
    num_classes = len(class_to_idx)

    # --- feature 테이블 -------------------------------------------------
    table = None
    if args.feature_mode != "none":
        if not args.vlm_features:
            raise SystemExit("--feature-mode 가 none 이 아니면 --vlm-features 가 필요하다.")
        table = load_feature_table(args.vlm_features)

    tf_train = build_transforms(args.image_size, train=True, strong_aug=not args.no_strong_aug)
    tf_eval = build_transforms(args.image_size, train=False)

    def make(split, train):
        return PatchFeatureDataset(
            args.manifest, split, class_to_idx, table,
            transform=tf_train if train else tf_eval,
            feature_mode=args.feature_mode, shuffle_seed=args.seed,
        )

    ds_train, ds_valid, ds_test = make("train", True), make("valid", False), make("test", False)
    if len(ds_train) == 0:
        raise SystemExit(f"{args.manifest}: train split 이 비어 있다.")

    # --- 커버리지 보고 (feature 가 실제로 얼마나 붙었는지) ----------------
    print(f"[data] classes={num_classes}  train={len(ds_train)} valid={len(ds_valid)} test={len(ds_test)}")
    if args.feature_mode != "none":
        for name, ds in (("train", ds_train), ("valid", ds_valid), ("test", ds_test)):
            n, tot = ds.coverage()
            pct = 100.0 * n / tot if tot else 0.0
            print(f"[features] {name}: {n}/{tot} ({pct:.1f}%) 에 VLM feature 가 붙음")
        n_tr, tot_tr = ds_train.coverage()
        cov = n_tr / tot_tr if tot_tr else 0.0
        if args.feature_mode == "patch" and n_tr == 0:
            raise SystemExit(
                "train split 에 feature 가 하나도 매칭되지 않았다. "
                "manifest 의 path 와 feature jsonl 의 path 표기가 다른지 확인한다."
            )
        if cov < args.require_feature_coverage:
            raise SystemExit(
                f"train feature 커버리지 {cov:.1%} < 요구치 {args.require_feature_coverage:.1%}."
            )

    pin = device.type == "cuda"
    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   num_workers=args.num_workers, pin_memory=pin, drop_last=False)
    l_train, l_valid, l_test = dl(ds_train, True), dl(ds_valid, False), dl(ds_test, False)

    # --- 모델 -----------------------------------------------------------
    model = ViTWithFeaturePatches(
        model_name=args.model_name, num_classes=num_classes,
        num_features=ds_train.num_features, dims_per_feature=ds_train.dims_per_feature,
        pretrained=args.pretrained, feature_mode=args.feature_mode,
        drop_rate=args.dropout, drop_path_rate=args.drop_path_rate,
    ).to(device)
    extra = 0 if model.tokenizer is None else ds_train.num_features
    print(f"[model] {args.model_name}  embed_dim={model.embed_dim}  "
          f"feature_mode={args.feature_mode}  추가 토큰 {extra}개")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.output_root) / f"{stamp}_{args.experiment_name}"
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "class_to_idx.json").write_text(json.dumps(class_to_idx, indent=2, ensure_ascii=False), encoding="utf-8")
    if table is not None:
        (out / "feature_names.json").write_text(
            json.dumps({"names": table.names, "components": table.components}, indent=2, ensure_ascii=False),
            encoding="utf-8")

    # --- 학습 -----------------------------------------------------------
    hist, best, best_ep, bad = [], -1.0, -1, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n, skipped = 0.0, 0, 0
        for img, y, feat, has, _ in l_train:
            img, y = img.to(device), y.to(device)
            feat, has = feat.to(device), has.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(img, feat, has), y)
            # 발산한 배치로 가중치를 오염시키지 않는다.
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * y.size(0)
            n += y.size(0)
        sched.step()
        train_loss = tot / max(n, 1)

        vm = run_eval(model, l_valid, device, idx_to_class, "valid") if len(ds_valid) else {}
        score = float(vm.get(args.monitor, 0.0))
        row = {"epoch": ep, "train_loss": train_loss,
               "valid_accuracy": vm.get("accuracy"), "valid_macro_f1": vm.get("macro_f1"),
               "skipped_batches": skipped}
        hist.append(row)
        print(f"[epoch {ep:3d}/{args.epochs}] loss={train_loss:.4f} "
              f"valid_{args.monitor}={score:.4f}" + (f"  (skipped {skipped} batches)" if skipped else ""))

        if score > best:
            best, best_ep, bad = score, ep, 0
            assert_finite_state_dict(model, f"epoch={ep}")
            torch.save({"model_state_dict": model.state_dict(), "args": vars(args),
                        "class_to_idx": class_to_idx, "epoch": ep},
                       out / "checkpoints" / "best.pt")
            if vm:
                save_metrics_json(out / "metrics" / "best_valid_metrics.json", vm)
        else:
            bad += 1
            if args.patience > 0 and bad >= args.patience:
                print(f"[early stop] {args.patience} epochs without improvement.")
                break

    import pandas as pd
    pd.DataFrame(hist).to_csv(out / "metrics" / "history.csv", index=False)

    assert_finite_state_dict(model, "final")
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args),
                "class_to_idx": class_to_idx, "epoch": len(hist)},
               out / "checkpoints" / "last.pt")

    # --- 최종 test 평가는 best checkpoint 로 ------------------------------
    ckpt = out / "checkpoints" / "best.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model_state_dict"])
    if len(ds_test):
        tm = run_eval(model, l_test, device, idx_to_class, "test", out / "metrics")
        print(f"[test] accuracy={tm['accuracy']:.4f}  macro_f1={tm['macro_f1']:.4f}")

    print(f"[done] best epoch {best_ep} ({args.monitor}={best:.4f})")
    print(f"RUN_DIR= {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
