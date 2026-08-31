#!/usr/bin/env python3
"""
zero-shot / FT 공통 평가. test 48장 기준. (RULES.md 2번: best epoch 확정 후 딱 한 번)

threshold를 0.05~0.50으로 스윕해 PR 곡선을 남기고, 각 모델의 '최적 threshold'에서
비교한다. 0.30 하나로 고정하면 zero-shot이 부당하게 낮게 나온다.

  # zero-shot
  python eval_coco.py --out results/00_zeroshot --split test

  # fine-tuned
  python eval_coco.py --out results/01/lr_1e-4 --split test \
      --lora <GDINO_ROOT>/weights/checkpoint_epoch_28.pth

출력 (규칙 4의 eval 산출물):
  config.json / threshold_sweep.csv / metrics.json / predictions.json
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

TEXT_PROMPT = "implant"
SWEEP = [round(0.05 + 0.05 * i, 2) for i in range(10)]      # 0.05 ~ 0.50
IOU_MATCH = 0.5      # P/R/F1 매칭 기준


def load_deps():
    """repo 의존 import. 로컬에는 repo가 없으므로 실행 시점에만 부른다."""
    try:
        import torch
        from torchvision.ops import box_convert
        from groundingdino.util.inference import load_model, load_image, predict
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as e:
        sys.exit(f"[ERROR] {e}\n"
                 "  repo 환경에서 실행해야 한다. (RULES.md 8번: 학습·추론은 pod)\n"
                 "  repo 루트에서 실행하고 LD_LIBRARY_PATH를 잡았는지 확인할 것.")
    return torch, box_convert, load_model, load_image, predict, COCO, COCOeval


class MC:
    """repo의 load_model이 기대하는 model_config 객체."""
    def __init__(self, cfg, weights, lora=None):
        self.config_path = cfg
        self.weights_path = weights
        self.lora_weights = lora
        self.lora_weigths = lora        # repo에 오타 필드가 있어 양쪽 다 채운다


def match_greedy(preds, gts, iou_thr):
    """score 내림차순으로 GT에 1:1 매칭. TP/FP/FN 계산."""
    def iou(a, b):
        ax, ay, aw, ah = a; bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
        inter = iw * ih
        u = aw * ah + bw * bh - inter
        return inter / u if u > 0 else 0.0

    used = set()
    tp = 0
    for p in sorted(preds, key=lambda p: -p["score"]):
        best, bi = 0.0, -1
        for i, g in enumerate(gts):
            if i in used:
                continue
            v = iou(p["bbox"], g)
            if v > best:
                best, bi = v, i
        if best >= iou_thr:
            used.add(bi); tp += 1
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="results 폴더. 예: results/00_zeroshot")
    ap.add_argument("--data-root", default="/workspace/dataset2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--gdino-root", default=os.environ.get("GDINO_ROOT", ""))
    ap.add_argument("--lora", default=None, help="LoRA checkpoint. 없으면 zero-shot")
    ap.add_argument("--text-threshold", type=float, default=0.20)
    a = ap.parse_args()

    torch, box_convert, load_model, load_image, predict, COCO, COCOeval = load_deps()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "viz").mkdir(exist_ok=True); (out / "attention").mkdir(exist_ok=True)

    split_dir = Path(a.data_root) / a.split
    gt_json = split_dir / "annotations.json"
    img_dir = split_dir / "images"
    if not gt_json.exists():
        sys.exit(f"[ERROR] GT 없음: {gt_json}")

    g = Path(a.gdino_root)
    model_cfg = str(g / "groundingdino/config/GroundingDINO_SwinT_OGC.py")
    base_w = str(g / "weights/groundingdino_swint_ogc.pth")

    coco_gt = COCO(str(gt_json))
    cat_id = coco_gt.dataset["categories"][0]["id"]
    name2id = {im["file_name"]: im["id"] for im in coco_gt.dataset["images"]}
    id2wh = {im["id"]: (im["width"], im["height"]) for im in coco_gt.dataset["images"]}
    gts_by_img = {i: [] for i in id2wh}
    for an in coco_gt.dataset["annotations"]:
        gts_by_img[an["image_id"]].append(an["bbox"])

    kind = "zero-shot" if not a.lora else f"FT ({Path(a.lora).name})"
    print(f"[eval] {kind} | {a.split} {len(name2id)}장 / "
          f"{len(coco_gt.dataset['annotations'])}박스")

    (out / "config.json").write_text(json.dumps({
        "kind": kind,
        "lora": a.lora,
        "base_weights": base_w,
        "model_config": model_cfg,
        "split": a.split,
        "data_root": str(a.data_root),
        "gt_json": str(gt_json),
        "n_images": len(name2id),
        "n_gt_boxes": len(coco_gt.dataset["annotations"]),
        "text_prompt": TEXT_PROMPT,
        "text_threshold": a.text_threshold,
        "sweep": SWEEP,
        "iou_match": IOU_MATCH,
    }, indent=2, ensure_ascii=False))

    model = load_model(MC(model_cfg, base_w, a.lora), use_lora=bool(a.lora))

    # 가장 낮은 threshold로 한 번만 추론하고, 스윕은 점수 필터링으로 처리한다.
    # (threshold마다 재추론하면 10배 느리다)
    lowest = min(SWEEP)
    raw = []
    for fn, img_id in sorted(name2id.items()):
        p = img_dir / fn
        if not p.exists():
            sys.exit(f"[ERROR] 이미지 없음: {p}")
        W, H = id2wh[img_id]
        _, image = load_image(str(p))
        boxes, logits, _ = predict(model=model, image=image, caption=TEXT_PROMPT,
                                   box_threshold=lowest, text_threshold=a.text_threshold)
        if boxes.shape[0] == 0:
            continue
        scaled = boxes * torch.tensor([W, H, W, H], dtype=boxes.dtype)
        xyxy = box_convert(scaled, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        for (x1, y1, x2, y2), sc in zip(xyxy, logits.numpy()):
            raw.append({"image_id": int(img_id), "category_id": cat_id,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(sc)})

    (out / "predictions.json").write_text(json.dumps(raw))
    print(f"  예측 박스 {len(raw)}개 (thr>={lowest})")
    if not raw:
        print("  예측 0개 — 이것도 유의미한 results다")
        (out / "metrics.json").write_text(json.dumps(
            {"note": "예측 박스 0개", "kind": kind}, indent=2, ensure_ascii=False))
        return

    # ── threshold 스윕 ───────────────────────────────────────────────────────
    sweep_rows = []
    for thr in SWEEP:
        kept = [r for r in raw if r["score"] >= thr]
        TP = FP = FN = 0
        for img_id, gts in gts_by_img.items():
            preds = [r for r in kept if r["image_id"] == img_id]
            tp, fp, fn = match_greedy(preds, gts, IOU_MATCH)
            TP += tp; FP += fp; FN += fn
        P = TP / (TP + FP) if TP + FP else 0.0
        R = TP / (TP + FN) if TP + FN else 0.0
        F1 = 2 * P * R / (P + R) if P + R else 0.0

        if kept:
            ev = COCOeval(copy.deepcopy(coco_gt), coco_gt.loadRes(copy.deepcopy(kept)),
                          iouType="bbox")
            ev.evaluate(); ev.accumulate()
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                ev.summarize()
            ap50, mAP, ar100 = ev.stats[1], ev.stats[0], ev.stats[8]
        else:
            ap50 = mAP = ar100 = 0.0

        sweep_rows.append({"threshold": thr, "n_pred": len(kept),
                           "TP": TP, "FP": FP, "FN": FN,
                           "precision": round(P, 4), "recall": round(R, 4),
                           "f1": round(F1, 4), "ap50": round(ap50, 4),
                           "map": round(mAP, 4), "ar100": round(ar100, 4)})
        print(f"  thr {thr:.2f}: P {P:.3f} R {R:.3f} F1 {F1:.3f} "
              f"AP50 {ap50:.3f} mAP {mAP:.3f}")

    import csv as _csv
    with open(out / "threshold_sweep.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(sweep_rows[0]))
        w.writeheader(); w.writerows(sweep_rows)

    best = max(sweep_rows, key=lambda r: r["f1"])
    (out / "metrics.json").write_text(json.dumps({
        "kind": kind,
        "lora": a.lora,
        "split": a.split,
        "n_images": len(name2id),
        "n_gt_boxes": len(coco_gt.dataset["annotations"]),
        "best_threshold": best["threshold"],
        "at_best_threshold": best,
        "note": "threshold는 모델별 최적값에서 비교한다. 0.30 고정은 zero-shot에 불리하다.",
    }, indent=2, ensure_ascii=False))

    print(f"\n  best thr {best['threshold']:.2f} → "
          f"P {best['precision']:.3f} R {best['recall']:.3f} F1 {best['f1']:.3f} "
          f"AP50 {best['ap50']:.3f} mAP {best['map']:.3f}")


if __name__ == "__main__":
    main()
