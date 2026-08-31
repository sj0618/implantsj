#!/usr/bin/env python3
"""
test 48장 평가. zero-shot / FT-106 / FT-206을 같은 코드로 잰다.
--weights만 다르고 나머지는 전부 동일하다.

  # zero-shot (base 가중치만)
  python eval_code/evaluate.py --out results/00_zeroshot

  # fine-tuned (base + LoRA 체크포인트를 얹고 추론)
  python eval_code/evaluate.py --out results/01 --weights weights/checkpoint_epoch_28.pth

  # 추론 없이 저장된 predictions.json으로 지표만 다시 계산
  python eval_code/evaluate.py --out results/00_zeroshot --reuse-preds

지표는 두 종류로 나뉜다:

  AP50 / mAP  threshold와 무관한 지표. 예측을 점수순으로 세워 PR 곡선 전체를
              적분한다. 예측을 threshold로 잘라내고 계산하면 곡선 뒷부분이
              사라져 값이 낮아지므로, 자르지 않은 전체 예측으로 잰다.

  P / R / F1  threshold를 정해야만 나오는 지표. threshold 10개 전부에 대해
              계산해서 threshold_sweep.csv에 낸다. 어느 지점을 볼지는 고르지
              않는다 — 표를 그대로 낸다.

출력: config.json / predictions.json / threshold_sweep.csv / metrics.json
"""

import argparse
import contextlib
import copy
import csv
import io
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path("/workspace/Grounding-Dino-FineTuning")
BASE = REPO / "2nd"

TEXT_PROMPT = "implant"
SWEEP = [round(0.05 + 0.05 * i, 2) for i in range(10)]   # 0.05 ~ 0.50
IOU_MATCH = 0.5


def dump(path: Path, obj):
    """pod의 locale이 POSIX(ASCII)라 encoding을 명시하지 않으면 한글에서 깨진다."""
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_greedy(preds, gts, iou_thr):
    """score 내림차순으로 GT에 1:1 매칭. TP/FP/FN."""
    used = set()
    tp = 0
    for p in sorted(preds, key=lambda p: -p["score"]):
        best, bi = 0.0, -1
        for i, g in enumerate(gts):
            if i in used:
                continue
            v = iou_xywh(p["bbox"], g)
            if v > best:
                best, bi = v, i
        if best >= iou_thr:
            used.add(bi)
            tp += 1
    return tp, len(preds) - tp, len(gts) - tp


def coco_ap(coco_gt, preds):
    """AP50 / mAP@.5:.95 / AR100."""
    if not preds:
        return 0.0, 0.0, 0.0
    ev = COCOeval(copy.deepcopy(coco_gt),
                  coco_gt.loadRes(copy.deepcopy(preds)), iouType="bbox")
    ev.evaluate(); ev.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        ev.summarize()
    return ev.stats[1], ev.stats[0], ev.stats[8]      # ap50, map, ar100


def run_inference(weights, img_dir, name2id, id2wh, cat_id, text_threshold):
    import torch
    from torchvision.ops import box_convert
    from groundingdino.util.inference import load_model, load_image, predict
    from config import ModelConfig

    model_cfg = str(REPO / "groundingdino/config/GroundingDINO_SwinT_OGC.py")
    base_w = str(REPO / "weights/groundingdino_swint_ogc.pth")

    # lora_weigths — repo config.py의 오타 필드명. 그대로 맞춰야 한다.
    mc = ModelConfig(config_path=model_cfg, weights_path=base_w, lora_weigths=weights)
    model = load_model(mc, use_lora=bool(weights))
    model.eval()

    lowest = min(SWEEP)      # 한 번만 추론하고 스윕은 점수 필터로 처리
    raw = []
    total = len(name2id)
    for n, (fn, img_id) in enumerate(sorted(name2id.items()), 1):
        p = img_dir / fn
        if not p.exists():
            sys.exit(f"[ERROR] 이미지 없음: {p}")
        W, H = id2wh[img_id]
        _, image = load_image(str(p))
        boxes, logits, _ = predict(model=model, image=image, caption=TEXT_PROMPT,
                                   box_threshold=lowest, text_threshold=text_threshold)
        if n % 10 == 0 or n == total:
            print(f"  {n}/{total}", flush=True)
        if boxes.shape[0] == 0:
            continue
        boxes = boxes.cpu()
        scaled = boxes * torch.tensor([W, H, W, H], dtype=boxes.dtype)
        xyxy = box_convert(scaled, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        for (x1, y1, x2, y2), sc in zip(xyxy, logits.cpu().numpy()):
            raw.append({"image_id": int(img_id), "category_id": cat_id,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(sc)})
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="결과 폴더. 예: results/00_zeroshot")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-root", default=str(BASE / "dataset2"))
    ap.add_argument("--weights", default=None,
                    help="LoRA 체크포인트를 얹고 추론. 없으면 base만 = zero-shot")
    ap.add_argument("--text-threshold", type=float, default=0.20)
    ap.add_argument("--reuse-preds", action="store_true",
                    help="추론을 건너뛰고 --out의 predictions.json으로 지표만 재계산")
    a = ap.parse_args()

    global COCOeval
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    split_dir = Path(a.data_root) / a.split
    gt_json = split_dir / "annotations.json"
    img_dir = split_dir / "images"
    if not gt_json.exists():
        sys.exit(f"[ERROR] GT 없음: {gt_json}")

    coco_gt = COCO(str(gt_json))
    cat_id = coco_gt.dataset["categories"][0]["id"]
    name2id = {im["file_name"]: im["id"] for im in coco_gt.dataset["images"]}
    id2wh = {im["id"]: (im["width"], im["height"]) for im in coco_gt.dataset["images"]}
    gts_by_img = {i: [] for i in id2wh}
    for an in coco_gt.dataset["annotations"]:
        gts_by_img[an["image_id"]].append(an["bbox"])

    kind = "zero-shot" if not a.weights else f"FT ({Path(a.weights).name})"
    n_gt = len(coco_gt.dataset["annotations"])
    print(f"[eval] {kind} | {a.split} {len(name2id)}장 / {n_gt}박스")

    dump(out / "config.json", {
        "kind": kind, "weights": a.weights, "split": a.split,
        "data_root": str(a.data_root), "n_images": len(name2id), "n_gt_boxes": n_gt,
        "text_prompt": TEXT_PROMPT, "text_threshold": a.text_threshold,
        "sweep": SWEEP, "iou_match": IOU_MATCH,
    })

    pred_path = out / "predictions.json"
    if a.reuse_preds:
        if not pred_path.exists():
            sys.exit(f"[ERROR] --reuse-preds인데 없음: {pred_path}")
        raw = json.loads(pred_path.read_text(encoding="utf-8"))
        print(f"  predictions.json 재사용 — 박스 {len(raw)}개 (추론 건너뜀)")
    else:
        raw = run_inference(a.weights, img_dir, name2id, id2wh, cat_id, a.text_threshold)
        pred_path.write_text(json.dumps(raw), encoding="utf-8")
        print(f"\n  예측 박스 {len(raw)}개 (thr>={min(SWEEP)})")

    if not raw:
        print("  예측 0개")
        dump(out / "metrics.json", {"kind": kind, "note": "예측 박스 0개"})
        return

    # ── AP: 자르지 않은 전체 예측으로 한 번만 ────────────────────────────────
    ap50, mAP, ar100 = coco_ap(coco_gt, raw)

    # ── P/R/F1: threshold별로 전부 ───────────────────────────────────────────
    rows = []
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
        rows.append({"threshold": thr, "n_pred": len(kept), "TP": TP, "FP": FP, "FN": FN,
                     "precision": round(P, 4), "recall": round(R, 4), "f1": round(F1, 4)})

    with open(out / "threshold_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    dump(out / "metrics.json", {
        "kind": kind, "weights": a.weights, "split": a.split,
        "n_images": len(name2id), "n_gt_boxes": n_gt, "n_pred_raw": len(raw),
        "ap": {
            "ap50": round(ap50, 4), "map": round(mAP, 4), "ar100": round(ar100, 4),
            "basis": "자르지 않은 전체 예측",
        },
        "threshold_sweep": rows,
    })

    print(f"\n  AP50 {ap50:.3f} | mAP@.5:.95 {mAP:.3f} | AR100 {ar100:.3f}"
          f"   (전체 예측 {len(raw)}개 기준)")
    print(f"\n  {'thr':>5} {'n_pred':>7} {'TP':>4} {'FP':>5} {'FN':>4} "
          f"{'P':>6} {'R':>6} {'F1':>6}")
    for r in rows:
        print(f"  {r['threshold']:>5.2f} {r['n_pred']:>7} {r['TP']:>4} {r['FP']:>5} "
              f"{r['FN']:>4} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f}")
    print(f"\n완료: {out}")


if __name__ == "__main__":
    main()
