#!/usr/bin/env python3
"""
예측 박스 + GT 박스 오버레이 시각화. (RULES.md 4번)

predictions.json + annotations.json + 이미지만 읽는다. repo에 의존하지 않으므로
로컬에서도 돌릴 수 있다 — pod에서 예측만 회수해오면 시각화는 로컬에서 가능.

  python viz_pred.py --run results/00_zeroshot --data-root dataset2 --split test

색:
  초록 = GT
  파랑 = TP (GT와 IoU>=0.5로 매칭된 예측)
  빨강 = FP (매칭 안 된 예측)
  노랑 점선 = FN (놓친 GT)  ← 다음 라운드 전략의 핵심
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

GT_C, TP_C, FP_C, FN_C = "#2ca02c", "#1f77b4", "#d62728", "#ffcc00"


def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


def rect(d, box, color, width=3, dash=False):
    x, y, w, h = box
    if not dash:
        d.rectangle([x, y, x + w, y + h], outline=color, width=width)
        return
    step = 9
    for i in range(int(x), int(x + w), step * 2):
        d.line([i, y, min(i + step, x + w), y], fill=color, width=width)
        d.line([i, y + h, min(i + step, x + w), y + h], fill=color, width=width)
    for i in range(int(y), int(y + h), step * 2):
        d.line([x, i, x, min(i + step, y + h)], fill=color, width=width)
        d.line([x + w, i, x + w, min(i + step, y + h)], fill=color, width=width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-root", default="/workspace/dataset2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--threshold", type=float, default=None,
                    help="미지정 시 metrics.json의 best_threshold 사용")
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args()

    run = Path(a.run)
    out = run / "viz"; out.mkdir(parents=True, exist_ok=True)

    preds_p = run / "predictions.json"
    if not preds_p.exists():
        raise SystemExit(f"[ERROR] 없음: {preds_p} — eval_coco.py를 먼저 돌릴 것")
    preds = json.loads(preds_p.read_text())

    thr = a.threshold
    if thr is None:
        m = run / "metrics.json"
        thr = json.loads(m.read_text()).get("best_threshold", 0.3) if m.exists() else 0.3
    print(f"[viz] threshold {thr}")

    split_dir = Path(a.data_root) / a.split
    coco = json.loads((split_dir / "annotations.json").read_text())
    id2name = {i["id"]: i["file_name"] for i in coco["images"]}
    gts = {i["id"]: [] for i in coco["images"]}
    for an in coco["annotations"]:
        gts[an["image_id"]].append(an["bbox"])

    by_img = {i: [] for i in id2name}
    for p in preds:
        if p["score"] >= thr:
            by_img[p["image_id"]].append(p)

    n_fn_total = 0
    for img_id, fn in sorted(id2name.items(), key=lambda x: x[1]):
        src = split_dir / "images" / fn
        im = Image.open(src).convert("RGB")
        d = ImageDraw.Draw(im)

        g = gts[img_id]
        p = sorted(by_img[img_id], key=lambda r: -r["score"])

        for b in g:
            rect(d, b, GT_C, 3)

        matched = set()
        for pr in p:
            best, bi = 0.0, -1
            for i, gb in enumerate(g):
                if i in matched:
                    continue
                v = iou(pr["bbox"], gb)
                if v > best:
                    best, bi = v, i
            ok = best >= a.iou
            if ok:
                matched.add(bi)
            rect(d, pr["bbox"], TP_C if ok else FP_C, 2)
            x, y = pr["bbox"][0], pr["bbox"][1]
            d.text((x + 2, max(0, y - 11)), f"{pr['score']:.2f}",
                   fill=TP_C if ok else FP_C)

        fns = [gb for i, gb in enumerate(g) if i not in matched]
        for gb in fns:
            rect(d, gb, FN_C, 3, dash=True)
        n_fn_total += len(fns)

        tag = f"GT {len(g)} | pred {len(p)} | FN {len(fns)}"
        d.text((5, 5), tag, fill="#ffffff")
        suffix = "_FN" if fns else ""
        im.save(out / f"{Path(fn).stem}{suffix}.jpg", quality=90)

    n = len(id2name)
    print(f"  {n}장 저장 -> {out}")
    print(f"  FN 총 {n_fn_total}개 (파일명 _FN 접미사) — attention 분석 대상")


if __name__ == "__main__":
    main()
