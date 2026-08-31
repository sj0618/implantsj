#!/usr/bin/env python3
"""
Attention 분석용 케이스 선별. (4th_plan ④)

기준 모델(FT-1000)의 운영 threshold에서 FN / FP / 대표 TP 이미지를 고른다.
  FN : GT 박스인데 못 잡은 것 -> 전부 (RULES 7의 핵심 분석 대상)
  FP : 잡았는데 GT가 없는 것 -> 상위 N (모델이 무엇을 임플란트로 착각하는가)
  TP : 전부 맞히고 점수도 높은 이미지 -> 상위 N (hook 검증용 대조군)

출력: cases.json  {fn: [...], fp: [...], tp: [...], meta: {...}}

  python select_cases_4th.py --results-root <...>/4th/results/test200 \
      --gt <...>/4th/dataset/test200/annotations.json --run FT-1000 --thr 0.30
"""

import argparse
import json
from pathlib import Path

IOU_THR = 0.5


def iou_xywh(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--run", default="FT-1000")
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--n-fp", type=int, default=6)
    ap.add_argument("--n-tp", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = Path(a.results_root)
    gt = json.loads(Path(a.gt).read_text())
    id2name = {im["id"]: im["file_name"] for im in gt["images"]}
    gts = {}
    for an in gt["annotations"]:
        gts.setdefault(an["image_id"], []).append(an["bbox"])

    preds_all = json.loads((root / a.run / "predictions.json").read_text())
    preds = {}
    for p in preds_all:
        if p["score"] >= a.thr:
            preds.setdefault(p["image_id"], []).append(p)

    fn_imgs, fp_imgs, tp_imgs = {}, {}, {}
    n_fn = n_fp = n_tp = 0
    for img_id, name in id2name.items():
        g = gts.get(img_id, [])
        ps = sorted(preds.get(img_id, []), key=lambda r: -r["score"])
        used, matched_scores = set(), []
        for p in ps:
            best, bi = 0.0, -1
            for i, gb in enumerate(g):
                if i in used:
                    continue
                v = iou_xywh(p["bbox"], gb)
                if v > best:
                    best, bi = v, i
            if best >= IOU_THR:
                used.add(bi); matched_scores.append(p["score"])
        miss = len(g) - len(used)
        extra = len(ps) - len(used)
        n_fn += miss; n_fp += extra; n_tp += len(used)

        if miss > 0:
            fn_imgs[name] = {"n_gt": len(g), "n_missed": miss, "n_pred": len(ps)}
        if extra > 0:
            fp_imgs[name] = {"n_gt": len(g), "n_fp": extra,
                             "top_unmatched": max((p["score"] for p in ps), default=0)}
        if g and miss == 0 and extra == 0 and matched_scores:
            tp_imgs[name] = {"n_gt": len(g), "min_score": min(matched_scores)}

    fn = sorted(fn_imgs, key=lambda n: -fn_imgs[n]["n_missed"])
    fp = sorted(fp_imgs, key=lambda n: -fp_imgs[n]["n_fp"])[:a.n_fp]
    tp = sorted(tp_imgs, key=lambda n: -tp_imgs[n]["min_score"])[:a.n_tp]

    out = {"meta": {"run": a.run, "threshold": a.thr, "iou": IOU_THR,
                    "totals": {"TP": n_tp, "FP": n_fp, "FN": n_fn},
                    "n_fn_images": len(fn_imgs), "n_fp_images": len(fp_imgs)},
           "fn": fn, "fp": fp, "tp": tp,
           "detail": {"fn": fn_imgs, "fp": {k: fp_imgs[k] for k in fp},
                      "tp": {k: tp_imgs[k] for k in tp}}}

    dest = Path(a.out) if a.out else root.parent / "attention" / "cases.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[{a.run} @ thr {a.thr}] TP {n_tp} / FP {n_fp} / FN {n_fn}")
    print(f"  FN 이미지 {len(fn_imgs)}장 (전부 선택)")
    for n in fn:
        d = fn_imgs[n]
        print(f"    {n}  GT {d['n_gt']} 중 {d['n_missed']}개 놓침 (예측 {d['n_pred']})")
    print(f"  FP 이미지 {len(fp_imgs)}장 중 상위 {len(fp)}장 선택")
    print(f"  TP 대조군 {len(tp)}장 선택")
    print("->", dest)
    print("\n[패널 대상 총계]", len(set(fn + fp + tp)), "장")


if __name__ == "__main__":
    main()
