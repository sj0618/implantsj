#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test200에서 zero-shot~FT-1000 (7모델)이 같은 이미지를 어떻게 다르게 보는지 정량화해,
'모델 간 예측 변화가 가장 큰' 이미지 상위 3개를 뽑는다.

- 고정 threshold 0.30, IoU 0.5 greedy 매칭 (viz_pred.py와 동일 기준)
- 이미지별 (TP,FP,FN)을 7모델에 대해 구하고,
  변화량 = 인접 모델 간 |ΔTP|+|ΔFP|+|ΔFN| 총합(total variation).
"""
import json
from pathlib import Path

R = Path("/workspace/Grounding-Dino-FineTuning")
GTP = R / "3rd/dataset/test200/annotations.json"
BASE = R / "3rd/results/test200"
RUNS = ["zeroshot", "FT-106", "FT-206", "FT-306", "FT-506", "FT-706", "FT-1000"]
THR, IOU = 0.30, 0.5


def iou_xywh(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih; u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


def tp_fp_fn(preds, gts):
    p = sorted(preds, key=lambda r: -r["score"])
    matched = set(); tp = 0
    for pr in p:
        best, bi = 0.0, -1
        for i, gb in enumerate(gts):
            if i in matched:
                continue
            v = iou_xywh(pr["bbox"], gb)
            if v > best:
                best, bi = v, i
        if best >= IOU:
            matched.add(bi); tp += 1
    return tp, len(p) - tp, len(gts) - len(matched)


gt = json.load(open(GTP))
id2name = {i["id"]: i["file_name"] for i in gt["images"]}
gts = {i["id"]: [] for i in gt["images"]}
for a in gt["annotations"]:
    gts[a["image_id"]].append(a["bbox"])

# run -> img_id -> [preds>=thr]
run_pred = {}
for run in RUNS:
    pp = BASE / run / "predictions.json"
    by = {i: [] for i in id2name}
    if pp.exists():
        for p in json.load(open(pp)):
            if p["score"] >= THR:
                by[p["image_id"]].append(p)
    run_pred[run] = by

rows = []
for iid, fname in id2name.items():
    seq = [tp_fp_fn(run_pred[run][iid], gts[iid]) for run in RUNS]
    tv = 0
    for k in range(1, len(seq)):
        tv += sum(abs(seq[k][j] - seq[k-1][j]) for j in range(3))
    rng = sum(max(s[j] for s in seq) - min(s[j] for s in seq) for j in range(3))
    rows.append((tv, rng, len(gts[iid]), fname, seq))

rows.sort(key=lambda r: (-r[0], -r[1]))

print(f"기준: thr {THR}, IoU {IOU}, 모델순 {RUNS}")
print(f"변화량(TV) = 인접 모델 간 |ΔTP|+|ΔFP|+|ΔFN| 합\n")
print("=== 변화 가장 큰 이미지 TOP 3 ===")
for rank, (tv, rng, ng, fname, seq) in enumerate(rows[:3], 1):
    print(f"\n[{rank}] {fname}  (GT {ng}박스, TV={tv}, range합={rng})")
    print("     %-9s %3s %3s %3s" % ("model", "TP", "FP", "FN"))
    for run, (tp, fp, fn) in zip(RUNS, seq):
        print("     %-9s %3d %3d %3d" % (run, tp, fp, fn))

print("\n--- 참고: 4~8위 ---")
for tv, rng, ng, fname, seq in rows[3:8]:
    print(f"   {fname}  TV={tv} range={rng} GT={ng}")
