#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test200 부트스트랩: 이미지 복원추출 B회 -> 각 모델 mAP@.5:.95 분포 + 모델간 차이 CI.
6개 모델(FT-106~FT-1000)에 매 iteration 동일 리샘플(paired) -> 차이의 불확실성을 올바르게 반영.
학습/재추론 없음. 저장된 predictions.json + test200 GT만 재분석.
"""
import json, io, contextlib, random
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

R = "/workspace/Grounding-Dino-FineTuning"
GT = json.load(open(R + "/3rd/dataset/test200/annotations.json"))
CATS = GT["categories"]
RUNS = ["FT-106", "FT-206", "FT-306", "FT-506", "FT-706", "FT-1000"]
B = 1000
random.seed(42)

gt_imgs = {im["id"]: im for im in GT["images"]}
gt_anns = {}
for a in GT["annotations"]:
    gt_anns.setdefault(a["image_id"], []).append(a)
img_ids = list(gt_imgs.keys())

dets_by_model = {}
for run in RUNS:
    preds = json.load(open(f"{R}/3rd/results/test200/{run}/predictions.json"))
    d = {}
    for p in preds:
        d.setdefault(p["image_id"], []).append(p)
    dets_by_model[run] = d


def map_for(sample, dets_by_img):
    images, anns, dets, aid = [], [], [], 1
    for new_id, old in enumerate(sample, 1):
        im = dict(gt_imgs[old]); im["id"] = new_id; images.append(im)
        for a in gt_anns.get(old, []):
            a2 = dict(a); a2["id"] = aid; a2["image_id"] = new_id; aid += 1; anns.append(a2)
        for p in dets_by_img.get(old, []):
            p2 = dict(p); p2["image_id"] = new_id; dets.append(p2)
    if not dets:
        return 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        g = COCO(); g.dataset = {"images": images, "annotations": anns, "categories": CATS}; g.createIndex()
        dt = g.loadRes(dets)
        E = COCOeval(g, dt, "bbox"); E.params.imgIds = [im["id"] for im in images]
        E.evaluate(); E.accumulate(); E.summarize()
    return float(E.stats[0])  # mAP@.5:.95


full = {run: map_for(img_ids, dets_by_model[run]) for run in RUNS}

samples = [[random.choice(img_ids) for _ in img_ids] for _ in range(B)]
boot = {run: [] for run in RUNS}
for s in samples:
    for run in RUNS:
        boot[run].append(map_for(s, dets_by_model[run]))
boot = {run: np.array(v) for run, v in boot.items()}

def ci(a):
    return np.percentile(a, 2.5), np.percentile(a, 97.5)

print("=== 모델별 test200 mAP@.5:.95 (점추정 + 부트스트랩 95%% CI, B=%d) ===" % B)
print("%-9s %8s   %s" % ("run", "mAP", "95% CI"))
for run in RUNS:
    lo, hi = ci(boot[run])
    print("%-9s %8.3f   [%.3f, %.3f]" % (run, full[run], lo, hi))

print("\n=== 모델간 차이 (paired, 95% CI, 0 포함이면 '노이즈') ===")
pairs = [("FT-206", "FT-106"), ("FT-306", "FT-206"), ("FT-506", "FT-306"),
         ("FT-706", "FT-506"), ("FT-1000", "FT-706"),
         ("FT-1000", "FT-306"), ("FT-1000", "FT-106")]
print("%-20s %8s   %-20s %8s %s" % ("비교", "Δ", "95% CI", "P(Δ>0)", "판정"))
for a, b in pairs:
    diff = boot[a] - boot[b]
    lo, hi = ci(diff)
    p_gt = float((diff > 0).mean())
    sig = "유의" if (lo > 0 or hi < 0) else "노이즈(0 포함)"
    print("%-20s %+8.3f   [%+.3f, %+.3f] %8.2f  %s" % (f"{a} - {b}", full[a]-full[b], lo, hi, p_gt, sig))
