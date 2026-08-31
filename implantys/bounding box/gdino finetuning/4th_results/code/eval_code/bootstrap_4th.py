#!/usr/bin/env python3
"""
4차 test200 부트스트랩 — mAP@.5:.95 와 AP95 둘 다.
3rd/bootstrap_test200.py의 4차판 (모델 목록·경로·AP95 추가만 다름).

이미지 단위 복원추출 B회, 매 iteration 모든 모델에 **동일 리샘플(paired)** 적용해
모델 간 차이의 불확실성을 올바르게 반영한다. 학습·재추론 없음(저장된 predictions.json만).

  python bootstrap_4th.py --results-root <...>/4th/results/test200 \
      --gt <...>/4th/dataset/test200/annotations.json -B 1000
"""

import argparse
import contextlib
import io
import json
import random
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

RUNS = ["FT-206v2", "FT-306v2", "FT-506v2", "FT-1000"]
PAIRS = [("FT-306v2", "FT-206v2"), ("FT-506v2", "FT-306v2"),
         ("FT-1000", "FT-506v2"), ("FT-1000", "FT-206v2")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("-B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--runs", nargs="+", default=RUNS)
    a = ap.parse_args()

    random.seed(a.seed)
    root = Path(a.results_root)
    GT = json.loads(Path(a.gt).read_text())
    CATS = GT["categories"]

    gt_imgs = {im["id"]: im for im in GT["images"]}
    gt_anns = {}
    for an in GT["annotations"]:
        gt_anns.setdefault(an["image_id"], []).append(an)
    img_ids = list(gt_imgs)

    iou_thrs = np.linspace(0.5, 0.95, 10)
    I95 = int(np.argmin(np.abs(iou_thrs - 0.95)))

    dets = {}
    runs = [r for r in a.runs if (root / r / "predictions.json").exists()]
    for r in runs:
        d = {}
        for p in json.loads((root / r / "predictions.json").read_text()):
            d.setdefault(p["image_id"], []).append(p)
        dets[r] = d
    if len(runs) < len(a.runs):
        print("[warn] predictions 없는 run 제외:", set(a.runs) - set(runs))

    def eval_sample(sample, dets_by_img):
        """리샘플된 이미지 집합에서 (mAP, AP95) 반환."""
        images, anns, dd, aid = [], [], [], 1
        for new_id, old in enumerate(sample, 1):
            im = dict(gt_imgs[old]); im["id"] = new_id; images.append(im)
            for an in gt_anns.get(old, []):
                a2 = dict(an); a2["id"] = aid; a2["image_id"] = new_id
                aid += 1; anns.append(a2)
            for p in dets_by_img.get(old, []):
                p2 = dict(p); p2["image_id"] = new_id; dd.append(p2)
        if not dd:
            return 0.0, 0.0
        with contextlib.redirect_stdout(io.StringIO()):
            g = COCO()
            g.dataset = {"images": images, "annotations": anns, "categories": CATS}
            g.createIndex()
            dt = g.loadRes(dd)
            E = COCOeval(g, dt, "bbox")
            E.params.imgIds = [im["id"] for im in images]
            E.evaluate(); E.accumulate(); E.summarize()
        pr = E.eval["precision"]
        m = pr[:, :, :, 0, 2]; m = m[m > -1]
        n = pr[I95:I95 + 1, :, :, 0, 2]; n = n[n > -1]
        return (float(m.mean()) if m.size else 0.0,
                float(n.mean()) if n.size else 0.0)

    full = {r: eval_sample(img_ids, dets[r]) for r in runs}

    samples = [[random.choice(img_ids) for _ in img_ids] for _ in range(a.B)]
    boot = {r: {"map": [], "ap95": []} for r in runs}
    for k, s in enumerate(samples, 1):
        for r in runs:
            m, n = eval_sample(s, dets[r])
            boot[r]["map"].append(m); boot[r]["ap95"].append(n)
        if k % 100 == 0:
            print(f"  bootstrap {k}/{a.B}", flush=True)
    for r in runs:
        for k in boot[r]:
            boot[r][k] = np.array(boot[r][k])

    def ci(x):
        return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

    result = {"B": a.B, "seed": a.seed, "point": {}, "ci": {}, "diff": {}}
    for metric, idx in [("map", 0), ("ap95", 1)]:
        print(f"\n=== 모델별 {metric} (점추정 + 95% CI, B={a.B}) ===")
        print("%-10s %8s   %s" % ("run", metric, "95% CI"))
        for r in runs:
            lo, hi = ci(boot[r][metric])
            print("%-10s %8.4f   [%.4f, %.4f]" % (r, full[r][idx], lo, hi))
            result["point"].setdefault(r, {})[metric] = full[r][idx]
            result["ci"].setdefault(r, {})[metric] = [lo, hi]

        print(f"\n=== 모델간 차이 ({metric}, paired, 0 포함이면 노이즈) ===")
        print("%-24s %9s   %-22s %8s %s" % ("비교", "Δ", "95% CI", "P(Δ>0)", "판정"))
        for x, y in PAIRS:
            if x not in runs or y not in runs:
                continue
            diff = boot[x][metric] - boot[y][metric]
            lo, hi = ci(diff)
            p = float((diff > 0).mean())
            sig = "유의" if (lo > 0 or hi < 0) else "노이즈(0 포함)"
            d = full[x][idx] - full[y][idx]
            print("%-24s %+9.4f   [%+.4f, %+.4f] %8.2f  %s"
                  % (f"{x} - {y}", d, lo, hi, p, sig))
            result["diff"].setdefault(f"{x}-{y}", {})[metric] = {
                "delta": d, "ci": [lo, hi], "p_gt0": p, "significant": bool(lo > 0 or hi < 0)}

    (root / "bootstrap.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n->", root / "bootstrap.json")


if __name__ == "__main__":
    main()
