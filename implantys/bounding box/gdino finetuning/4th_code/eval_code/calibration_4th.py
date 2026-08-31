#!/usr/bin/env python3
"""
4차 Calibration — test200. (4th_plan ③)

저장된 predictions.json + GT만 사용. 재추론 없음(GPU 불필요).

만드는 것:
  1. reliability_test200.png   score 구간별 실제 정답률(precision) + 구간별 예측 수
                               "confidence가 높으면 정말 맞는가"
  2. pr_curve_test200.png      threshold별 precision·recall 동시 표시
                               calibration은 예측된 박스만 다루므로 recall이 안 보인다.
                               놓친 것까지 같이 보려면 이 그림이 필요하다.
  3. calibration.csv           위 두 그림의 원자료 (score 구간별 / threshold별)

TP/FP 판정: 이미지별로 score 내림차순 greedy 매칭, IoU >= 0.5 (evaluate.py와 동일 기준).
매칭이 score 내림차순이므로 낮은 점수 예측을 잘라내도 높은 점수 예측의 판정은 변하지 않는다
= threshold별 지표를 사전 판정 결과에서 그대로 집계해도 정확하다.

  python calibration_4th.py --results-root <...>/4th/results/test200 \
      --gt <...>/4th/dataset/test200/annotations.json \
      --runs zeroshot FT-206v2 FT-306v2 FT-506v2 FT-1000
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IOU_THR = 0.5
BIN_W = 0.05


def iou_xywh(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


def label_preds(preds, gts_by_img):
    """예측마다 (score, is_tp) 판정. score 내림차순 greedy, IoU>=0.5."""
    out = []
    by_img = {}
    for p in preds:
        by_img.setdefault(p["image_id"], []).append(p)
    for img_id, ps in by_img.items():
        gts = gts_by_img.get(img_id, [])
        used = set()
        for p in sorted(ps, key=lambda r: -r["score"]):
            best, bi = 0.0, -1
            for i, g in enumerate(gts):
                if i in used:
                    continue
                v = iou_xywh(p["bbox"], g)
                if v > best:
                    best, bi = v, i
            hit = best >= IOU_THR
            if hit:
                used.add(bi)
            out.append((float(p["score"]), bool(hit)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=None, help="기본 <results-root>/../calibration")
    a = ap.parse_args()

    root = Path(a.results_root)
    out = Path(a.out) if a.out else root.parent / "calibration"
    out.mkdir(parents=True, exist_ok=True)

    gt = json.loads(Path(a.gt).read_text())
    gts_by_img = {}
    for an in gt["annotations"]:
        gts_by_img.setdefault(an["image_id"], []).append(an["bbox"])
    n_gt = len(gt["annotations"])
    print(f"GT: {len(gt['images'])}장 / {n_gt}박스")

    runs = [r for r in a.runs if (root / r / "predictions.json").exists()]
    labeled = {}
    for nm in runs:
        preds = json.loads((root / nm / "predictions.json").read_text())
        labeled[nm] = label_preds(preds, gts_by_img)
        print(f"  {nm}: 예측 {len(labeled[nm])}개")

    edges = np.arange(0.05, 1.0 + 1e-9, BIN_W)
    thrs = np.round(np.arange(0.05, 0.95 + 1e-9, 0.025), 3)

    rows_bin, rows_thr = [], []
    for nm in runs:
        arr = labeled[nm]
        sc = np.array([s for s, _ in arr])
        hit = np.array([h for _, h in arr])

        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (sc >= lo) & (sc < hi)
            n, tp = int(m.sum()), int(hit[m].sum())
            rows_bin.append({"run": nm, "bin_lo": round(float(lo), 3),
                             "bin_hi": round(float(hi), 3), "n": n, "TP": tp,
                             "FP": n - tp,
                             "precision": round(tp / n, 4) if n else ""})

        for t in thrs:
            m = sc >= t
            tp, fp = int(hit[m].sum()), int((~hit[m]).sum())
            fn = n_gt - tp
            P = tp / (tp + fp) if tp + fp else 0.0
            R = tp / (tp + fn) if tp + fn else 0.0
            F1 = 2 * P * R / (P + R) if P + R else 0.0
            rows_thr.append({"run": nm, "threshold": float(t), "TP": tp, "FP": fp,
                             "FN": fn, "precision": round(P, 4),
                             "recall": round(R, 4), "f1": round(F1, 4)})

    with open(out / "calibration.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["# score 구간별 (reliability diagram 원자료)"])
        w.writerow(list(rows_bin[0]))
        for r in rows_bin:
            w.writerow(list(r.values()))
        w.writerow([])
        w.writerow(["# threshold별 (P/R 곡선 원자료)"])
        w.writerow(list(rows_thr[0]))
        for r in rows_thr:
            w.writerow(list(r.values()))
    print("->", out / "calibration.csv")

    # ── 1. reliability diagram ──────────────────────────────────────────────
    ncol = len(runs)
    fig, axes = plt.subplots(2, ncol, figsize=(3.4 * ncol, 6.2),
                             gridspec_kw={"height_ratios": [3, 1.4]}, squeeze=False)
    centers = (edges[:-1] + edges[1:]) / 2
    for j, nm in enumerate(runs):
        br = [r for r in rows_bin if r["run"] == nm]
        prec = [r["precision"] if r["precision"] != "" else np.nan for r in br]
        cnt = [r["n"] for r in br]

        ax = axes[0][j]
        ax.plot([0, 1], [0, 1], "--", color="#999", lw=1, label="perfect calib.")
        ax.bar(centers, prec, width=BIN_W * 0.9, color="#4C78A8", edgecolor="white")
        ax.set_title(nm, fontsize=11)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.set_xlabel("score"); ax.grid(alpha=0.25)
        if j == 0:
            ax.set_ylabel("actual precision")
            ax.legend(fontsize=8, loc="lower right")

        ax2 = axes[1][j]
        ax2.bar(centers, cnt, width=BIN_W * 0.9, color="#B0B0B0", edgecolor="white")
        ax2.set_xlim(0, 1); ax2.set_yscale("log")
        ax2.set_xlabel("score"); ax2.grid(alpha=0.25)
        if j == 0:
            ax2.set_ylabel("n predictions (log)")
    fig.suptitle("Reliability diagram — test200 (IoU 0.5)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "reliability_test200.png", dpi=140)
    plt.close(fig)
    print("->", out / "reliability_test200.png")

    # ── 2. threshold - P/R 곡선 ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, ncol, figsize=(3.4 * ncol, 3.6), squeeze=False)
    for j, nm in enumerate(runs):
        tr = [r for r in rows_thr if r["run"] == nm]
        t = [r["threshold"] for r in tr]
        ax = axes[0][j]
        ax.plot(t, [r["precision"] for r in tr], color="#4C78A8", lw=2, label="precision")
        ax.plot(t, [r["recall"] for r in tr], color="#E45756", lw=2, label="recall")
        ax.plot(t, [r["f1"] for r in tr], color="#54A24B", lw=1.2, ls="--", label="F1")
        ax.set_title(nm, fontsize=11)
        ax.set_xlabel("threshold"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.25)
        if j == 0:
            ax.set_ylabel("value"); ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Precision / Recall vs threshold — test200", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out / "pr_curve_test200.png", dpi=140)
    plt.close(fig)
    print("->", out / "pr_curve_test200.png")


if __name__ == "__main__":
    main()
