#!/usr/bin/env python3
"""
저장된 predictions.json으로 AP50 / mAP@.5:.95 / **AP95(IoU=0.95)** 를 계산한다.
(4th_plan ① — 기존 지표에 AP95 열 추가)

COCOeval의 stats에는 AP@.5(stats[1])와 AP@.75(stats[2])만 있고 AP95는 없다.
eval["precision"] 배열에서 IoU=0.95 슬라이스를 직접 꺼내야 한다.
  precision.shape = (IoU 10, recall 101, class, area 4, maxDet 3)
  IoU 인덱스 9 = 0.95, area 0 = all, maxDet 2 = 100

재추론 없음 — GPU 불필요. 학습·평가와 독립적으로 언제든 다시 돌릴 수 있다.

  python add_ap95.py --results-root <...>/4th/results/test200 \
      --gt <...>/4th/dataset/test200/annotations.json \
      --runs zeroshot FT-206v2 FT-306v2 FT-506v2 FT-1000
"""

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def ap_at(ev: COCOeval, iou_idx: int | None) -> float:
    """iou_idx=None이면 .5:.95 평균, 아니면 해당 IoU만."""
    p = ev.eval["precision"]                    # (T, R, K, A, M)
    s = p if iou_idx is None else p[iou_idx:iou_idx + 1]
    s = s[:, :, :, 0, 2]                        # area=all, maxDet=100
    s = s[s > -1]
    return float(np.mean(s)) if s.size else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=None, help="기본 <results-root>/ap95.json")
    a = ap.parse_args()

    root = Path(a.results_root)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(a.gt)

    iou_thrs = np.linspace(0.5, 0.95, 10)
    i95 = int(np.argmin(np.abs(iou_thrs - 0.95)))
    i50 = int(np.argmin(np.abs(iou_thrs - 0.50)))
    i75 = int(np.argmin(np.abs(iou_thrs - 0.75)))

    out = {}
    print("%-11s %8s %8s %8s %8s  %s" % ("model", "AP50", "AP75", "mAP", "AP95", "n_pred"))
    for nm in a.runs:
        pf = root / nm / "predictions.json"
        if not pf.exists():
            print("%-11s (predictions.json 없음)" % nm)
            continue
        preds = json.loads(pf.read_text())
        if not preds:
            print("%-11s (예측 0개)" % nm)
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            dt = gt.loadRes([dict(p) for p in preds])
            ev = COCOeval(gt, dt, "bbox")
            ev.evaluate(); ev.accumulate(); ev.summarize()

        r = {"ap50": ap_at(ev, i50), "ap75": ap_at(ev, i75),
             "map": ap_at(ev, None), "ap95": ap_at(ev, i95),
             "n_pred": len(preds)}
        out[nm] = r
        print("%-11s %8.4f %8.4f %8.4f %8.4f  %d"
              % (nm, r["ap50"], r["ap75"], r["map"], r["ap95"], r["n_pred"]))

        # 각 run의 metrics.json에도 병합해 둔다 (원본 키는 건드리지 않고 추가만)
        mp = root / nm / "metrics.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            m["ap_extended"] = r
            mp.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")

    dest = Path(a.out) if a.out else root / "ap95.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n->", dest)
    print("* AP95는 IoU 0.95에서의 AP — 박스를 얼마나 정밀하게 그리는지를 본다.")


if __name__ == "__main__":
    main()
