#!/usr/bin/env python3
"""
metrics.csv -> best_epoch.json  (RULES.md 5번)

선택 지표는 val mAP@.5:.95 하나뿐이다. AP50은 쓰지 않는다.
1차 학습에서 AP50이 epoch_20·25 모두 1.000으로 포화되어 epoch를 구분하지 못했다.
(같은 조건에서 mAP@.5:.95는 0.877 vs 0.855로 구분됨)

동률(기본 0.005 이내)이면 더 이른 epoch를 고른다 — 과적합을 보수적으로 피한다.

  python select_best.py --run results/01/lr_1e-4
"""

import argparse
import csv
import json
from pathlib import Path

TIE_TOL = 0.005


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tie-tol", type=float, default=TIE_TOL,
                    help="이 차이 이내면 동률로 보고 더 이른 epoch 선택")
    a = ap.parse_args()

    run = Path(a.run)
    rows = list(csv.DictReader(open(run / "metrics.csv")))
    cand = [(int(r["epoch"]), float(r["val_map"]))
            for r in rows if r.get("val_map") not in ("", None)]
    if not cand:
        raise SystemExit("[ERROR] metrics.csv에 val_map이 없다. best epoch를 고를 수 없다.")

    top = max(v for _, v in cand)
    # 최고값과 tie-tol 이내인 것들 중 가장 이른 epoch
    best_ep, best_v = min(((e, v) for e, v in cand if top - v <= a.tie_tol),
                          key=lambda x: x[0])

    row = next(r for r in rows if int(r["epoch"]) == best_ep)
    tied = [e for e, v in cand if top - v <= a.tie_tol and e != best_ep]

    out = {
        "best_epoch": best_ep,
        "selection_metric": "val_map (mAP@.5:.95)",
        "selection_rule": (
            f"val mAP 최대값 {top:.4f} 기준 ±{a.tie_tol} 이내를 동률로 보고 최소 epoch 선택. "
            "AP50은 포화되어 사용하지 않음 (RULES.md 5번)."
        ),
        "val_map": best_v,
        "val_map_max": top,
        "tied_epochs": tied,
        "metrics_at_best": {k: row.get(k) for k in
                            ["train_loss", "val_loss", "val_ap50", "val_map"]},
        "checkpoint": f"checkpoint_epoch_{best_ep}.pth",
        "note": "test는 이 epoch를 확정한 뒤 딱 한 번만 연다 (RULES.md 2번).",
    }
    (run / "best_epoch.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"  best epoch = {best_ep}  (val mAP {best_v:.4f}, 최대 {top:.4f})")
    if tied:
        print(f"  동률 epoch {tied} 중 가장 이른 것 선택")


if __name__ == "__main__":
    main()
