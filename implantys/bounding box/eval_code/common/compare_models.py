#!/usr/bin/env python3
"""
zero-shot / FT-106 / FT-206의 test results를 모아 최종 비교표를 만든다.

전부 동일한 test 48장 기준. 각 모델의 '최적 threshold'에서 비교한다
(0.30 하나로 고정하면 zero-shot이 부당하게 낮게 나온다).

repo에 의존하지 않는다. 각 run의 metrics.json / threshold_sweep.csv만 읽는다.

  python compare_models.py --result-root results --out results/COMPARISON.md
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_runs(root: Path) -> list[tuple[str, str, Path]]:
    """(표시명, 학습데이터, run 경로). test results가 있는 것만."""
    out = []
    zs = root / "00_zeroshot"
    if (zs / "metrics.json").exists():
        out.append(("zero-shot", "-", zs))
    for rnd, label, n in [("01", "FT-106", "106장"), ("02", "FT-206", "206장")]:
        d = root / rnd
        if not d.is_dir():
            continue
        for lr in sorted(d.iterdir()):
            if (lr / "metrics.json").exists():
                out.append((f"{label} ({lr.name})", n, lr))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-root", default="results")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = Path(a.result_root)
    runs = find_runs(root)
    if not runs:
        raise SystemExit(f"[ERROR] {root} 아래에 test results(metrics.json)가 없다")

    out_p = Path(a.out) if a.out else root / "COMPARISON.md"

    L = ["# 최종 비교 (test 48장 / 81박스, 전부 동일 기준)", ""]
    L.append("| 모델 | 학습 데이터 | AP50 | mAP@.5:.95 | P | R | F1 | best thr |")
    L.append("|---|---|---|---|---|---|---|---|")

    fig, ax = plt.subplots(figsize=(7, 5))
    rows_for_check = []

    for name, data, p in runs:
        m = json.loads((p / "metrics.json").read_text())
        b = m.get("at_best_threshold")
        if not b:
            L.append(f"| {name} | {data} | — | — | — | — | — | — | _{m.get('note','results 없음')}_")
            continue
        L.append(f"| {name} | {data} | {b['ap50']:.3f} | {b['map']:.3f} | "
                 f"{b['precision']:.3f} | {b['recall']:.3f} | {b['f1']:.3f} | "
                 f"{b['threshold']:.2f} |")
        rows_for_check.append((name, b))

        sweep = p / "threshold_sweep.csv"
        if sweep.exists():
            rs = list(csv.DictReader(open(sweep)))
            ax.plot([float(r["recall"]) for r in rs],
                    [float(r["precision"]) for r in rs],
                    "o-", ms=3, label=name)

    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(alpha=.3)
    ax.set_title("PR curve (threshold 0.05~0.50), test n=48")
    ax.legend(fontsize=8)
    fig.tight_layout()
    pr = out_p.parent / "pr_curve.png"
    fig.savefig(pr, dpi=130); plt.close(fig)

    L += ["", f"![PR](./{pr.name})", ""]
    L += ["## 읽는 법", ""]
    L.append("- threshold는 **모델별 최적값**에서 비교했다. 0.30 하나로 고정하면 "
             "zero-shot이 부당하게 낮게 나온다.")
    L.append("- best epoch는 val로만 골랐고, test는 확정 후 한 번만 열었다. (RULES 2·5)")
    L.append("")
    L.append("## 한계")
    L.append("")
    L.append("- **nobel 브랜드는 test에 0장**이라 성능을 측정할 수 없다. "
             "원본 12장이 전부 train에 있다.")
    L.append("- test 81박스 기준이라 recall 추정의 95% 신뢰구간이 대략 ±0.10이다. "
             "0.64 → 0.70 정도의 차이는 통계적으로 우연과 구분되지 않는다. "
             "큰 폭의 개선만 유의미하게 주장할 수 있다.")

    out_p.write_text("\n".join(x for x in L if isinstance(x, str)))
    print(f"  {out_p}")
    print(f"  {pr}")
    for name, b in rows_for_check:
        print(f"    {name:24} F1 {b['f1']:.3f}  R {b['recall']:.3f}  mAP {b['map']:.3f}")


if __name__ == "__main__":
    main()
