#!/usr/bin/env python3
"""
metrics.csv / steps.csv -> curves/loss_curve.png, curves/lr_schedule.png  (RULES.md 4번)

repo에 의존하지 않는다. CSV만 읽는다.

  python make_curves.py --run results/01/lr_1e-4
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(p: Path) -> list[dict]:
    if not p.exists():
        raise SystemExit(f"[ERROR] 없음: {p}  (학습이 metrics를 안 남겼다)")
    return list(csv.DictReader(open(p)))


def fnum(row, key):
    v = row.get(key, "")
    return float(v) if v not in ("", None) else None


def plot_loss(rows, out: Path):
    ep = [int(r["epoch"]) for r in rows]
    tr = [fnum(r, "train_loss") for r in rows]
    va = [fnum(r, "val_loss") for r in rows]
    vm = [fnum(r, "val_map") for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(ep, tr, "o-", color="#1f77b4", label="train loss", ms=3)
    if any(v is not None for v in va):
        ax1.plot(ep, va, "s-", color="#d62728", label="val loss", ms=3)
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
    ax1.grid(alpha=.3)

    # best epoch는 val mAP@.5:.95 기준 (RULES.md 5번)
    ax2 = ax1.twinx()
    if any(v is not None for v in vm):
        ax2.plot(ep, vm, "^--", color="#2ca02c", label="val mAP@.5:.95", ms=3, alpha=.8)
        ax2.set_ylabel("val mAP@.5:.95")
        best = max((v, e) for e, v in zip(ep, vm) if v is not None)
        ax1.axvline(best[1], color="#2ca02c", ls=":", alpha=.7)
        ax1.annotate(f"best epoch {best[1]}\nmAP {best[0]:.3f}",
                     xy=(best[1], ax1.get_ylim()[1]), ha="center", va="top",
                     fontsize=9, color="#2ca02c")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    ax1.set_title("train / val loss  (RULES 4: train loss alone tells you nothing)")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_lr(steps, out: Path):
    gs = [int(r["global_step"]) for r in steps]
    lr = [float(r["lr"]) for r in steps]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(gs, lr, "-", color="#9467bd", lw=1.2)
    ax.set_xlabel("global step"); ax.set_ylabel("learning rate")
    ax.grid(alpha=.3)
    ax.set_title("LR schedule (measured: scheduler.get_last_lr())")

    peak = max(range(len(lr)), key=lambda i: lr[i])
    ax.annotate(f"peak {lr[peak]:.2e} @ step {gs[peak]}",
                xy=(gs[peak], lr[peak]), xytext=(0.35, 0.9),
                textcoords="axes fraction", fontsize=9,
                arrowprops=dict(arrowstyle="->", alpha=.6))
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run 폴더. 예: results/01/lr_1e-4")
    a = ap.parse_args()

    run = Path(a.run)
    (run / "curves").mkdir(parents=True, exist_ok=True)

    rows = read_csv(run / "metrics.csv")
    plot_loss(rows, run / "curves" / "loss_curve.png")
    print(f"  curves/loss_curve.png  ({len(rows)} epochs)")

    steps = read_csv(run / "steps.csv")
    plot_lr(steps, run / "curves" / "lr_schedule.png")
    print(f"  curves/lr_schedule.png ({len(steps)} steps)")


if __name__ == "__main__":
    main()
