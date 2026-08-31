#!/usr/bin/env python3
"""
run의 곡선 2개를 그린다. (RULES 4)

  curves/lr_schedule.png   steps.csv의 scheduler.get_last_lr() 실측값
  curves/loss_curve.png    train/val loss 오버레이 + val mAP (best epoch 선택 근거)

steps.csv / metrics.csv만 읽는다. 모델도 GPU도 필요 없어 학습 중에도 돌릴 수 있다.

  python train_code/make_curves.py --run results/FT-106_lr1e-4

RULES 4: "LR 스케줄 곡선은 실측값으로 그린다. 의도한 수식이 아니라 step마다
scheduler.get_last_lr()을 로깅한 값." — 계획대로 그리면 warmup이 실제로 동작했는지
확인할 수 없으므로 의미가 없다.

판정 로직은 넣지 않는다. 곡선 해석은 사람이 한다. (RULES 4)
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(p: Path):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(row, key):
    v = row.get(key, "")
    return float(v) if v not in ("", None) else float("nan")


def lr_schedule(steps, out: Path, cfg):
    xs = [int(r["step"]) for r in steps]
    ys = [f(r, "lr") for r in steps]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(xs, ys, lw=1.4, color="#1f77b4")

    peak_i = max(range(len(ys)), key=lambda i: ys[i])
    ax.plot(xs[peak_i], ys[peak_i], "o", color="#d62728", ms=6, zorder=5)
    ax.annotate(f"peak {ys[peak_i]:.2e}\nstep {xs[peak_i]}",
                (xs[peak_i], ys[peak_i]), textcoords="offset points",
                xytext=(10, -6), fontsize=9, color="#d62728")

    hp = cfg.get("hyperparams", {})
    spe = hp.get("steps_per_epoch")
    if spe:
        for e in range(1, cfg["hyperparams"]["epochs"] + 1):
            ax.axvline(e * spe, color="#dddddd", lw=0.5, zorder=0)

    ax.set_xlabel("step")
    ax.set_ylabel("learning rate (measured)")
    ax.set_title(f"{cfg.get('run_id','')} — LR schedule "
                 f"(scheduler.get_last_lr() measured, max_lr={hp.get('lr'):g}, "
                 f"pct_start={hp.get('pct_start')})")
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def loss_curve(rows, out: Path, cfg, best_epoch=None):
    ep = [int(r["epoch"]) for r in rows]
    tr = [f(r, "train_loss") for r in rows]
    va = [f(r, "val_loss") for r in rows]
    ap50 = [f(r, "val_ap50") for r in rows]
    mAP = [f(r, "val_map") for r in rows]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True,
                                 gridspec_kw={"height_ratios": [1.25, 1]})

    # RULES 4: train만으로는 학습 상태를 알 수 없다. 반드시 같이 그린다.
    a1.plot(ep, tr, "-o", ms=3, lw=1.5, color="#1f77b4", label="train loss")
    a1.plot(ep, va, "-o", ms=3, lw=1.5, color="#d62728", label="val loss")
    a1.set_ylabel("loss")
    a1.legend(loc="upper right")
    a1.grid(alpha=0.3)
    a1.set_title(f"{cfg.get('run_id','')} — lr {cfg['hyperparams']['lr']:g}, "
                 f"seed {cfg.get('seed')}, train {cfg['data']['train']['n_images']} imgs / "
                 f"val {cfg['data']['val']['n_images']} imgs")

    a2.plot(ep, mAP, "-o", ms=3, lw=1.6, color="#2ca02c", label="val mAP@.5:.95")
    a2.plot(ep, ap50, "-o", ms=3, lw=1.2, color="#9467bd", alpha=0.65,
            label="val AP50")
    a2.set_xlabel("epoch")
    a2.set_ylabel("val AP")
    a2.grid(alpha=0.3)
    a2.set_ylim(0, 1.02)

    if best_epoch:
        for ax in (a1, a2):
            ax.axvline(best_epoch, color="#ff7f0e", ls="--", lw=1.3, zorder=0)
        i = ep.index(best_epoch)
        a2.plot(best_epoch, mAP[i], "*", color="#ff7f0e", ms=16, zorder=5,
                label=f"best epoch {best_epoch} (val mAP {mAP[i]:.4f})")
    a2.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()

    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    curves = run / "curves"
    curves.mkdir(exist_ok=True)

    steps = read_csv(run / "steps.csv")
    rows = read_csv(run / "metrics.csv")
    if not rows:
        raise SystemExit("[ERROR] metrics.csv가 비어있다 — 학습이 epoch 1을 못 끝냈다")

    be = None
    bp = run / "best_epoch.json"
    if bp.exists():
        be = json.loads(bp.read_text(encoding="utf-8"))["best_epoch"]

    lr_schedule(steps, curves / "lr_schedule.png", cfg)
    loss_curve(rows, curves / "loss_curve.png", cfg, be)

    print(f"[curves] {run.name} | epoch {len(rows)}개 | step {len(steps)}개"
          + (f" | best epoch {be}" if be else " | best_epoch.json 없음 (학습 중)"))
    print(f"  {curves}/lr_schedule.png")
    print(f"  {curves}/loss_curve.png")


if __name__ == "__main__":
    main()
