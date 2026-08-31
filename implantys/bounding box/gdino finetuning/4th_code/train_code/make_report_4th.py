#!/usr/bin/env python3
"""
run 폴더의 REPORT.md를 자동 생성한다. (RULES 4 — 모든 run 동일 양식)

config.json / metrics.csv / steps.csv / best_epoch.json만 읽는다.

  python train_code/make_report.py --run results/FT-106_lr1e-4

곡선 해석은 넣지 않는다. RULES 4: "학습이 잘 됐는지에 대한 판단은 사람이 그래프를
보고 내린다. 코드는 그래프를 만들어주는 데까지만 한다. 자동 판정 로직을 넣지 않는다."
"""

import argparse
import csv
import json
from pathlib import Path


def read_csv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()

    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    rows = read_csv(run / "metrics.csv")
    steps = read_csv(run / "steps.csv")
    hp = cfg["hyperparams"]

    bp = run / "best_epoch.json"
    best = json.loads(bp.read_text(encoding="utf-8")) if bp.exists() else None

    lrs = [float(r["lr"]) for r in steps]
    L = []
    w = L.append

    w(f"# {cfg['run_id']}\n")
    # 4차: config에 'rounds'(라운드 목록) 대신 'set'(학습 세트) + 고정 val 정보가 들어간다
    w(f"생성 {cfg['created']} · seed {cfg['seed']} · 학습세트 train_{cfg['set']} "
      f"· val 고정 {Path(cfg['val_dir']).name}\n")

    w("\n## 결과\n")
    if best:
        s = best["selected"]
        w(f"**best epoch {best['best_epoch']}** — {best['criterion']}\n")
        w("\n| | val mAP@.5:.95 | val AP50 | val loss | train loss |")
        w("|---|---|---|---|---|")
        w(f"| epoch {best['best_epoch']} | **{s['val_map']:.4f}** | {s['val_ap50']:.4f} "
          f"| {s['val_loss']:.4f} | {s['train_loss']:.4f} |\n")
        w(f"\n체크포인트 `{Path(best['checkpoint']).name}`\n")
        w(f"\n> {best['note']}\n")
    else:
        w("**학습 미완료** — best_epoch.json이 아직 없다. "
          f"({len(rows)}/{hp['epochs']} epoch)\n")

    w("\n## 설정\n")
    w("| 항목 | 값 |")
    w("|---|---|")
    w(f"| learning rate (max) | {hp['lr']:g} |")
    w(f"| scheduler | {hp['scheduler']} (pct_start {hp['pct_start']}, "
      f"div_factor {hp['div_factor']}) |")
    w(f"| optimizer | {hp['optimizer']} |")
    w(f"| epochs | {hp['epochs']} (early stopping {'없음' if not hp['early_stopping'] else '사용'}) |")
    w(f"| batch size | {hp['batch_size']} |")
    w(f"| steps/epoch | {hp['steps_per_epoch']} |")
    w(f"| total steps | {hp['total_steps']} |")
    w(f"| loss 계수 | ce {hp['class_loss_coef']}, bbox {hp['bbox_loss_coef']}, "
      f"giou {hp['giou_loss_coef']} |")
    w(f"| grad clip | {hp['max_grad_norm']} |")
    w(f"| text prompt | `{hp['text_prompt']}` |")
    w(f"| LoRA 학습 파라미터 | {cfg['model']['trainable_params']:,} / "
      f"{cfg['model']['total_params']:,} "
      f"({100*cfg['model']['trainable_params']/cfg['model']['total_params']:.2f}%) |")

    w("\n## 데이터\n")
    w("| split | 이미지 | 박스 | split_hash | box_hash |")
    w("|---|---|---|---|---|")
    for k in ("train", "val"):
        d = cfg["data"][k]
        w(f"| {k} | {d['n_images']} | {d['n_boxes']} | `{d['split_hash']}` "
          f"| `{d['box_hash']}` |")
    w(f"\n코드 버전: repo `{cfg['code']['repo_git_rev']}` · "
      f"train.py `{cfg['code']['train_py_sha']}`\n")

    w("\n## LR 실측\n")
    if lrs:
        peak_i = max(range(len(lrs)), key=lambda i: lrs[i])
        w(f"- 시작 `{lrs[0]:.3e}` (max_lr / div_factor = "
          f"{hp['lr']:g} / {hp['div_factor']} = {hp['lr']/hp['div_factor']:.3e})")
        w(f"- 최고 `{lrs[peak_i]:.3e}` @ step {steps[peak_i]['step']} "
          f"(epoch {steps[peak_i]['epoch']})")
        w(f"- 마지막 `{lrs[-1]:.3e}` @ step {steps[-1]['step']}")
        w(f"\n`scheduler.get_last_lr()` 실측값. 의도한 수식이 아니다. (RULES 4)\n")

    w("\n## epoch별 지표\n")
    w("| epoch | train loss | val loss | val AP50 | val mAP@.5:.95 | LR |")
    w("|---|---|---|---|---|---|")
    for r in rows:
        mark = " **←best**" if best and int(r["epoch"]) == best["best_epoch"] else ""
        w(f"| {r['epoch']}{mark} | {float(r['train_loss']):.4f} "
          f"| {float(r['val_loss']):.4f} | {float(r['val_ap50']):.4f} "
          f"| {float(r['val_map']):.4f} | {float(r['lr']):.2e} |")

    w("\n## 곡선\n")
    for f_, cap in [("curves/lr_schedule.png", "LR schedule (실측)"),
                    ("curves/loss_curve.png", "train/val loss + val mAP")]:
        w(f"### {cap}\n")
        w(f"![{cap}]({f_})\n" if (run / f_).exists()
          else f"_{f_} 없음 — make_curves.py를 돌릴 것_\n")

    w("\n## 산출물\n")
    for f_, desc in [
        ("config.json", "하이퍼파라미터 + seed + 분할 해시 + 코드 버전"),
        ("metrics.csv", "epoch별 train/val loss, val AP50/mAP, LR"),
        ("steps.csv", "step별 LR 실측값"),
        ("best_epoch.json", "선택된 epoch + 근거"),
        ("curves/lr_schedule.png", "LR 곡선"),
        ("curves/loss_curve.png", "loss 곡선"),
        ("viz", "best epoch val 예측 + GT 오버레이"),
        ("attention", "'implant' ↔ 이미지 cross-attention"),
        ("data", "이 run이 쓴 CSV + 이미지"),
    ]:
        p = run / f_
        ok = p.exists() and (any(p.iterdir()) if p.is_dir() else p.stat().st_size > 0)
        w(f"- {'[o]' if ok else '[ ]'} `{f_}` — {desc}")

    w("\n---\n")
    w("곡선 해석(정상 / 과적합 / underfitting / 불안정)은 RULES 4의 표를 보고 "
      "사람이 판단한다. 이 문서는 자동 판정을 하지 않는다.\n")

    (run / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[report] {run / 'REPORT.md'} ({len(rows)} epoch"
          + (f", best {best['best_epoch']}" if best else ", 학습 중") + ")")


if __name__ == "__main__":
    main()
