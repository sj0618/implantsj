#!/usr/bin/env python3
"""
run 폴더 -> REPORT.md  (RULES.md 4번: 모든 run 동일 양식)

config.json / metrics.csv / best_epoch.json을 읽어 한 장으로 모으기만 한다.
repo에 의존하지 않는다.

⚠️ 곡선이 잘 됐는지 아닌지는 판정하지 않는다. 그건 사람이 그래프를 보고 한다.
   (RULES.md 4번의 train/val 해석표는 사람용 참고표이며 코드로 옮기지 않는다)

  python make_report.py --run results/01/lr_1e-4
"""

import argparse
import csv
import json
from pathlib import Path


def read_metrics(run: Path):
    p = run / "metrics.csv"
    return list(csv.DictReader(open(p))) if p.exists() else []


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()

    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
    best = json.loads((run / "best_epoch.json").read_text()) if (run / "best_epoch.json").exists() else {}
    rows = read_metrics(run)

    hp = cfg.get("hyperparameters", {})
    data = cfg.get("data", {})

    L = []
    L.append(f"# {cfg.get('run_id', run.name)}")
    L.append("")
    L.append(f"- 생성: {cfg.get('started_at', '-')}")
    L.append(f"- 라운드: {' + '.join(cfg.get('rounds', [])) or '-'}")
    L.append(f"- repo 버전: `{cfg.get('code_version', {}).get('gdino_repo', '-')}`")
    L.append("")

    L.append("## 하이퍼파라미터")
    L.append("")
    if hp:
        L.append("| 파라미터 | 값 |")
        L.append("|---|---|")
        for k, v in hp.items():
            L.append(f"| {k} | `{v}` |")
    else:
        L.append("_config.json에 하이퍼파라미터 없음_")
    L.append("")

    L.append("## 데이터")
    L.append("")
    if data:
        L.append("| split | 이미지 | 박스 | split_hash |")
        L.append("|---|---|---|---|")
        for k, v in data.items():
            L.append(f"| {k} | {v['n_images']} | {v['n_boxes']} | `{v['split_hash']}` |")
        L.append("")
        L.append("`split_hash`는 파일명 목록의 해시다. 한 장이라도 바뀌면 값이 바뀐다.")
    L.append("")

    L.append("## best epoch")
    L.append("")
    if best:
        L.append(f"**epoch {best['best_epoch']}** — val mAP@.5:.95 = {best['val_map']:.4f}")
        L.append("")
        L.append(f"- 선택 규칙: {best['selection_rule']}")
        if best.get("tied_epochs"):
            L.append(f"- 동률 epoch: {best['tied_epochs']} → 가장 이른 것 선택")
        L.append(f"- checkpoint: `{best['checkpoint']}`")
        m = best.get("metrics_at_best", {})
        if m:
            L.append("")
            L.append("| train_loss | val_loss | val_AP50 | val_mAP@.5:.95 |")
            L.append("|---|---|---|---|")
            L.append(f"| {m.get('train_loss','-')} | {m.get('val_loss','-')} | "
                     f"{m.get('val_ap50','-')} | {m.get('val_map','-')} |")
    else:
        L.append("_best_epoch.json 없음_")
    L.append("")

    L.append("## 학습 곡선")
    L.append("")
    L.append("![loss](curves/loss_curve.png)")
    L.append("")
    L.append("![lr](curves/lr_schedule.png)")
    L.append("")
    L.append("> LR 곡선은 `scheduler.get_last_lr()` 실측값이다. 의도한 수식이 아니라")
    L.append("> 실제로 warmup → decay가 동작했는지를 본다.")
    L.append("")

    if rows:
        L.append("## epoch별 지표")
        L.append("")
        L.append("| epoch | train_loss | val_loss | val_AP50 | val_mAP@.5:.95 | LR |")
        L.append("|---|---|---|---|---|---|")
        b = best.get("best_epoch")
        for r in rows:
            mark = " **←best**" if b is not None and int(r["epoch"]) == b else ""
            L.append(f"| {r['epoch']}{mark} | {r.get('train_loss','')} | {r.get('val_loss','')} "
                     f"| {r.get('val_ap50','')} | {r.get('val_map','')} | {r.get('lr_epoch_end','')} |")
        L.append("")

    n_viz = len(list((run / "viz").glob("*"))) if (run / "viz").is_dir() else 0
    n_att = len(list((run / "attention").glob("*"))) if (run / "attention").is_dir() else 0
    L.append("## 시각화")
    L.append("")
    L.append(f"- `viz/` — best epoch val 예측 + GT 오버레이 ({n_viz}개)")
    L.append(f"- `attention/` — attention map ({n_att}개)")
    L.append("")
    L.append("---")
    L.append("")
    L.append("_이 파일은 `make_report.py`가 자동 생성한다. 직접 수정하지 말 것._")

    (run / "REPORT.md").write_text("\n".join(L))
    print(f"  REPORT.md ({len(rows)} epochs)")


if __name__ == "__main__":
    main()
