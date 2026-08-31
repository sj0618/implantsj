#!/usr/bin/env python3
"""
run 폴더가 규칙 4의 산출물을 전부 갖췄는지 검사한다. 하나라도 빠지면 exit 1.

규칙 4의 "하나라도 빠지면 실패로 간주한다"를 실제로 강제하는 곳.
사람이 기억해서 챙기는 구조면 언젠가 빠진다.

  python verify_run.py --run results/01/lr_1e-4
  python verify_run.py --run results/00_zeroshot --kind eval
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# (경로, 설명, 디렉토리면 최소 파일 수)
TRAIN_REQUIRED = [
    ("config.json", "하이퍼파라미터 + 데이터 해시", None),
    ("metrics.csv", "epoch별 지표", None),
    ("steps.csv", "step별 LR 실측값", None),
    ("best_epoch.json", "선택된 epoch + 근거", None),
    ("curves/lr_schedule.png", "LR 스케줄 곡선", None),
    ("curves/loss_curve.png", "train/val loss 곡선", None),
    ("viz", "best epoch val 예측 시각화", 1),
    ("attention", "attention map", 1),
    ("REPORT.md", "자동 요약", None),
]

EVAL_REQUIRED = [
    ("config.json", "평가 설정", None),
    ("threshold_sweep.csv", "threshold 스윕 0.05~0.50", None),
    ("metrics.json", "AP50 / mAP / P / R / F1 @ best thr", None),
    ("viz", "예측 + GT 오버레이", 1),
    ("attention", "attention map", 1),
    ("REPORT.md", "자동 요약", None),
]


def check(run: Path, required) -> list[str]:
    problems = []
    for rel, desc, min_files in required:
        p = run / rel
        if not p.exists():
            problems.append(f"없음      {rel:28} — {desc}")
            continue
        if min_files is not None:
            n = len([f for f in p.iterdir() if f.is_file() and not f.name.startswith(".")])
            if n < min_files:
                problems.append(f"비어있음  {rel:28} — {desc} (파일 {n}개)")
        elif p.is_file() and p.stat().st_size == 0:
            problems.append(f"0바이트   {rel:28} — {desc}")
    return problems


def check_contents(run: Path, kind: str) -> list[str]:
    """존재만이 아니라 내용이 규칙을 지키는지."""
    problems = []

    cfg_p = run / "config.json"
    if cfg_p.exists():
        cfg = json.loads(cfg_p.read_text())
        if kind == "train":
            data = cfg.get("data", {})
            if not data:
                problems.append("config.json에 데이터 해시가 없다 (RULES 4: 반드시 박을 것)")
            elif not any("split_hash" in v for v in data.values()):
                problems.append("config.json의 data에 split_hash가 없다")

    m = run / "metrics.csv"
    if kind == "train" and m.exists():
        rows = list(csv.DictReader(open(m)))
        if not rows:
            problems.append("metrics.csv가 헤더뿐이다")
        else:
            if all(r.get("val_map", "") == "" for r in rows):
                problems.append("metrics.csv에 val_map이 전부 비어있다 "
                                "(RULES 5: best epoch를 고를 수 없다)")
            if all(r.get("val_loss", "") == "" for r in rows):
                problems.append("metrics.csv에 val_loss가 전부 비어있다 "
                                "(RULES 4: train만으로는 학습 상태를 알 수 없다)")

    s = run / "steps.csv"
    if kind == "train" and s.exists():
        rows = list(csv.DictReader(open(s)))
        if len(rows) < 2:
            problems.append("steps.csv에 LR 기록이 사실상 없다")
        else:
            lrs = [float(r["lr"]) for r in rows]
            if len(set(lrs)) == 1:
                problems.append(f"steps.csv의 LR이 전 구간 상수({lrs[0]:.2e})다 — "
                                "warmup/decay가 동작하지 않았다")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--kind", choices=["train", "eval"], default="train")
    a = ap.parse_args()

    run = Path(a.run)
    if not run.is_dir():
        sys.exit(f"[FAIL] run 폴더가 없다: {run}")

    required = TRAIN_REQUIRED if a.kind == "train" else EVAL_REQUIRED
    problems = check(run, required) + check_contents(run, a.kind)

    print(f"[verify:{a.kind}] {run}")
    if problems:
        print(f"\n규칙 4 위반 {len(problems)}건:")
        for p in problems:
            print(f"  ✗ {p}")
        print("\n[FAIL] 산출물이 불완전하다. 이 run은 기록으로 인정되지 않는다.")
        sys.exit(1)

    print(f"  ✓ 필수 산출물 {len(required)}개 전부 존재, 내용 검사 통과")


if __name__ == "__main__":
    main()
