#!/usr/bin/env python3
"""
셀 러너 — 캐시된 임베딩으로 성능 셀을 계산해 results.csv 에 누적한다.

  python run_cells.py --cond M1
  python run_cells.py --all          # embeddings/ 에 있는 조건 전부

셀 = (조건 × 시나리오 × 거리). 총 22셀.

시나리오
  E1  13개로 학습 → 13개 분류          (M2 M4 M6)
  E2   9개로 학습 →  9개 분류          (M3 M5 M7)
  E3   9개로 학습 → 13개 분류          (M3 M5 M7)  ← closed 4개는 처음 보는 회사
  M1(Base)은 학습이 없어 E1 과 E3 가 같은 계산이 된다. 중복을 피해
  13개 분류 / 9개 분류 두 가지만 돌리고 라벨을 BASE13 / BASE9 로 붙인다.

프로토타입은 항상 train split 에서 만든다 (val/test 사용은 누수).
E3 에서 closed 4개의 프로토타입도 train 임베딩으로 만든다 — 모델이 그 클래스를
학습한 적은 없지만, 대표 벡터를 만들 예시는 있다는 것이 프로토타입 분류의 전제다.
"""

import argparse
import csv
import json
from pathlib import Path

from dataset import load_split
from metrics import evaluate, filter_to_classes
from prototype import EMB_ROOT, METRICS, load_embeddings

_ROOT = Path(__file__).parent
RESULTS_CSV = _ROOT / "results.csv"
PERCLASS_JSON = _ROOT / "results_per_class.json"

# 조건 → (아키텍처, 학습 범위). 학습 범위 None = 학습 없음.
CONDITIONS = {
    "M1": ("Base", None),
    "M2": ("Head", "all"),
    "M3": ("Head", "open"),
    "M4": ("FullFT", "all"),
    "M5": ("FullFT", "open"),
    "M6": ("FullFT-SupCon", "all"),
    "M7": ("FullFT-SupCon", "open"),
    # 변형 조건 — 계획서의 7조건 밖. M6 이 M4 보다 나빴던 원인이
    # balanced sampler 의 소수 클래스 과표집인지 확인하려고 추가.
    # (nobel 은 train 3장인데 13클래스 균형 샘플링이면 에폭당 ~70회 뽑힌다)
    "M6-nobal": ("FullFT-SupCon (no balanced sampler)", "all"),
}
VARIANTS = {"M6-nobal"}  # 22셀 집계에서 제외하고 별도 표기

# 시나리오 → (평가 클래스 범위, 설명)
SCENARIOS = {
    "E1": ("all", "13개로 학습 → 13개 분류"),
    "E2": ("open", "9개로 학습 → 9개 분류"),
    "E3": ("all", "9개로 학습 → 13개 분류 (closed 4개는 미학습)"),
    "BASE13": ("all", "학습 없음 → 13개 분류"),
    "BASE9": ("open", "학습 없음 → 9개 분류"),
}

FIELDS = ["cond", "arch", "train_scope", "scenario", "scenario_desc", "metric",
          "n_classes", "n_test", "top1", "top3", "macro_f1", "unseen_macro_f1"]


def scenarios_for(cond):
    _, scope = CONDITIONS[cond]
    if scope is None:
        return ["BASE13", "BASE9"]
    if scope == "all":
        return ["E1"]
    return ["E2", "E3"]


def run_cond(cond, meta, emb_root=EMB_ROOT):
    arch, train_scope = CONDITIONS[cond]
    tr_v, tr_l, _ = load_embeddings(cond, "train", emb_root)
    te_v, te_l, _ = load_embeddings(cond, "test", emb_root)

    all_classes = sorted(meta["classes"])
    open_classes = sorted(meta["open_classes"])
    closed_classes = sorted(meta["closed_classes"])

    rows, per_class = [], {}
    for scen in scenarios_for(cond):
        eval_scope, desc = SCENARIOS[scen]
        classes = all_classes if eval_scope == "all" else open_classes

        trv, trl = filter_to_classes(tr_v, tr_l, classes)
        tev, tel = filter_to_classes(te_v, te_l, classes)

        for metric in METRICS:
            r = evaluate(tev, tel, trv, trl, classes, metric)

            # E3 에서만: 학습에 안 쓴 closed 4개만 따로 본 macro-F1.
            # 전체 macro-F1 에 묻히는 "처음 보는 회사" 성능을 분리해 기록한다.
            unseen = ""
            if scen == "E3":
                f1s = [r["per_class"][c]["f1"] for c in closed_classes]
                unseen = round(sum(f1s) / len(f1s), 6)

            rows.append({
                "cond": cond, "arch": arch,
                "train_scope": train_scope or "none",
                "scenario": scen, "scenario_desc": desc, "metric": metric,
                "n_classes": r["n_classes"], "n_test": r["n_test"],
                "top1": round(r["top1"], 6),
                "top3": round(r["top3"], 6),
                "macro_f1": round(r["macro_f1"], 6),
                "unseen_macro_f1": unseen,
            })
            per_class[f"{cond}|{scen}|{metric}"] = r["per_class"]
    return rows, per_class


def merge_rows(existing, new):
    """(cond, scenario, metric) 를 키로 갱신. 재실행이 중복 행을 만들지 않게."""
    key = lambda r: (r["cond"], r["scenario"], r["metric"])
    merged = {key(r): r for r in existing}
    for r in new:
        merged[key(r)] = r
    order = list(CONDITIONS)
    return sorted(merged.values(),
                  key=lambda r: (order.index(r["cond"]) if r["cond"] in order else 99,
                                 r["scenario"], r["metric"]))


def load_existing(path, fields):
    if not Path(path).is_file():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return [{k: row.get(k, "") for k in fields} for row in csv.DictReader(fh)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="*", default=None, help="M1 ~ M7")
    ap.add_argument("--all", action="store_true",
                    help="embeddings/ 에 추출돼 있는 조건 전부")
    ap.add_argument("--emb-root", default=EMB_ROOT)
    args = ap.parse_args()

    emb_root = Path(args.emb_root)
    if args.all:
        conds = [c for c in CONDITIONS if (emb_root / c / "train.npz").is_file()]
        if not conds:
            raise SystemExit(f"{emb_root} 에 추출된 조건이 없다. extract.py 를 먼저 실행할 것.")
    elif args.cond:
        conds = args.cond
    else:
        raise SystemExit("--cond 또는 --all 중 하나가 필요하다.")

    unknown = [c for c in conds if c not in CONDITIONS]
    if unknown:
        raise SystemExit(f"알 수 없는 조건: {unknown}  (가능: {list(CONDITIONS)})")

    meta = load_split()
    new_rows, per_class = [], {}
    for cond in conds:
        rows, pc = run_cond(cond, meta, emb_root)
        new_rows += rows
        per_class.update(pc)
        print(f"[{cond}] {len(rows)}셀")
        for r in rows:
            print(f"  {r['scenario']:<7} {r['metric']:<7} "
                  f"top1={r['top1']:.4f}  top3={r['top3']:.4f}  "
                  f"macroF1={r['macro_f1']:.4f}"
                  + (f"  미학습4개F1={r['unseen_macro_f1']:.4f}"
                     if r["unseen_macro_f1"] != "" else ""))

    merged = merge_rows(load_existing(RESULTS_CSV, FIELDS), new_rows)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(merged)

    old_pc = {}
    if PERCLASS_JSON.is_file():
        with open(PERCLASS_JSON, encoding="utf-8") as fh:
            old_pc = json.load(fh)
    old_pc.update(per_class)
    with open(PERCLASS_JSON, "w", encoding="utf-8") as fh:
        json.dump(old_pc, fh, ensure_ascii=False, indent=2)

    done = sorted({r["cond"] for r in merged}, key=lambda c: list(CONDITIONS).index(c))
    core = [r for r in merged if r["cond"] not in VARIANTS]
    var = sorted({r["cond"] for r in merged if r["cond"] in VARIANTS})
    print(f"\n저장: {RESULTS_CSV.name} ({len(core)}행 / 목표 22) — 완료 조건 "
          f"{[c for c in done if c not in VARIANTS]}")
    if var:
        print(f"      변형 조건 {var} — {len(merged)-len(core)}행 (22셀 집계와 별도)")
    print(f"      {PERCLASS_JSON.name} (클래스별 성능)")


if __name__ == "__main__":
    main()
