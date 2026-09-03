#!/usr/bin/env python3
"""arcface_attributes 의 attributes.json 여러 개를 한 표로 비교한다.

메뉴 18 이 head_source 두 가지(embedding / backbone)를 돌리고 이 도구로 묶는다.
attribute 정확도는 최빈값 baseline 이 이미 높아서(예: Apex_Hole 91%) 절대값만
보면 안 된다. 항상 baseline 과 그 차이를 같이 낸다.

  python3 tools/afa_compare.py <run>/attributes.json <run2>/attributes.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(paths: list[str]) -> list[dict]:
    out = []
    for p in paths:
        f = Path(p)
        if not f.is_file():
            print(f"!! 없음: {p}")
            continue
        out.append(json.load(f.open(encoding="utf-8")))
    return out


def main(argv: list[str]) -> int:
    rows = load(argv)
    if not rows:
        print("비교할 리포트가 없다.")
        return 1

    print(f"{'head_source':12s} {'macro':>7s} {'baseline':>9s} {'차이':>7s} "
          f"{'seen':>7s} {'unseen':>7s}  분기(갤러리/head)")
    for d in rows:
        b = d["branch"]
        print(f"{d['head_source']:12s} {d['macro_attr_acc']:7.3f} "
              f"{d['macro_majority_baseline']:9.3f} {d['macro_gain_over_baseline']:+7.3f} "
              f"{d['seen_macro_acc']:7.3f} {d['unseen_macro_acc']:7.3f}  "
              f"{b['gallery']}/{b['head']}")

    # attribute 별로 baseline 을 넘었는지가 실제로 봐야 할 것이다.
    attrs = rows[0]["per_attribute"].keys()
    print(f"\n{'attribute':14s} {'baseline':>9s} " +
          " ".join(f"{d['head_source'][:9]:>9s}" for d in rows))
    for a in attrs:
        base = rows[0]["per_attribute"][a]["majority_baseline"]
        cells = []
        for d in rows:
            v = d["per_attribute"].get(a, {}).get("acc", float("nan"))
            mark = "+" if v > base else " "
            cells.append(f"{v:8.3f}{mark}")
        print(f"{a:14s} {base:9.3f} " + " ".join(cells))
    print("  ('+' 는 최빈값 baseline 을 넘긴 것. 넘기지 못한 항목은 학습이 "
          "아무것도 더하지 못했다는 뜻이다.)")

    if rows[0].get("unseen_probe_count", 0) == 0:
        print("\n주의: unseen 판정 칸이 0 개다. 갤러리가 모든 클래스를 알고 있어")
        print("      head 분기가 돌지 않았다. exp.conf 의 ARCFACE_HOLDOUT 을 올리거나")
        print("      unseen 이 분리된 데이터셋을 쓴다. 지금 숫자는 seen 성능일 뿐이다.")

    spec = rows[0].get("spec", {})
    if spec.get("dropped_constant"):
        print(f"\n값이 1종뿐이라 제외된 attribute: {spec['dropped_constant']}")
    if spec.get("unmatched_labels"):
        print(f"표에 없어 제외된 클래스: {spec['unmatched_labels']}")
    if spec.get("unreachable_values"):
        print(f"head 가 예측할 수 없는 값(학습에 없음): {spec['unreachable_values']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
