"""
spec table 매칭 점검 도구 (폐쇄망 들어가기 전에 돌려볼 것)

데이터셋 클래스가 spec table의 어느 행에 붙는지 확인하고, 못 붙는 것을 찾아준다.
--fill 을 주면 매칭된 결과를 표의 dataset_label 열에 채워 저장하므로,
폐쇄망에서는 못 붙은 것만 손으로 채우면 된다.

실행:
  python src/check_spec_table.py --config configs/05_attributes.yaml
  python src/check_spec_table.py --config configs/05_attributes.yaml --fill
  python src/check_spec_table.py --config configs/05_attributes.yaml --labels A,B,C
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_utils import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--protocol", choices=["closed27_open15", "full42"])
    parser.add_argument("--labels", help="manifest 대신 콤마로 구분한 클래스 이름 목록")
    parser.add_argument("--fill", action="store_true",
                        help="매칭 결과를 spec table의 dataset_label 열에 채워 저장")
    args = parser.parse_args()

    from attributes.core import read_manifest
    from attributes.spec_table import (
        DATASET_LABEL_COLUMN,
        FEATURES,
        build_spec_table,
        load_spec_csv,
        normalize_column,
    )

    cfg = load_config(args.config)
    spec_cfg = cfg["spec"]
    protocol = args.protocol or cfg["protocol"]
    spec_csv = Path(spec_cfg["csv"])

    # ---- 데이터셋 클래스 목록 ----
    if args.labels:
        labels = sorted({s.strip() for s in args.labels.split(",") if s.strip()})
        source = "--labels 인자"
    else:
        d = cfg["data"]
        manifest = d["closed_manifest"] if protocol == "closed27_open15" else d["full_manifest"]
        if not Path(manifest).is_file():
            print(f"manifest가 없습니다: {manifest}")
            print("데이터가 아직 없으면 --labels 로 클래스 이름을 직접 넘겨 점검하세요.")
            return 1
        frame = read_manifest(manifest)
        labels = sorted(frame["label"].unique().tolist())
        source = manifest

    print(f"데이터셋 클래스 {len(labels)}개 (출처: {source})")
    print(f"spec table: {spec_csv}")

    table = build_spec_table(
        labels, spec_csv,
        attributes=[a for a, c in cfg["features"].items() if (c or {}).get("train")],
        label_aliases=spec_cfg.get("label_aliases"),
        unknown_values=spec_cfg.get("unknown_values"),
        value_aliases=spec_cfg.get("value_aliases"),
        drop_constant=bool(spec_cfg.get("drop_constant", True)),
    )
    df = load_spec_csv(spec_csv, unknown_values=spec_cfg.get("unknown_values"),
                       value_aliases=spec_cfg.get("value_aliases"))

    print()
    print(f"{'데이터셋 클래스':38s} {'매칭 방식':14s} 표의 행")
    print("-" * 88)
    for label in labels:
        if label in table.match_method:
            row = table.rows.iloc[table.classes.index(label)]
            print(f"{label:38s} {table.match_method[label]:14s} "
                  f"{row['Company']} {row['Name']}")
        else:
            print(f"{label:38s} {'✗ 못 붙음':14s} -")

    print()
    print(f"매칭 {len(table.classes)}/{len(labels)}")
    if table.unmatched:
        print(f"⚠ 못 붙은 클래스 {len(table.unmatched)}개:")
        for label in table.unmatched:
            print(f"    {label}")
        print(f"  → {spec_csv}의 {DATASET_LABEL_COLUMN} 열에 이 폴더명을 적어 등록하세요.")
        print(f"    (한 행에 여러 폴더명이면 ';'로 구분)")
    if table.unused_spec_rows:
        print(f"표에만 있고 데이터에 없는 행 {len(table.unused_spec_rows)}개 — 사용되지 않습니다:")
        for name in table.unused_spec_rows:
            print(f"    {name}")

    print()
    print(f"학습 feature {len(table.attributes)}개: {table.attributes}")
    if table.dropped_constant:
        print(f"값이 1종뿐이라 제외: {table.dropped_constant}")
    unknown = {k: v for k, v in table.unknown_report().items() if v}
    if unknown:
        print(f"unknown(정답 없음) 칸: {unknown}")
    print("feature별 값:")
    for a in table.attributes:
        print(f"    {a:15s} {len(table.vocab[a]):2d}종  {table.vocab[a]}")

    # 슬래시로 여러 값이 묶인 칸 안내
    slashes = [(df.at[i, "Company"], df.at[i, "Name"], a, df.at[i, a])
               for i in range(len(df)) for a in FEATURES if "/" in str(df.at[i, a])]
    if slashes:
        print()
        print(f"슬래시로 여러 값이 묶인 칸 {len(slashes)}개 (지금은 각각 별개 값):")
        for company, name, attribute, value in slashes:
            print(f"    {company} {name} / {attribute} = {value!r}")
        print("  → 합치려면 spec.value_aliases에 적으세요.")

    # ---- --fill: 매칭 결과를 dataset_label 열에 기록 ----
    if args.fill:
        raw = pd.read_csv(spec_csv, dtype=str).fillna("")
        columns = [normalize_column(c) for c in raw.columns]
        if DATASET_LABEL_COLUMN not in columns:
            raw.insert(1, DATASET_LABEL_COLUMN, "")
            columns = [normalize_column(c) for c in raw.columns]
        position = columns.index(DATASET_LABEL_COLUMN)
        column_name = raw.columns[position]

        # 클래스 → 표 행 번호를 되찾아 기록
        by_key = {(str(df.at[i, "Company"]), str(df.at[i, "Name"])): i for i in range(len(df))}
        filled = 0
        for label in table.classes:
            row = table.rows.iloc[table.classes.index(label)]
            index = by_key.get((str(row["Company"]), str(row["Name"])))
            if index is None:
                continue
            existing = [s.strip() for s in str(raw.at[index, column_name]).split(";") if s.strip()]
            if label not in existing:
                existing.append(label)
                raw.at[index, column_name] = ";".join(existing)
                filled += 1
        raw.to_csv(spec_csv, index=False)
        print()
        print(f"{DATASET_LABEL_COLUMN} 열에 {filled}개 클래스 이름을 채워 저장했습니다: {spec_csv}")
        print("못 붙은 클래스는 이 열에 직접 타이핑해 등록하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
