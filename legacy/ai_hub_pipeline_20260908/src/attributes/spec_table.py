"""spec table (클래스 → 13 feature 정답표) 로딩 + 데이터셋 클래스 매칭.

configs/spec_table.csv 한 장이 정답의 출처다.
데이터셋 폴더명(`03_Osstem_SSII`, `SS II`, `IS I` ...)을 표의 행에 붙여
"이 이미지의 System은 Bone Level" 같은 feature 정답을 만든다.

원본 표(임플란트 라벨링.xlsx)를 그대로 읽을 수 있게 처리하는 것들:
  - 컬럼명 정규화: `Connection(1)`→`Connection_1`, `Body Shape`→`Body_Shape`,
    `Conection(2)`(오타)→`Connection_2`
  - unknown 표기: `확인 불가` 등은 -1로 두어 loss/metric에서 제외한다
    (정답을 모르는 칸을 틀렸다고 셀 수는 없다)
  - 값 표기 흔들림: yaml의 value_aliases로 통합 (`Hexagon`→`HEX` 등)

출처: implantsj src/class_features.py의 매칭 로직
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# 학습 대상 feature 11개.
# 제외한 컬럼:
#   Name      데이터셋 클래스 이름 그 자체(= 라벨)라 feature가 아니다
#   Body_Type 표의 모든 클래스가 같은 값(Threaded)이라 분류에 기여하지 않는다
FEATURES: List[str] = [
    "Company", "System", "Connection_1", "Connection_2", "Flange",
    "Collar", "Microthread", "Body_Shape", "Thread_Shape",
    "Apex_Shape", "Apex_Hole",
]

# 클래스 식별용 컬럼 (feature가 아님)
LABEL_COLUMNS: List[str] = ["Company", "Name"]

# 데이터셋 폴더명을 직접 적는 열. 비어 있으면 Company+Name으로 자동 매칭하고,
# 값이 있으면 그 값을 데이터셋 클래스 이름으로 우선 사용한다.
# 여러 폴더명이 한 클래스에 대응하면 ';'로 구분해 적는다.
DATASET_LABEL_COLUMN = "dataset_label"

# 하위 호환 (Name 포함 13개 컬럼이 표에 존재하는지 확인할 때 사용)
ATTRIBUTES: List[str] = FEATURES

# 정답이 없는 칸으로 볼 표기 (대소문자/공백 무시하고 비교)
DEFAULT_UNKNOWN_VALUES: List[str] = ["확인 불가", "확인불가", "unknown", "n/a", "na", "-", "?"]

_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8}


def normalize_column(name: str) -> str:
    """`Connection(1)`→`Connection_1`, `Body Shape`→`Body_Shape`, `Conection`오타 교정."""
    s = str(name).strip()
    s = s.replace("Conection", "Connection")     # 원본 표의 오타
    s = re.sub(r"\((\d)\)", r"_\1", s)           # Connection(1) -> Connection_1
    s = re.sub(r"\s+", "_", s)                   # Body Shape -> Body_Shape
    return s


def normalize_key(name: str) -> str:
    """`03_Osstem_SSII`, `SS II`, `SSII`를 모두 같은 키로 만든다."""
    s = unicodedata.normalize("NFKC", str(name)).lower()
    s = re.sub(r"^\d+[_\-\s]+", "", s)           # 폴더 앞 일련번호 제거
    s = re.sub(r"[^a-z0-9]+", "", s)
    # 접미 로마숫자만 아라비아로 (UF II ↔ UF2). 숫자 접두(Biomet "3i")는 건드리지 않는다.
    m = re.match(r"^(.*?)([iv]+)$", s)
    if m:
        prefix, roman = m.group(1), m.group(2)
        if prefix and not prefix.isdigit() and roman in _ROMAN:
            s = f"{prefix}{_ROMAN[roman]}"
    return s


@dataclass
class SpecTable:
    """매칭된 클래스에 대한 feature vocab과 정답 행렬."""

    classes: List[str]                     # 데이터셋 클래스 (표에 붙은 것만)
    attributes: List[str]                  # 실제로 쓰는 feature
    vocab: Dict[str, List[str]]            # feature → 값 목록 (unknown 제외)
    target: np.ndarray                     # [num_classes, num_attributes], unknown은 -1
    dropped_constant: List[str]            # 값이 1종뿐이라 제외한 feature
    unmatched: List[str]                   # 표에 못 붙은 데이터셋 클래스
    rows: pd.DataFrame                     # 클래스별 원본 표 행
    match_method: Dict[str, str]           # 라벨별 매칭 방식 (dataset_label / company+name / name)
    unused_spec_rows: List[str]            # 표에만 있고 데이터셋에 없는 행 (45행 vs 42클래스)

    def head_sizes(self) -> Dict[str, int]:
        return {a: len(self.vocab[a]) for a in self.attributes}

    def unknown_report(self) -> Dict[str, int]:
        """feature별 unknown 칸 수 — 정답이 없어 학습/평가에서 빠지는 양."""
        return {a: int((self.target[:, i] < 0).sum()) for i, a in enumerate(self.attributes)}

    def describe(self) -> dict:
        return {
            "classes": self.classes,
            "attributes": self.attributes,
            "vocab": self.vocab,
            "dropped_constant_attributes": self.dropped_constant,
            "unmatched_classes": self.unmatched,
            "match_method": self.match_method,
            "unused_spec_rows": self.unused_spec_rows,
            "unknown_cells_per_attribute": self.unknown_report(),
            "unknown_cells_total": int((self.target < 0).sum()),
        }


def load_spec_csv(
    csv_path: str | Path,
    *,
    unknown_values: Optional[Sequence[str]] = None,
    value_aliases: Optional[Dict[str, Dict[str, str]]] = None,
) -> pd.DataFrame:
    """원본 표를 읽어 컬럼명 정규화 + unknown 통일 + 값 alias 적용."""
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    df.columns = [normalize_column(c) for c in df.columns]
    missing = [c for c in FEATURES + ["Name"] if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}에 없는 컬럼: {missing}")
    if DATASET_LABEL_COLUMN not in df.columns:
        df[DATASET_LABEL_COLUMN] = ""      # 없으면 전부 자동 매칭

    unknown_set = {str(v).strip().lower() for v in (unknown_values or DEFAULT_UNKNOWN_VALUES)}
    aliases = value_aliases or {}
    for column in FEATURES:
        mapping = {str(k).strip().lower(): v for k, v in (aliases.get(column) or {}).items()}
        cleaned = []
        for raw in df[column]:
            value = str(raw).strip()
            key = value.lower()
            if not value or key in unknown_set:
                cleaned.append("")            # unknown은 빈 문자열로 통일 → 뒤에서 -1
            else:
                cleaned.append(mapping.get(key, value))
        df[column] = cleaned

    df["match_full"] = [normalize_key(f"{c}_{n}") for c, n in zip(df["Company"], df["Name"])]
    df["match_name"] = [normalize_key(n) for n in df["Name"]]
    return df


def match_labels(labels: Sequence[str], df: pd.DataFrame,
                 aliases: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, int], List[str], Dict[str, str]]:
    """데이터셋 클래스 이름 → 표의 행 번호.

    우선순위:
      1) dataset_label 열에 직접 적은 이름 (';'로 여러 개 가능) — 폐쇄망 대비 수동 등록
      2) yaml의 label_aliases
      3) Company+Name 자동 매칭
      4) Name 단독 자동 매칭

    반환: (매핑, 못 붙은 라벨, 라벨별 매칭 방식)
    """
    alias_norm = {normalize_key(k): v for k, v in (aliases or {}).items()}
    by_full = {k: i for i, k in enumerate(df["match_full"])}
    by_name = {k: i for i, k in enumerate(df["match_name"])}

    # dataset_label 열은 정확히 적은 이름이므로 자동 매칭보다 우선한다
    manual: Dict[str, int] = {}
    for row_index, raw in enumerate(df.get(DATASET_LABEL_COLUMN, [""] * len(df))):
        for piece in str(raw).split(";"):
            piece = piece.strip()
            if piece:
                manual[normalize_key(piece)] = row_index

    mapping: Dict[str, int] = {}
    unmatched: List[str] = []
    how: Dict[str, str] = {}
    for label in labels:
        key = normalize_key(label)
        if key in manual:
            mapping[label] = manual[key]
            how[label] = DATASET_LABEL_COLUMN
            continue
        if key in alias_norm:
            key = normalize_key(alias_norm[key])
        if key in by_full:
            mapping[label] = by_full[key]
            how[label] = "company+name"
        elif key in by_name:
            mapping[label] = by_name[key]
            how[label] = "name"
        else:
            unmatched.append(label)          # 조용히 버리지 않고 그대로 돌려준다
    return mapping, unmatched, how


def build_spec_table(
    labels: Sequence[str],
    spec_csv: str | Path,
    *,
    attributes: Optional[Sequence[str]] = None,
    label_aliases: Optional[Dict[str, str]] = None,
    unknown_values: Optional[Sequence[str]] = None,
    value_aliases: Optional[Dict[str, Dict[str, str]]] = None,
    drop_constant: bool = True,
) -> SpecTable:
    """데이터셋 라벨 목록으로 SpecTable을 만든다.

    attributes를 주면 그 feature만 쓴다(yaml의 feature 선택).
    값이 1종뿐인 feature는 분류에 정보가 없으므로 기본적으로 제외한다.
    """
    df = load_spec_csv(spec_csv, unknown_values=unknown_values, value_aliases=value_aliases)
    mapping, unmatched, how = match_labels(labels, df, label_aliases)
    matched = [l for l in labels if l in mapping]
    if not matched:
        raise RuntimeError(
            f"spec table({spec_csv})에 붙는 클래스가 하나도 없습니다. "
            f"데이터셋 클래스 예: {list(labels)[:5]}"
        )
    rows = df.iloc[[mapping[l] for l in matched]].reset_index(drop=True)

    # 표에만 있고 데이터셋에 없는 행 (표 45행 vs 데이터 42클래스 같은 경우).
    # 학습/분류에는 쓰이지 않으므로 문제는 없지만, 매칭 실수와 구분되게 남긴다.
    used_rows = set(mapping.values())
    unused = [f"{df.at[i, 'Company']} {df.at[i, 'Name']}"
              for i in range(len(df)) if i not in used_rows]

    attrs = list(attributes) if attributes else list(FEATURES)
    unknown_attrs = [a for a in attrs if a not in FEATURES]
    if unknown_attrs:
        raise ValueError(
            f"알 수 없는 feature: {unknown_attrs}. 가능한 값: {FEATURES} "
            f"(Name은 라벨이므로 feature가 아닙니다)"
        )

    vocab: Dict[str, List[str]] = {}
    kept: List[str] = []
    dropped: List[str] = []
    for a in attrs:
        values = sorted({v for v in rows[a] if v})
        if drop_constant and len(values) < 2:
            dropped.append(a)                # 값이 하나뿐이면 분류에 기여 못 함
            continue
        vocab[a] = values
        kept.append(a)

    target = np.full((len(matched), len(kept)), -1, dtype="int64")
    for ci in range(len(matched)):
        for ai, a in enumerate(kept):
            value = rows.at[ci, a]
            if value and value in vocab[a]:
                target[ci, ai] = vocab[a].index(value)

    return SpecTable(classes=matched, attributes=kept, vocab=vocab, target=target,
                     dropped_constant=dropped, unmatched=unmatched, rows=rows,
                     match_method=how, unused_spec_rows=unused)
