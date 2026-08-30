"""클래스 이름 -> 13 feature 정답표.

`configs/class_features.csv` 한 장이 정답의 출처다. VLM 은 쓰지 않는다.
데이터셋 폴더명(`03_Osstem_SSII`, `SS II`, `IS I` ...)을 이 표의 행에 붙여서
"이 이미지의 System 은 Bone Level" 같은 attribute 정답을 만든다.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

UNKNOWN = 'unknown'

# 표의 13 feature. Company/Name 도 원 표에서 feature 로 세었으므로 포함한다.
ATTRIBUTES: List[str] = [
    'Company', 'Name', 'System', 'Connection_1', 'Connection_2', 'Flange',
    'Collar', 'Microthread', 'Body_Shape', 'Body_Type', 'Thread_Shape',
    'Apex_Shape', 'Apex_Hole',
]

# Name 은 클래스 이름 그 자체라 "처음 보는 클래스"에서는 정의상 맞힐 수 없다.
# 학습 대상에서 기본 제외한다. --attributes 로 되돌릴 수 있다.
DEFAULT_TRAIN_ATTRIBUTES: List[str] = [a for a in ATTRIBUTES if a != 'Name']

DEFAULT_TABLE = Path(__file__).resolve().parents[1] / 'configs' / 'class_features.csv'

_ROMAN = {'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5, 'vi': 6, 'vii': 7, 'viii': 8}


def normalize_key(name: str) -> str:
    """`03_Osstem_SSII`, `SS II`, `SSII` 를 모두 같은 키로 만든다.

    끝에 붙은 로마숫자를 아라비아로 바꿔야 `UF II` 와 `UF2`, `IS I` 와 `ISI` 가 붙는다.
    """
    s = unicodedata.normalize('NFKC', str(name)).lower()
    s = re.sub(r'^\d+[_\-\s]+', '', s)          # 폴더 앞 일련번호 제거: 03_Osstem_SSII
    s = re.sub(r'[^a-z0-9]+', '', s)
    # ponytail: 접미 로마숫자만 변환하는 휴리스틱. 숫자로만 된 접두(Biomet "3i")는
    # 건드리지 않는다. 애매하면 변환하지 않고 미매칭 리포트로 흘려보낸다.
    m = re.match(r'^(.*?)([iv]+)$', s)
    if m:
        prefix, roman = m.group(1), m.group(2)
        if prefix and not prefix.isdigit() and roman in _ROMAN:
            s = f'{prefix}{_ROMAN[roman]}'
    return s


@dataclass
class FeatureTable:
    df: pd.DataFrame                       # 정규화된 45행
    attributes: List[str]                  # 실제로 학습할 attribute
    vocab: Dict[str, List[str]]            # attribute -> 값 목록 (unknown 제외)
    dropped_constant: List[str]            # 값이 1종뿐이라 제외한 attribute

    def value_to_idx(self, attr: str) -> Dict[str, int]:
        return {v: i for i, v in enumerate(self.vocab[attr])}


def load_table(csv_path: str | Path = DEFAULT_TABLE) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str).fillna('')
    missing = [c for c in ATTRIBUTES if c not in df.columns]
    if missing:
        raise ValueError(f'{csv_path} 에 없는 컬럼: {missing}')
    if 'class_key' not in df.columns:
        df['class_key'] = [normalize_key(f'{c}{n}') for c, n in zip(df['Company'], df['Name'])]
    df['match_full'] = [normalize_key(f'{c}_{n}') for c, n in zip(df['Company'], df['Name'])]
    df['match_name'] = [normalize_key(n) for n in df['Name']]
    return df


def match_labels(
    labels: Sequence[str],
    df: pd.DataFrame,
    aliases: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, int], List[str]]:
    """데이터셋 클래스 이름 -> 표의 행 번호. (매핑, 미매칭 라벨) 을 돌려준다.

    company+name 완전일치를 먼저 보고, 그다음 name 단독 일치를 본다.
    실패한 라벨은 조용히 버리지 않고 그대로 돌려준다.
    """
    alias_norm = {normalize_key(k): v for k, v in (aliases or {}).items()}
    by_full = {k: i for i, k in enumerate(df['match_full'])}
    by_name = {k: i for i, k in enumerate(df['match_name'])}

    mapping: Dict[str, int] = {}
    unmatched: List[str] = []
    for label in labels:
        key = normalize_key(label)
        if key in alias_norm:
            key = normalize_key(alias_norm[key])
        row = by_full.get(key)
        if row is None:
            row = by_name.get(key)
        if row is None:
            unmatched.append(label)
        else:
            mapping[label] = int(row)
    return mapping, unmatched


def load_aliases(csv_path: Optional[str | Path]) -> Dict[str, str]:
    if not csv_path:
        return {}
    p = Path(csv_path)
    if not p.exists():
        return {}
    a = pd.read_csv(p, dtype=str).fillna('')
    return dict(zip(a['source_label'], a['canonical_label']))


def build_targets(
    labels: Sequence[str],
    mapping: Dict[str, int],
    df: pd.DataFrame,
    attributes: Optional[Sequence[str]] = None,
    drop_constant: bool = True,
) -> Tuple[FeatureTable, 'object']:
    """매칭된 클래스만으로 (표, class_idx -> attribute target 행렬) 을 만든다.

    반환 행렬은 [num_classes, num_attributes] int64. `unknown` 은 -1 이고
    학습·평가 양쪽에서 loss/metric 에서 제외된다(정답을 모르니 틀렸다고도 못한다).
    """
    import numpy as np

    attrs = list(attributes) if attributes else list(DEFAULT_TRAIN_ATTRIBUTES)
    classes = [l for l in labels if l in mapping]
    rows = df.iloc[[mapping[l] for l in classes]].reset_index(drop=True)

    vocab: Dict[str, List[str]] = {}
    dropped: List[str] = []
    kept: List[str] = []
    for a in attrs:
        vals = sorted({v for v in rows[a] if v and v != UNKNOWN})
        if drop_constant and len(vals) < 2:
            dropped.append(a)
            continue
        vocab[a] = vals
        kept.append(a)

    target = np.full((len(classes), len(kept)), -1, dtype='int64')
    for ci in range(len(classes)):
        for ai, a in enumerate(kept):
            v = rows.at[ci, a]
            if v and v != UNKNOWN and v in vocab[a]:
                target[ci, ai] = vocab[a].index(v)

    table = FeatureTable(df=rows, attributes=kept, vocab=vocab, dropped_constant=dropped)
    return table, target
