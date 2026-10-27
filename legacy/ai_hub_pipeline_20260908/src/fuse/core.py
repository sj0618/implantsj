"""브랜치 점수 로딩 + 방향/스케일 통일. 다른 패키지를 import하지 않는다.

세 브랜치가 내는 [N, C] 점수 CSV를 읽어 하나의 좌표계로 맞춘다.

  브랜치       파일                          방향
  metric      outputs/metric/.../scores_*.csv    거리 (작을수록 가까움)
  classifier  outputs/classifier/.../scores.csv  확률 (클수록 좋음)
  attributes  outputs/attributes/.../scores_*.csv 점수 (클수록 좋음)

방향이 다르므로 yaml의 higher_is_better로 알려주고, 여기서 전부
"클수록 좋음"으로 뒤집은 뒤 정규화한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

NORMALIZE_METHODS = ("softmax", "minmax", "rank", "none")


@dataclass
class BranchScores:
    """한 브랜치의 [N, C] 점수. 항상 '클수록 좋음'으로 맞춰져 있다."""

    name: str
    paths: List[str]
    classes: List[str]
    scores: np.ndarray                 # [N, C]
    source: str

    def __len__(self) -> int:
        return len(self.paths)


def load_branch_scores(
    name: str,
    csv_path: str | Path,
    *,
    higher_is_better: bool = True,
    path_column: str = "path",
) -> BranchScores:
    """브랜치 점수 CSV를 읽어 '클수록 좋음'으로 통일."""
    csv_path = Path(csv_path).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(f"[{name}] 점수 파일이 없습니다: {csv_path}")
    frame = pd.read_csv(csv_path)
    if path_column not in frame.columns:
        raise ValueError(f"[{name}] {csv_path}에 {path_column!r} 컬럼이 없습니다")

    paths = [str(p) for p in frame[path_column]]
    classes = [c for c in frame.columns if c != path_column]
    scores = frame[classes].to_numpy(dtype=np.float64)
    if not higher_is_better:
        scores = -scores               # 거리 → 클수록 좋음
    return BranchScores(name=name, paths=paths, classes=classes,
                        scores=scores, source=str(csv_path))


def align_branches(branches: Sequence[BranchScores]) -> tuple[List[str], List[str], Dict[str, np.ndarray]]:
    """여러 브랜치를 공통 (샘플, 클래스) 축으로 정렬.

    브랜치마다 이미지 순서와 클래스 집합이 다를 수 있으므로,
    모든 브랜치에 존재하는 path의 교집합과 클래스의 합집합으로 맞춘다.
    어떤 브랜치에 없는 클래스는 그 브랜치에서 -inf 취급(=선택되지 않음)이 아니라
    중립값(행 최소값)으로 채워, 그 브랜치가 해당 클래스를 배제하지 않도록 한다.
    """
    if not branches:
        raise ValueError("결합할 브랜치가 없습니다")

    common_paths = set(branches[0].paths)
    for branch in branches[1:]:
        common_paths &= set(branch.paths)
    if not common_paths:
        raise ValueError(
            "브랜치들 사이에 공통 이미지가 없습니다. "
            "같은 manifest/split으로 평가한 결과인지 확인하세요."
        )
    paths = [p for p in branches[0].paths if p in common_paths]   # 첫 브랜치 순서 유지

    classes: List[str] = []
    for branch in branches:
        for c in branch.classes:
            if c not in classes:
                classes.append(c)
    classes = sorted(classes)

    aligned: Dict[str, np.ndarray] = {}
    for branch in branches:
        row_of = {p: i for i, p in enumerate(branch.paths)}
        column_of = {c: i for i, c in enumerate(branch.classes)}
        rows = np.asarray([row_of[p] for p in paths])
        block = branch.scores[rows]                               # [N, C_branch]
        out = np.empty((len(paths), len(classes)), dtype=np.float64)
        neutral = block.min(axis=1, keepdims=True)                # 모르는 클래스는 중립
        for ci, c in enumerate(classes):
            if c in column_of:
                out[:, ci] = block[:, column_of[c]]
            else:
                out[:, ci] = neutral[:, 0]
        aligned[branch.name] = out
    return paths, classes, aligned


def normalize_scores(scores: np.ndarray, method: str = "softmax",
                     temperature: float = 1.0) -> np.ndarray:
    """행(샘플)별로 스케일을 통일. 브랜치마다 단위가 다르므로 결합 전에 필요하다."""
    if method not in NORMALIZE_METHODS:
        raise ValueError(f"normalize must be one of {NORMALIZE_METHODS}")
    scores = np.asarray(scores, dtype=np.float64)

    if method == "none":
        return scores
    if method == "softmax":
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        shifted = scores / temperature
        shifted -= shifted.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)
    if method == "minmax":
        low = scores.min(axis=1, keepdims=True)
        high = scores.max(axis=1, keepdims=True)
        return (scores - low) / np.maximum(high - low, 1e-12)
    # rank: 행마다 0..1 순위 (분포 모양에 둔감)
    order = scores.argsort(axis=1).argsort(axis=1).astype(np.float64)
    return order / max(scores.shape[1] - 1, 1)


def load_labels(csv_path: str | Path, path_column: str = "path",
                label_column: str = "label") -> Dict[str, str]:
    """정답 라벨 매핑 (평가용). manifest 또는 predictions CSV 아무거나."""
    frame = pd.read_csv(csv_path)
    if label_column not in frame.columns:
        for candidate in ("true_label", "label"):
            if candidate in frame.columns:
                label_column = candidate
                break
        else:
            raise ValueError(f"{csv_path}에 라벨 컬럼이 없습니다")
    return {str(p): str(l) for p, l in zip(frame[path_column], frame[label_column])}
