"""Open-set 거부 성능 — threshold와 무관하게 "걸러낼 수 있는가"를 잰다.

지금까지는 임계를 정한 뒤 unknown 비율만 봤는데, 그러면 임계 선택에 따라
평가가 흔들린다. AUROC/FPR@95는 **임계를 정하지 않고** 순위만으로 재므로
모델끼리 비교하기에 낫다.

  reject_auroc  unknown 점수가 known 점수보다 높게 나올 확률 (1.0 = 완벽)
  reject_fpr95  unknown을 95% 잡을 때 known을 얼마나 잘못 버리는가 (낮을수록 좋음)

거부 점수 5종을 함께 비교한다 — 무엇으로 unknown을 판정할지가 성능을 크게 가른다:
  min_dist   최근접 prototype까지의 거리        (기본. 멀면 unknown)
  margin     1등과 2등 거리의 차                (작으면 애매 = unknown)
  dist_ent   거리를 softmax한 분포의 엔트로피    (고르면 애매)
  logit_msp  거리 기반 확률의 최대값             (작으면 unknown)
  logit_ent  같은 확률의 엔트로피                (크면 unknown)

closed27_open15에서 open 15는 임베딩 학습에 쓰이지 않았으므로
모델 입장에서는 처음 보는 클래스다 — 이들을 unknown으로 본다.
출처: implantys "metric learning" loo_eval.py / diagnostics.py
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

REJECT_SCORES = ("min_dist", "margin", "dist_ent", "logit_msp", "logit_ent")


def auroc(unknown: np.ndarray, known: np.ndarray) -> float:
    """P(unknown 점수 > known 점수). 동점은 0.5로 센다 (순위 평균)."""
    unknown = np.asarray(unknown, dtype=np.float64)
    known = np.asarray(known, dtype=np.float64)
    if len(unknown) == 0 or len(known) == 0:
        return float("nan")

    values = np.concatenate([unknown, known])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)

    # 동점 구간은 평균 순위로 (동점을 0.5로 세는 효과)
    sorted_values = values[order]
    i = 0
    while i < len(sorted_values):
        j = i
        while j + 1 < len(sorted_values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1

    n_unknown, n_known = len(unknown), len(known)
    return float((ranks[:n_unknown].sum() - n_unknown * (n_unknown + 1) / 2)
                 / (n_unknown * n_known))


def fpr_at_tpr(unknown: np.ndarray, known: np.ndarray, target_tpr: float = 0.95) -> float:
    """unknown을 target_tpr만큼 잡는 임계에서, known을 잘못 버리는 비율."""
    unknown = np.asarray(unknown, dtype=np.float64)
    known = np.asarray(known, dtype=np.float64)
    if len(unknown) == 0 or len(known) == 0:
        return float("nan")
    threshold = np.quantile(unknown, 1 - target_tpr)   # 이 값 이상이면 unknown 판정
    return float((known >= threshold).mean())


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def reject_scores(distances: np.ndarray, *, temperature: float = 1.0,
                  which: Sequence[str] = REJECT_SCORES) -> Dict[str, np.ndarray]:
    """[N, C] 거리 행렬 → 거부 점수들. 값이 **클수록 unknown**으로 통일한다.

    distances는 '작을수록 가까움'인 거리다 (metric 브랜치의 score와 같은 방향).
    """
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[1] < 2:
        raise ValueError(f"거리 행렬은 [N, C>=2] 여야 합니다: {distances.shape}")
    ordered = np.sort(distances, axis=1)
    probabilities = _softmax(-distances / max(temperature, 1e-8), axis=1)

    out: Dict[str, np.ndarray] = {}
    for name in which:
        if name == "min_dist":
            out[name] = ordered[:, 0]                       # 멀수록 unknown
        elif name == "margin":
            out[name] = -(ordered[:, 1] - ordered[:, 0])    # 차이가 작을수록 unknown
        elif name == "dist_ent":
            out[name] = -(probabilities * np.log(probabilities + 1e-12)).sum(axis=1)
        elif name == "logit_msp":
            out[name] = -probabilities.max(axis=1)          # 확신이 낮을수록 unknown
        elif name == "logit_ent":
            out[name] = -(probabilities * np.log(probabilities + 1e-12)).sum(axis=1)
        else:
            raise ValueError(f"reject score는 {REJECT_SCORES} 중 하나여야 합니다: {name}")
    return out


def evaluate_rejection(distances: np.ndarray, is_unknown: np.ndarray, *,
                       which: Sequence[str] = REJECT_SCORES,
                       temperature: float = 1.0) -> dict:
    """거부 점수별 AUROC / FPR@95를 한 번에.

    is_unknown: 학습에 없던 클래스(= open)인지 여부 [N]
    """
    is_unknown = np.asarray(is_unknown, dtype=bool)
    if not is_unknown.any() or is_unknown.all():
        return {"available": False,
                "reason": "known 또는 unknown 한쪽이 비어 거부 성능을 잴 수 없습니다",
                "n_known": int((~is_unknown).sum()), "n_unknown": int(is_unknown.sum())}

    scores = reject_scores(distances, temperature=temperature, which=which)
    result = {"available": True,
              "n_known": int((~is_unknown).sum()), "n_unknown": int(is_unknown.sum()),
              "scores": {}}
    for name, value in scores.items():
        result["scores"][name] = {
            "auroc": round(auroc(value[is_unknown], value[~is_unknown]), 6),
            "fpr_at_95tpr": round(fpr_at_tpr(value[is_unknown], value[~is_unknown]), 6),
        }
    best = max(result["scores"], key=lambda k: result["scores"][k]["auroc"])
    result["best_score"] = best
    result["best_auroc"] = result["scores"][best]["auroc"]
    return result
