"""feature 예측 → 최종 클래스 점수.

각 feature 모델이 값별 확률 P_f(v|x)를 낸다. spec table에서 클래스 c의 feature f
정답이 s(c,f)이므로, "이 이미지가 클래스 c와 얼마나 맞는가"는
클래스마다 feature별 일치 확률 M[i,c,f] = P_f(s(c,f) | x_i) 로 정리된다.

fuse 방식 2가지 (yaml의 fuse.mode):
  learned    feature별 가중치 w_f를 valid에서 직접 학습 → score = Σ_f w_f · M[:,:,f]
             (w는 softplus로 음수 방지, 합=1로 정규화. 파라미터가 feature 수만큼뿐)
  min_match  사용자가 지정한 n개 이상의 feature가 일치해야 그 라벨로 인정.
             일치 = argmax P_f == s(c,f). n 미만이면 unknown 판정.

두 방식 모두 **[N, C] 연속 점수**를 낸다 (클수록 그 클래스일 가능성이 높음).
다른 브랜치(metric / classifier)와 Stage 5에서 합칠 수 있도록
predictions.csv에 정규화 점수와 상위 후보를 함께 저장한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

FUSE_MODES = ("learned", "min_match")
UNKNOWN_LABEL = "__unknown__"


def match_probability_tensor(
    probabilities: Dict[str, np.ndarray],
    target: np.ndarray,
    attributes: Sequence[str],
) -> np.ndarray:
    """M[i, c, f] = P_f(spec[c,f] | x_i). unknown 칸(-1)은 중립값 1/|values|.

    probabilities[f]: [N, n_values_f]
    target: [C, F] (spec table 정답 인덱스)
    """
    n = next(iter(probabilities.values())).shape[0]
    num_classes, num_attrs = target.shape
    matrix = np.zeros((n, num_classes, num_attrs), dtype=np.float64)
    for ai, a in enumerate(attributes):
        probs = probabilities[a]                       # [N, V]
        for ci in range(num_classes):
            value = int(target[ci, ai])
            if value < 0:
                # 정답을 모르는 칸: 어느 쪽으로도 유리/불리하지 않게 균등 확률
                matrix[:, ci, ai] = 1.0 / probs.shape[1]
            else:
                matrix[:, ci, ai] = probs[:, value]
    return matrix


def match_indicator_tensor(
    probabilities: Dict[str, np.ndarray],
    target: np.ndarray,
    attributes: Sequence[str],
) -> np.ndarray:
    """I[i, c, f] = 1 if argmax P_f(x_i) == spec[c,f] else 0. unknown 칸은 0."""
    n = next(iter(probabilities.values())).shape[0]
    num_classes, num_attrs = target.shape
    indicator = np.zeros((n, num_classes, num_attrs), dtype=np.float64)
    for ai, a in enumerate(attributes):
        predicted = probabilities[a].argmax(axis=1)    # [N]
        for ci in range(num_classes):
            value = int(target[ci, ai])
            if value >= 0:
                indicator[:, ci, ai] = (predicted == value).astype(np.float64)
    return indicator


# =====================================================================
# [1] learned — feature 가중치를 직접 학습
# =====================================================================

def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)          # log(1+e^x), overflow 안전


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def learn_feature_weights(
    matrix: np.ndarray,
    true_class_index: np.ndarray,
    attributes: Sequence[str],
    *,
    epochs: int = 300,
    lr: float = 0.05,
    seed: int = 42,
    verbose: bool = False,
) -> np.ndarray:
    """valid 세트에서 score = Σ_f w_f·M[:,:,f] 가 정답 클래스를 고르도록 w를 학습.

    w = softplus(raw)로 양수 제약, 마지막에 합=1로 정규화.
    파라미터가 feature 수만큼(13개)뿐이라 numpy Adam으로 충분하다.
    softmax cross-entropy의 기울기를 직접 계산한다:
        p = softmax(score);  dL/dscore = (p - onehot)/N
        dL/dw = Σ_{i,c} dL/dscore[i,c] · M[i,c,f]
        dL/draw = dL/dw · sigmoid(raw)
    """
    matrix = np.asarray(matrix, dtype=np.float64)                   # [N, C, F]
    y = np.asarray(true_class_index, dtype=np.int64)                # [N]
    n, num_classes, num_features = matrix.shape
    if num_features != len(attributes):
        raise ValueError(f"matrix feature 수 {num_features} != attributes {len(attributes)}")

    raw = np.zeros(num_features)                                    # softplus(0)≈0.69, 균등 시작
    m_state = np.zeros(num_features)
    v_state = np.zeros(num_features)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    onehot = np.zeros((n, num_classes))
    onehot[np.arange(n), y] = 1.0

    for step in range(1, epochs + 1):
        weights = _softplus(raw)
        scores = matrix @ weights                                   # [N, C]
        scores -= scores.max(axis=1, keepdims=True)                 # softmax 안정화
        exp_scores = np.exp(scores)
        probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        grad_scores = (probabilities - onehot) / n                  # [N, C]
        grad_w = np.einsum("nc,ncf->f", grad_scores, matrix)
        grad_raw = grad_w * _sigmoid(raw)

        m_state = beta1 * m_state + (1 - beta1) * grad_raw
        v_state = beta2 * v_state + (1 - beta2) * grad_raw ** 2
        m_hat = m_state / (1 - beta1 ** step)
        v_hat = v_state / (1 - beta2 ** step)
        raw -= lr * m_hat / (np.sqrt(v_hat) + eps)

        if verbose and step % max(epochs // 5, 1) == 0:
            loss = -np.log(np.maximum(probabilities[np.arange(n), y], 1e-12)).mean()
            print(f"    weight fit step={step} loss={loss:.4f}")

    weights = _softplus(raw)
    total = weights.sum()
    return weights / total if total > 0 else np.ones(num_features) / num_features


# =====================================================================
# 점수 산출
# =====================================================================

def fuse_scores(
    probabilities: Dict[str, np.ndarray],
    target: np.ndarray,
    attributes: Sequence[str],
    *,
    mode: str = "learned",
    weights: np.ndarray | None = None,
    min_match: int = 8,
) -> dict:
    """[N, C] 점수 + 부가 정보. 점수는 클수록 그 클래스일 가능성이 높다."""
    if mode not in FUSE_MODES:
        raise ValueError(f"fuse mode must be one of {FUSE_MODES}")

    matrix = match_probability_tensor(probabilities, target, attributes)
    indicator = match_indicator_tensor(probabilities, target, attributes)
    match_counts = indicator.sum(axis=2)                            # [N, C] 일치 feature 수

    if mode == "learned":
        if weights is None:
            weights = np.ones(len(attributes)) / len(attributes)
        scores = (matrix * weights[None, None, :]).sum(axis=2)      # [N, C]
        rejected = np.zeros(len(scores), dtype=bool)
    else:  # min_match
        # 점수는 균등 가중 확률 합(연속값) — 순위와 Fuse 입력으로 쓴다
        scores = matrix.mean(axis=2)
        best = scores.argmax(axis=1)
        # 최고 점수 클래스의 일치 feature 수가 기준 미만이면 unknown
        rejected = match_counts[np.arange(len(best)), best] < min_match
        weights = np.ones(len(attributes)) / len(attributes)

    return {
        "scores": scores,                     # [N, C] 클수록 좋음
        "match_counts": match_counts,         # [N, C]
        "rejected": rejected,                 # [N] min_match 미달
        "weights": np.asarray(weights),
        "mode": mode,
    }


def scores_to_frame(
    result: dict,
    classes: Sequence[str],
    paths: Sequence[str],
    true_labels: Sequence[str],
    *,
    top_k: int = 2,
) -> pd.DataFrame:
    """Stage 5 Fuse가 읽을 예측 CSV. 점수는 클수록 좋음(higher_is_better)."""
    scores = result["scores"]
    counts = result["match_counts"]
    rejected = result["rejected"]
    order = np.argsort(-scores, axis=1)                    # 내림차순
    class_array = np.asarray(classes, dtype=object)

    total = scores.sum(axis=1, keepdims=True)
    normalized = scores / np.maximum(total, 1e-12)         # 합=1 정규화 (브랜치 간 비교용)

    rows = []
    for i in range(len(scores)):
        best = int(order[i, 0])
        row = {
            "path": paths[i],
            "true_label": true_labels[i],
            "pred_label": UNKNOWN_LABEL if rejected[i] else str(class_array[best]),
            "score": float(scores[i, best]),               # 원 점수 (클수록 좋음)
            "score_normalized": float(normalized[i, best]),  # 합=1 정규화 점수
            "matched_features": int(counts[i, best]),
            "rejected_unknown": bool(rejected[i]),
            "correct": (not rejected[i]) and str(class_array[best]) == true_labels[i],
        }
        for k in range(1, min(top_k, scores.shape[1])):
            j = int(order[i, k])
            row[f"top{k + 1}_label"] = str(class_array[j])
            row[f"top{k + 1}_score"] = float(scores[i, j])
        rows.append(row)
    return pd.DataFrame(rows)


def save_score_matrix(path: str | Path, scores: np.ndarray, classes: Sequence[str],
                      paths: Sequence[str]) -> Path:
    """전체 [N, C] 점수 행렬 — Stage 5에서 다른 브랜치와 결합할 때 사용."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(scores, columns=list(classes))
    frame.insert(0, "path", list(paths))
    frame.to_csv(path, index=False)
    return path
