"""브랜치 결합 + open-set gate + threshold + top-k 평가.

결합 순서:
  1) 정규화된 브랜치 점수를 가중합 → [N, C] 최종 점수
       weights.mode = manual(사용자 지정) | learned(valid에서 자동 학습)
  2) open-set gate (노션): metric이 open 클래스로 판정한 샘플은
       classifier/attributes를 반영하지 않고 metric 결과만 쓴다
  3) threshold: 최고 점수가 기준 미만이면 __unknown__ (open-set reject)
       기준값은 사용자가 정하거나(manual) valid에서 자동으로 찾는다(auto)
  4) 평가: top1(기본) 또는 top2(상위 2개 중 하나라도 맞으면 정답)
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

UNKNOWN_LABEL = "__unknown__"
WEIGHT_MODES = ("manual", "learned")
THRESHOLD_MODES = ("none", "auto", "manual")


# =====================================================================
# 1) 가중치
# =====================================================================

def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def learn_branch_weights(
    branch_scores: Dict[str, np.ndarray],
    true_class_index: np.ndarray,
    *,
    epochs: int = 400,
    lr: float = 0.05,
    verbose: bool = False,
) -> Dict[str, float]:
    """valid에서 브랜치 가중치를 학습. score = Σ_b w_b · S_b, w는 softplus로 양수 제약.

    파라미터가 브랜치 수(2~3개)뿐이라 numpy Adam으로 충분하다.
    """
    names = list(branch_scores)
    stacked = np.stack([branch_scores[n] for n in names], axis=2)   # [N, C, B]
    y = np.asarray(true_class_index, dtype=np.int64)
    n, num_classes, num_branches = stacked.shape

    raw = np.zeros(num_branches)
    m_state = np.zeros(num_branches)
    v_state = np.zeros(num_branches)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    onehot = np.zeros((n, num_classes))
    onehot[np.arange(n), y] = 1.0

    for step in range(1, epochs + 1):
        weights = _softplus(raw)
        scores = stacked @ weights
        scores -= scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        grad_scores = (probabilities - onehot) / n
        grad_w = np.einsum("nc,ncb->b", grad_scores, stacked)
        grad_raw = grad_w * _sigmoid(raw)

        m_state = beta1 * m_state + (1 - beta1) * grad_raw
        v_state = beta2 * v_state + (1 - beta2) * grad_raw ** 2
        raw -= lr * (m_state / (1 - beta1 ** step)) / (np.sqrt(v_state / (1 - beta2 ** step)) + eps)

        if verbose and step % max(epochs // 4, 1) == 0:
            loss = -np.log(np.maximum(probabilities[np.arange(n), y], 1e-12)).mean()
            print(f"    branch weight fit step={step} loss={loss:.4f}")

    weights = _softplus(raw)
    total = weights.sum()
    weights = weights / total if total > 0 else np.ones(num_branches) / num_branches
    return {name: float(w) for name, w in zip(names, weights)}


def combine(branch_scores: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    """가중합 → [N, C]."""
    missing = [n for n in branch_scores if n not in weights]
    if missing:
        raise ValueError(f"가중치가 없는 브랜치: {missing}")
    total = sum(weights[n] for n in branch_scores)
    if total <= 0:
        raise ValueError("브랜치 가중치 합이 0입니다")
    out = None
    for name, scores in branch_scores.items():
        term = (weights[name] / total) * scores
        out = term if out is None else out + term
    return out


# =====================================================================
# 2) open-set gate — metric이 open 판정하면 metric만 사용
# =====================================================================

def apply_openset_gate(
    fused: np.ndarray,
    metric_scores: np.ndarray,
    classes: Sequence[str],
    open_classes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """metric의 최근접이 open 클래스인 샘플은 metric 점수로 대체.

    노션 규칙: "metric learning 결과 open-set 클래스로 분류되었다면
    vit classifier, attributes 모델은 최종결과에 미반영".

    반환: (게이트 적용된 점수, 게이트가 걸린 샘플 마스크)
    """
    open_set = set(open_classes)
    if not open_set:
        return fused, np.zeros(len(fused), dtype=bool)
    is_open_class = np.asarray([c in open_set for c in classes])
    metric_choice = metric_scores.argmax(axis=1)
    gated = is_open_class[metric_choice]                  # [N]

    out = fused.copy()
    out[gated] = metric_scores[gated]                     # 다른 브랜치를 무시
    return out, gated


# =====================================================================
# 3) threshold — 기준 미만이면 unknown
# =====================================================================

def find_threshold_auto(scores: np.ndarray, true_class_index: np.ndarray,
                        *, steps: int = 200, max_reject_rate: float | None = None) -> float:
    """valid에서 자동으로 임계값을 찾는다.

    최고 점수가 임계 이상인 것만 판정하고 나머지는 보류(unknown)했을 때,
    "맞은 것은 통과시키고 틀린 것은 걸러내는" 능력(Youden's J = TPR - FPR)이
    가장 큰 지점을 고른다. 정답/오답 점수 분포의 최적 분리점에 해당한다.

    max_reject_rate를 주면 그 비율보다 많이 버리는 임계값은 후보에서 뺀다.
    (브랜치 성능이 낮으면 J가 과하게 공격적인 값을 고를 수 있으므로 안전장치)
    """
    best = scores.max(axis=1)
    correct = scores.argmax(axis=1) == np.asarray(true_class_index)
    if correct.all() or not correct.any():
        # 한쪽 분포가 비면 분리점을 정의할 수 없다 → 전부 통과시키는 값
        return float(best.min())

    candidates = np.linspace(best.min(), best.max(), steps)
    if max_reject_rate is not None:
        allowed = [t for t in candidates if (best < t).mean() <= max_reject_rate]
        if allowed:
            candidates = np.asarray(allowed)
        else:
            return float(best.min())          # 어떤 값도 상한을 못 지키면 reject 안 함

    positives, negatives = best[correct], best[~correct]
    scores_j = [
        (positives >= t).mean() - (negatives >= t).mean() for t in candidates
    ]
    return float(candidates[int(np.argmax(scores_j))])


def apply_threshold(scores: np.ndarray, threshold: float | None) -> np.ndarray:
    """최고 점수가 threshold 미만인 샘플 마스크 (= unknown으로 뺄 대상)."""
    if threshold is None:
        return np.zeros(len(scores), dtype=bool)
    return scores.max(axis=1) < threshold


# =====================================================================
# 3b) open-set 거부 성능 (threshold와 무관한 지표)
# =====================================================================

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
    return float((known >= np.quantile(unknown, 1 - target_tpr)).mean())


def evaluate_rejection_quality(scores: np.ndarray, is_unknown: np.ndarray,
                               *, which: Sequence[str] = ("min_dist", "margin", "dist_ent"),
                               temperature: float = 1.0) -> dict:
    """최종 점수로 open 클래스를 걸러낼 수 있는지 AUROC/FPR@95로 잰다.

    threshold를 정하지 않고 순위만으로 재므로 설정끼리 비교하기에 낫다.
    fuse의 scores는 '클수록 좋음'이라 거리(작을수록 가까움)로 뒤집어 계산한다.
    출처: implantys "metric learning" loo_eval.py / diagnostics.py
    """
    distances = -np.asarray(scores, dtype=np.float64)
    is_unknown = np.asarray(is_unknown, dtype=bool)
    if not is_unknown.any() or is_unknown.all():
        return {"available": False,
                "reason": "known 또는 unknown 한쪽이 비어 거부 성능을 잴 수 없습니다",
                "n_known": int((~is_unknown).sum()), "n_unknown": int(is_unknown.sum())}
    if distances.shape[1] < 2:
        return {"available": False, "reason": "클래스가 2개 미만입니다"}

    ordered = np.sort(distances, axis=1)
    shifted = -distances / max(temperature, 1e-8)
    shifted -= shifted.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    entropy = -(probabilities * np.log(probabilities + 1e-12)).sum(axis=1)

    # 값이 클수록 unknown으로 통일
    candidates = {
        "min_dist": ordered[:, 0],
        "margin": -(ordered[:, 1] - ordered[:, 0]),
        "dist_ent": entropy,
        "logit_msp": -probabilities.max(axis=1),
        "logit_ent": entropy,
    }

    result = {"available": True,
              "n_known": int((~is_unknown).sum()), "n_unknown": int(is_unknown.sum()),
              "scores": {}}
    for name in which:
        if name not in candidates:
            raise ValueError(f"reject score는 {REJECT_SCORES} 중 하나여야 합니다: {name}")
        value = candidates[name]
        result["scores"][name] = {
            "auroc": round(auroc(value[is_unknown], value[~is_unknown]), 6),
            "fpr_at_95tpr": round(fpr_at_tpr(value[is_unknown], value[~is_unknown]), 6),
        }
    best = max(result["scores"], key=lambda k: result["scores"][k]["auroc"])
    result["best_score"] = best
    result["best_auroc"] = result["scores"][best]["auroc"]
    return result


# =====================================================================
# 4) top-k 평가
# =====================================================================

def evaluate_topk(
    scores: np.ndarray,
    classes: Sequence[str],
    true_labels: Sequence[str],
    rejected: np.ndarray,
    *,
    top_k: int = 1,
) -> dict:
    """top1(기본) 또는 top2 기준 성능.

    top1: 최종 1개가 정답이어야 맞음
    top2: 점수 상위 2개 중 하나라도 정답이면 맞음
    reject된 샘플은 항상 오답으로 센다(판정을 포기한 것이므로).
    별도로 '판정된 것만'의 정확도도 함께 낸다.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    class_array = np.asarray(classes, dtype=object)
    truth = np.asarray(true_labels, dtype=object)
    k = min(top_k, scores.shape[1])
    top_index = np.argsort(-scores, axis=1)[:, :k]
    top_labels = class_array[top_index]                       # [N, k]

    hit = (top_labels == truth[:, None]).any(axis=1) & ~rejected
    decided = ~rejected
    known = np.isin(truth, class_array)                       # 갤러리에 있는 클래스인지

    return {
        "top_k": k,
        "n": int(len(truth)),
        "accuracy": float(hit.mean()),
        "n_decided": int(decided.sum()),
        "accuracy_on_decided": float(hit[decided].mean()) if decided.any() else float("nan"),
        "reject_rate": float(rejected.mean()),
        # reject가 제 역할을 하는지: 원래 틀렸을 샘플을 얼마나 걸러냈나
        "reject_precision": float(
            (~(top_labels[rejected] == truth[rejected, None]).any(axis=1)).mean()
        ) if rejected.any() else float("nan"),
        "coverage_known_classes": float(known.mean()),
    }


def build_prediction_frame(
    paths: Sequence[str],
    scores: np.ndarray,
    classes: Sequence[str],
    true_labels: Sequence[str],
    rejected: np.ndarray,
    gated: np.ndarray,
    *,
    top_k: int = 2,
) -> pd.DataFrame:
    """샘플별 최종 판정 + 상위 후보 점수."""
    class_array = np.asarray(classes, dtype=object)
    order = np.argsort(-scores, axis=1)
    rows = []
    for i in range(len(paths)):
        best = int(order[i, 0])
        row = {
            "path": paths[i],
            "true_label": true_labels[i],
            "pred_label": UNKNOWN_LABEL if rejected[i] else str(class_array[best]),
            "score": float(scores[i, best]),
            "rejected_unknown": bool(rejected[i]),
            "openset_gated": bool(gated[i]),
            "correct_top1": (not rejected[i]) and str(class_array[best]) == true_labels[i],
        }
        for k in range(1, min(top_k, scores.shape[1])):
            j = int(order[i, k])
            row[f"top{k + 1}_label"] = str(class_array[j])
            row[f"top{k + 1}_score"] = float(scores[i, j])
        if scores.shape[1] >= 2:
            top2 = class_array[order[i, :2]]
            row["correct_top2"] = (not rejected[i]) and bool((top2 == true_labels[i]).any())
        rows.append(row)
    return pd.DataFrame(rows)
