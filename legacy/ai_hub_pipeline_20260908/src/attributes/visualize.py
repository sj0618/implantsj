"""Attributes 시각화 — feature별 성능과 최종 판정 근거를 그림으로.

  feature_performance   feature별 정확도/macro-F1 막대 — 어떤 feature가 약한지
  feature_weights       learned 모드에서 학습된 feature 가중치 (어디에 의존하는지)
  feature_confusions    feature별 혼동행렬을 한 장에 격자로
  match_count           일치 feature 수 분포 (정답 vs 오답) — min_match 기준 근거
  score_distribution    최종 클래스 점수 분포 (정답 vs 오답)
  final_confusion       spec table 매칭 후 최종 클래스 혼동행렬

그림 안 글자는 전부 영어 — 폐쇄망 서버에 한글 폰트가 없어도 깨지지 않게.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from .core import log


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    _plt().close(fig)
    return path


def plot_feature_performance(metrics: dict, output_png: Path, *, split: str = "test") -> Path:
    """feature별 accuracy / macro-F1 가로 막대. 약한 feature가 바로 보인다."""
    plt = _plt()
    per = metrics.get("per_attribute", {})
    names = [a for a, m in per.items() if "accuracy" in m]
    if not names:
        raise ValueError("평가된 feature가 없습니다")
    names = sorted(names, key=lambda a: per[a]["macro_f1"])
    accuracy = [per[a]["accuracy"] for a in names]
    macro_f1 = [per[a]["macro_f1"] for a in names]
    support = [per[a]["n"] for a in names]

    fig, ax = plt.subplots(figsize=(8.5, max(3.0, 0.42 * len(names) + 1.2)))
    y = np.arange(len(names))
    ax.barh(y - 0.2, accuracy, height=0.38, label="accuracy", color="#0e7c86")
    ax.barh(y + 0.2, macro_f1, height=0.38, label="macro F1", color="#b4661a")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a}  (n={s})" for a, s in zip(names, support)], fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("score")
    ax.set_title(f"per-feature performance ({split})")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    for i, (a, f) in enumerate(zip(accuracy, macro_f1)):
        ax.text(min(a + 0.015, 0.99), i - 0.2, f"{a:.3f}", va="center", fontsize=7.5)
        ax.text(min(f + 0.015, 0.99), i + 0.2, f"{f:.3f}", va="center", fontsize=7.5)
    return _save(fig, output_png)


def plot_feature_weights(weights: Dict[str, float], output_png: Path) -> Path:
    """learned 모드에서 학습된 feature 가중치. 균등선을 함께 그려 편중을 본다."""
    plt = _plt()
    names = sorted(weights, key=lambda a: weights[a])
    values = [weights[a] for a in names]
    uniform = 1.0 / max(len(names), 1)

    fig, ax = plt.subplots(figsize=(8.0, max(2.8, 0.4 * len(names) + 1.1)))
    colors = ["#0e7c86" if v >= uniform else "#8b97a1" for v in values]
    ax.barh(names, values, color=colors, height=0.62)
    ax.axvline(uniform, color="#b4661a", linestyle="--", linewidth=1.2,
               label=f"uniform = {uniform:.3f}")
    ax.set_xlabel("learned weight")
    ax.set_title("feature weights learned on valid")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.015, i, f"{v:.3f}", va="center", fontsize=8)
    return _save(fig, output_png)


def plot_feature_confusions(probabilities: Dict[str, np.ndarray], class_index: np.ndarray,
                            table, attributes: Sequence[str], output_png: Path) -> Path:
    """feature마다 혼동행렬을 하나의 격자에. 값이 적어 한 장에 담긴다."""
    from sklearn.metrics import confusion_matrix

    plt = _plt()
    usable = [a for a in attributes if a in probabilities]
    if not usable:
        raise ValueError("혼동행렬을 그릴 feature가 없습니다")
    columns = min(4, len(usable))
    rows = math.ceil(len(usable) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.5 * columns, 3.3 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, usable):
        ai = table.attributes.index(name)
        truth = table.target[class_index, ai]
        keep = truth >= 0
        values = table.vocab[name]
        if not keep.any():
            ax.axis("off")
            continue
        predicted = probabilities[name].argmax(axis=1)
        matrix = confusion_matrix(truth[keep], predicted[keep], labels=range(len(values)))
        ax.imshow(matrix, cmap="Blues", interpolation="nearest")
        ax.set_title(name, fontsize=10)
        ticks = range(len(values))
        short = [v[:10] for v in values]
        ax.set_xticks(ticks); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(ticks); ax.set_yticklabels(short, fontsize=7)
        threshold = matrix.max() / 2 if matrix.size else 0
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                ax.text(c, r, int(matrix[r, c]), ha="center", va="center", fontsize=7,
                        color="white" if matrix[r, c] > threshold else "black")
    for ax in axes[len(usable):]:
        ax.axis("off")
    fig.suptitle("per-feature confusion (true vs predicted value)", fontsize=12)
    fig.tight_layout()
    return _save(fig, output_png)


def plot_match_counts(match_counts: np.ndarray, correct: np.ndarray, output_png: Path,
                      *, min_match: int | None = None) -> Path:
    """예측 클래스와 일치한 feature 수 분포. min_match를 어디 둘지 근거."""
    plt = _plt()
    correct = np.asarray(correct, dtype=bool)
    top = match_counts.max(axis=1) if match_counts.ndim == 2 else np.asarray(match_counts)
    bins = np.arange(top.min(), top.max() + 2) - 0.5

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.hist(top[correct], bins=bins, alpha=0.7, color="#0e7c86",
            label=f"correct (n={int(correct.sum())})")
    ax.hist(top[~correct], bins=bins, alpha=0.7, color="#b4661a",
            label=f"wrong (n={int((~correct).sum())})")
    if min_match is not None:
        ax.axvline(min_match - 0.5, color="#101519", linestyle="--", linewidth=1.4,
                   label=f"min_match = {min_match}")
    ax.set(xlabel="matched features for the predicted class", ylabel="count",
           title="how many features agreed")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    return _save(fig, output_png)


def plot_score_distribution(best_scores: np.ndarray, correct: np.ndarray,
                            output_png: Path, *, title: str = "final score distribution") -> Path:
    plt = _plt()
    correct = np.asarray(correct, dtype=bool)
    bins = np.linspace(float(best_scores.min()), float(best_scores.max()), 40)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.hist(best_scores[correct], bins=bins, alpha=0.7, color="#0e7c86",
            label=f"correct (n={int(correct.sum())})")
    ax.hist(best_scores[~correct], bins=bins, alpha=0.7, color="#b4661a",
            label=f"wrong (n={int((~correct).sum())})")
    ax.set(xlabel="score of the predicted class", ylabel="count", title=title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    return _save(fig, output_png)


def plot_final_confusion(true_labels: Sequence[str], pred_labels: Sequence[str],
                         classes: Sequence[str], output_png: Path) -> Path:
    """spec table 매칭 후 최종 클래스 혼동행렬."""
    from sklearn.metrics import confusion_matrix

    plt = _plt()
    names = list(classes)
    index = {c: i for i, c in enumerate(names)}
    keep = [i for i, p in enumerate(pred_labels) if str(p) in index]
    if not keep:
        raise ValueError("unknown이 아닌 예측이 없습니다")
    truth = [index[true_labels[i]] for i in keep]
    predicted = [index[str(pred_labels[i])] for i in keep]
    matrix = confusion_matrix(truth, predicted, labels=range(len(names)))

    size = max(6.0, min(20.0, len(names) * 0.42))
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    image = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    ax.set(xlabel="predicted", ylabel="true",
           title=f"final class confusion (excl. unknown, n={len(keep)})")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=6.5)
    return _save(fig, output_png)


def visualize_run(
    *,
    output_dir: str | Path,
    split: str,
    metrics: dict,
    probabilities: Dict[str, np.ndarray],
    class_index: np.ndarray,
    table,
    fuse_features: Sequence[str],
    scores: np.ndarray,
    match_counts: np.ndarray,
    classes: Sequence[str],
    true_labels: Sequence[str],
    pred_labels: Sequence[str],
    weights: Dict[str, float] | None = None,
    min_match: int | None = None,
    which: Sequence[str] = ("performance", "weights", "confusions", "match", "score", "final"),
) -> dict:
    """한 split의 시각화 묶음. 개별 실패가 나머지를 막지 않는다."""
    out = Path(output_dir) / "visualization"
    out.mkdir(parents=True, exist_ok=True)
    saved: dict = {}
    correct = np.asarray([str(p) == t for p, t in zip(pred_labels, true_labels)])
    best = scores[np.arange(len(scores)), scores.argmax(axis=1)]

    jobs = {
        "performance": lambda: plot_feature_performance(
            metrics, out / f"feature_performance_{split}.png", split=split),
        "weights": (lambda: plot_feature_weights(weights, out / "feature_weights.png"))
                   if weights else None,
        "confusions": lambda: plot_feature_confusions(
            probabilities, class_index, table, fuse_features,
            out / f"feature_confusions_{split}.png"),
        "match": lambda: plot_match_counts(
            match_counts, correct, out / f"match_counts_{split}.png", min_match=min_match),
        "score": lambda: plot_score_distribution(
            best, correct, out / f"score_distribution_{split}.png",
            title=f"final score distribution ({split})"),
        "final": lambda: plot_final_confusion(
            true_labels, pred_labels, classes, out / f"final_confusion_{split}.png"),
    }
    for name in which:
        job = jobs.get(name)
        if job is None:
            continue
        try:
            saved[name] = str(job())
        except Exception as exc:
            log(f"    시각화 '{name}' 건너뜀: {type(exc).__name__}: {exc}")
    return saved
