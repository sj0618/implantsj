"""Fuse 시각화 — 합친 게 실제로 도움이 됐는지 보이는 그림.

  branch_weights      브랜치 가중치 (learned면 무엇에 의존하기로 했는지)
  branch_comparison   브랜치 단독 vs 결합 vs gate 적용 후 정확도 — fuse의 실익
  openset_breakdown   closed / open 클래스별 정확도 (gate 전후 비교)
  score_distribution  최종 점수 분포 + threshold 선 (정답 vs 오답)
  final_confusion     최종 클래스 혼동행렬

그림 안 글자는 전부 영어 — 폐쇄망 서버에 한글 폰트가 없어도 깨지지 않게.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np

TEAL = "#0e7c86"
AMBER = "#b4661a"
GREY = "#8b97a1"
INK = "#101519"


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


def plot_branch_weights(weights: Dict[str, float], output_png: Path, *, mode: str = "learned") -> Path:
    plt = _plt()
    names = sorted(weights, key=lambda n: weights[n])
    values = [weights[n] for n in names]
    uniform = 1.0 / max(len(names), 1)

    fig, ax = plt.subplots(figsize=(7.0, 2.2 + 0.5 * len(names)))
    ax.barh(names, values, color=[TEAL if v >= uniform else GREY for v in values], height=0.55)
    ax.axvline(uniform, color=AMBER, linestyle="--", linewidth=1.2, label=f"uniform = {uniform:.3f}")
    ax.set(xlabel="weight", title=f"branch weights ({mode})")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.02, i, f"{v:.3f}", va="center", fontsize=9)
    return _save(fig, output_png)


def plot_branch_comparison(branch_scores: Dict[str, np.ndarray], fused: np.ndarray,
                           gated_scores: np.ndarray, true_index: np.ndarray,
                           output_png: Path) -> Path:
    """브랜치 단독 정확도와 결합 결과를 나란히. fuse가 실익이 있었는지 한눈에."""
    plt = _plt()
    entries = [(name, float((s.argmax(axis=1) == true_index).mean()))
               for name, s in sorted(branch_scores.items())]
    entries.append(("fused", float((fused.argmax(axis=1) == true_index).mean())))
    entries.append(("fused\n+ gate", float((gated_scores.argmax(axis=1) == true_index).mean())))
    labels = [e[0] for e in entries]
    values = [e[1] for e in entries]
    colors = [GREY] * (len(entries) - 2) + [TEAL, AMBER]

    fig, ax = plt.subplots(figsize=(1.5 * len(entries) + 2.5, 4.4))
    bars = ax.bar(labels, values, color=colors, width=0.6)
    best_single = max(values[:-2]) if len(values) > 2 else 0
    ax.axhline(best_single, color=GREY, linestyle=":", linewidth=1.1,
               label=f"best single branch = {best_single:.3f}")
    ax.set(ylabel="accuracy", ylim=(0, 1.05), title="does combining help?")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=9)
    return _save(fig, output_png)


def plot_openset_breakdown(fused: np.ndarray, gated_scores: np.ndarray,
                           true_index: np.ndarray, is_open: np.ndarray,
                           output_png: Path) -> Path:
    """closed / open 클래스별로 gate 전후 정확도. gate가 노린 효과를 검증한다."""
    plt = _plt()
    is_open = np.asarray(is_open, dtype=bool)
    groups = [("closed", ~is_open), ("open", is_open), ("all", np.ones_like(is_open))]
    before, after, labels = [], [], []
    for name, mask in groups:
        if not mask.any():
            continue
        labels.append(f"{name}\n(n={int(mask.sum())})")
        before.append(float((fused.argmax(axis=1) == true_index)[mask].mean()))
        after.append(float((gated_scores.argmax(axis=1) == true_index)[mask].mean()))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(2.0 * len(labels) + 3.0, 4.4))
    ax.bar(x - 0.19, before, width=0.36, color=GREY, label="before gate")
    ax.bar(x + 0.19, after, width=0.36, color=TEAL, label="after gate")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set(ylabel="accuracy", ylim=(0, 1.05), title="open-set gate effect")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i - 0.19, b + 0.015, f"{b:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.19, a + 0.015, f"{a:.3f}", ha="center", fontsize=8)
    return _save(fig, output_png)


def plot_score_distribution(best_scores: np.ndarray, correct: np.ndarray,
                            output_png: Path, *, threshold: float | None = None) -> Path:
    plt = _plt()
    correct = np.asarray(correct, dtype=bool)
    bins = np.linspace(float(best_scores.min()), float(best_scores.max()), 40)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.hist(best_scores[correct], bins=bins, alpha=0.7, color=TEAL,
            label=f"correct (n={int(correct.sum())})")
    ax.hist(best_scores[~correct], bins=bins, alpha=0.7, color=AMBER,
            label=f"wrong (n={int((~correct).sum())})")
    if threshold is not None:
        rejected = float((best_scores < threshold).mean())
        ax.axvline(threshold, color=INK, linestyle="--", linewidth=1.5,
                   label=f"threshold = {threshold:.4f}  (rejects {rejected:.1%})")
    ax.set(xlabel="fused score of the top class", ylabel="count",
           title="final score distribution")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    return _save(fig, output_png)


def plot_final_confusion(true_labels: Sequence[str], pred_labels: Sequence[str],
                         classes: Sequence[str], output_png: Path) -> Path:
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
           title=f"final confusion (excl. unknown, n={len(keep)})")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=6.5)
    return _save(fig, output_png)


def visualize_run(
    *,
    output_dir: str | Path,
    branch_scores: Dict[str, np.ndarray],
    fused: np.ndarray,
    gated_scores: np.ndarray,
    true_index: np.ndarray,
    is_open: np.ndarray,
    classes: Sequence[str],
    true_labels: Sequence[str],
    pred_labels: Sequence[str],
    weights: Dict[str, float],
    weight_mode: str = "learned",
    threshold: float | None = None,
    log=print,
    which: Sequence[str] = ("weights", "comparison", "openset", "score", "confusion"),
) -> dict:
    """fuse 결과 시각화 묶음. 개별 실패가 나머지를 막지 않는다."""
    out = Path(output_dir) / "visualization"
    out.mkdir(parents=True, exist_ok=True)
    saved: dict = {}
    best = gated_scores.max(axis=1)
    correct = np.asarray([str(p) == t for p, t in zip(pred_labels, true_labels)])

    jobs = {
        "weights": lambda: plot_branch_weights(weights, out / "branch_weights.png", mode=weight_mode),
        "comparison": lambda: plot_branch_comparison(
            branch_scores, fused, gated_scores, true_index, out / "branch_comparison.png"),
        "openset": (lambda: plot_openset_breakdown(
            fused, gated_scores, true_index, is_open, out / "openset_breakdown.png"))
            if np.asarray(is_open).any() else None,
        "score": lambda: plot_score_distribution(
            best, correct, out / "score_distribution.png", threshold=threshold),
        "confusion": lambda: plot_final_confusion(
            true_labels, pred_labels, classes, out / "final_confusion.png"),
    }
    for name in which:
        job = jobs.get(name)
        if job is None:
            continue
        try:
            saved[name] = str(job())
        except Exception as exc:
            log(f"  시각화 '{name}' 건너뜀: {type(exc).__name__}: {exc}")
    return saved
