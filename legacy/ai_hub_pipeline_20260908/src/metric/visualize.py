"""Metric learning 시각화 — 임베딩이 어떻게 배치됐는지 눈으로 확인.

  embedding    쿼리 임베딩 t-SNE / UMAP (클래스별 색, closed/open 구분)
  prototype    갤러리 prototype을 임베딩과 함께 그려 open-set 배치를 본다
  score        정답/오답 score 분포 (threshold를 어디에 둘지 판단 근거)

closed27_open15에서는 open 클래스를 다른 마커로 그려,
"학습에 없던 클래스가 임베딩 공간에서 따로 뭉치는가"를 바로 볼 수 있다.

출처: implantsj pipelines/visualize/visualize_metric_tsne.py,
      visualize_umap_from_tsne.py, evaluate_metric_diagnosis.py의 prototype t-SNE
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .core import log

REDUCERS = ("tsne", "umap", "none")


def stratified_subsample(labels: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    """클래스 비율을 유지하며 표본 축소 (t-SNE는 샘플이 많으면 매우 느리다)."""
    if max_samples <= 0 or len(labels) <= max_samples:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = max(1, math.ceil(max_samples / len(classes)))
    selected = []
    for c in classes:
        idx = np.flatnonzero(labels == c)
        selected.append(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
    out = np.concatenate(selected)
    if len(out) > max_samples:
        out = rng.choice(out, size=max_samples, replace=False)
    return np.sort(out)


def reduce_2d(features: np.ndarray, method: str = "tsne", *, seed: int = 42,
              pca_dim: int = 50, perplexity: float = 30.0,
              n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """고차원 임베딩 → 2D. 먼저 PCA로 줄인 뒤 t-SNE/UMAP을 돌린다(속도·안정성)."""
    if method not in REDUCERS:
        raise ValueError(f"reducer must be one of {REDUCERS}")
    features = np.asarray(features, dtype=np.float64)
    if method == "none":
        return features[:, :2]
    if len(features) < 3:
        raise ValueError("2D 축소에는 최소 3개 샘플이 필요합니다")

    from sklearn.decomposition import PCA

    x = features
    if pca_dim > 0 and x.shape[1] > pca_dim:
        x = PCA(n_components=min(pca_dim, len(x) - 1), random_state=seed).fit_transform(x)

    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise RuntimeError(
                "UMAP을 쓰려면 umap-learn이 필요합니다. "
                "설치가 어려우면 reducer: tsne 를 쓰세요."
            ) from exc
        reducer = umap.UMAP(
            n_components=2, random_state=seed,
            n_neighbors=max(2, min(int(n_neighbors), x.shape[0] - 1)),
            min_dist=float(min_dist),
        )
        return np.asarray(reducer.fit_transform(x))

    from sklearn.manifold import TSNE

    safe_perplexity = min(perplexity, max(1.0, (len(x) - 1) / 3.0))
    return np.asarray(TSNE(
        n_components=2, perplexity=safe_perplexity, init="pca",
        learning_rate="auto", metric="euclidean", random_state=seed,
    ).fit_transform(x))


def _figure(width: float = 10.0, height: float = 8.0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt, plt.subplots(figsize=(width, height))


# prototype 종류별 마커 — GT(대표 이미지)를 한눈에 구분하기 위해
# 마커 기호 → 제목/범례에 쓸 이름
MARKER_NAMES = {"*": "star", "P": "plus", "X": "cross", "D": "diamond", "s": "square"}

PROTOTYPE_STYLE = {
    "reference": ("*", 300, "GT / reference"),
    "mean": ("P", 150, "mean"),
    "ss": ("X", 150, "ss"),
    "medoid": ("D", 110, "medoid"),
    "kmeans_multi": ("s", 110, "kmeans"),
}


def plot_embedding(
    coords: np.ndarray,
    labels: Sequence[str],
    output_png: str | Path,
    *,
    open_classes: Sequence[str] = (),
    title: str = "metric embedding",
    prototype_coords: np.ndarray | None = None,
    prototype_labels: Sequence[str] | None = None,
    prototype_method: str = "mean",
    prototype_is_reference: Sequence[bool] | None = None,
    max_legend: int = 30,
) -> Path:
    """2D 임베딩 산점도. closed는 동그라미, open은 세모, prototype은 방법별 마커.

    prototype_is_reference를 주면 대표 이미지에서 온 prototype과
    갤러리 평균으로 대체된 것을 다른 마커로 구분한다.
    """
    plt, (fig, ax) = _figure()
    labels = np.asarray(labels, dtype=object)
    classes = sorted(set(labels.tolist()))
    open_set = set(open_classes)
    cmap = plt.get_cmap("tab20")

    for index, name in enumerate(classes):
        mask = labels == name
        is_open = name in open_set
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=26, alpha=0.75, color=cmap(index % 20),
            marker="^" if is_open else "o",
            edgecolors="black" if is_open else "white",
            linewidths=0.5 if is_open else 0.25,
            label=f"{name}{' (open)' if is_open else ''}" if index < max_legend else None,
        )

    if prototype_coords is not None and prototype_labels is not None:
        prototype_labels = np.asarray(prototype_labels, dtype=object)
        marker, size, _ = PROTOTYPE_STYLE.get(prototype_method, ("*", 260, prototype_method))
        is_ref = (np.asarray(prototype_is_reference, dtype=bool)
                  if prototype_is_reference is not None else None)
        for index, name in enumerate(classes):
            mask = prototype_labels == name
            if not mask.any():
                continue
            if is_ref is None:
                groups = [(mask, marker, size, 1.0)]
            else:
                # 대표 이미지에서 온 것 vs 갤러리 평균으로 대체된 것
                groups = [(mask & is_ref, "*", 300, 1.2),
                          (mask & ~is_ref, "P", 130, 0.8)]
            for group, mk, sz, lw in groups:
                if not group.any():
                    continue
                ax.scatter(
                    prototype_coords[group, 0], prototype_coords[group, 1],
                    s=sz, marker=mk, color=cmap(index % 20),
                    edgecolors="black", linewidths=lw, zorder=5,
                )

    ax.set(title=title, xlabel="dim 1", ylabel="dim 2")
    ax.grid(True, alpha=0.25)
    if len(classes) <= max_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)
    else:
        proto_marker = PROTOTYPE_STYLE.get(prototype_method, ("*", 260, prototype_method))[0]
        marker_note = ("star = GT prototype\nplus = fallback (gallery mean)"
                       if prototype_is_reference is not None
                       else f"{MARKER_NAMES.get(proto_marker, 'marker')} = prototype")
        ax.text(1.02, 0.5,
                f"{len(classes)} classes\n(legend omitted)\ntriangle = open class\n{marker_note}",
                transform=ax.transAxes, fontsize=9, va="center")
    fig.tight_layout()
    output = Path(output_png)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_score_distribution(
    best_scores: np.ndarray,
    correct: np.ndarray,
    output_png: str | Path,
    *,
    title: str = "score distribution",
    threshold: float | None = None,
) -> Path:
    """정답/오답의 best score 분포. Stage 5에서 threshold를 어디에 둘지 근거가 된다."""
    plt, (fig, ax) = _figure(9.0, 5.0)
    correct = np.asarray(correct, dtype=bool)
    bins = np.linspace(float(best_scores.min()), float(best_scores.max()), 40)
    ax.hist(best_scores[correct], bins=bins, alpha=0.65, label=f"correct (n={int(correct.sum())})")
    ax.hist(best_scores[~correct], bins=bins, alpha=0.65, label=f"wrong (n={int((~correct).sum())})")
    if threshold is not None:
        ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5,
                   label=f"threshold={threshold:.4f}")
    ax.set(title=title, xlabel="best score", ylabel="count")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output = Path(output_png)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def visualize_run(
    *,
    query_features: np.ndarray,
    query_labels: Sequence[str],
    query_paths: Sequence[str],
    protos: Dict[str, np.ndarray] | None,
    classes: Sequence[str],
    open_classes: Sequence[str],
    output_dir: str | Path,
    tag: str,
    prototype_method: str = "mean",
    reference_labels: Sequence[str] = (),
    reducer: str = "tsne",
    max_samples: int = 2000,
    seed: int = 42,
    pca_dim: int = 50,
    perplexity: float = 30.0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    best_scores: np.ndarray | None = None,
    correct: np.ndarray | None = None,
) -> dict:
    """한 조합(prototype × distance)의 임베딩/점수 시각화를 만든다.

    prototype을 같은 2D 공간에 놓기 위해 임베딩과 함께 축소한 뒤 다시 분리한다
    (따로 축소하면 좌표계가 달라 겹쳐 그릴 수 없다).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict = {}

    labels = np.asarray(query_labels, dtype=object)
    keep = stratified_subsample(labels, max_samples, seed)
    features = np.asarray(query_features)[keep]
    labels = labels[keep]
    paths = [str(query_paths[i]) for i in keep]
    log(f"    시각화: {len(features)}장 (전체 {len(query_labels)}장에서 표본) reducer={reducer}")

    # prototype을 함께 축소해 같은 좌표계에 놓는다
    prototype_matrix, prototype_labels, prototype_is_ref = None, None, None
    if protos:
        stacked, names = [], []
        for name in classes:
            block = np.asarray(protos[name])
            stacked.append(block)
            names.extend([name] * len(block))
        prototype_matrix = np.concatenate(stacked, axis=0)
        prototype_labels = np.asarray(names, dtype=object)
        if prototype_method == "reference" and reference_labels:
            have = set(reference_labels)
            prototype_is_ref = np.asarray([n in have for n in names])
        combined = np.concatenate([features, prototype_matrix], axis=0)
    else:
        combined = features

    coords = reduce_2d(combined, reducer, seed=seed, pca_dim=pca_dim,
                       perplexity=perplexity, n_neighbors=n_neighbors, min_dist=min_dist)
    query_coords = coords[:len(features)]
    prototype_coords = coords[len(features):] if prototype_matrix is not None else None

    png = plot_embedding(
        query_coords, labels, output_dir / f"embedding_{reducer}_{tag}.png",
        open_classes=open_classes,
        # prototype 마커는 방식마다 다르므로(PROTOTYPE_STYLE) 제목에도 실제 이름을 쓴다
        title=f"{tag} - {reducer.upper()} (triangle=open, "
              f"{MARKER_NAMES.get(PROTOTYPE_STYLE.get(prototype_method, ('*',))[0], 'marker')}"
              f"={prototype_method} prototype)"
              + (" | GT prototypes marked" if prototype_is_ref is not None else ""),
        prototype_coords=prototype_coords, prototype_labels=prototype_labels,
        prototype_method=prototype_method, prototype_is_reference=prototype_is_ref,
    )
    saved["embedding_png"] = str(png)

    frame = pd.DataFrame({
        "path": paths, "label": labels,
        "is_open_class": [l in set(open_classes) for l in labels],
        "x": query_coords[:, 0], "y": query_coords[:, 1],
    })
    csv = output_dir / f"embedding_{reducer}_{tag}.csv"
    frame.to_csv(csv, index=False)
    saved["embedding_csv"] = str(csv)

    if best_scores is not None and correct is not None:
        png = plot_score_distribution(
            np.asarray(best_scores), np.asarray(correct),
            output_dir / f"score_distribution_{tag}.png",
            title=f"{tag} - score distribution (correct vs wrong)",
        )
        saved["score_distribution_png"] = str(png)
    return saved
