"""Prototype 구성 — 클래스별 대표 벡터 [k, D]를 만든다 (yaml 선택).

  mean         클래스 전체 평균 (기본)
               spherical_mean=True면 구면 평균 — cosine 거리를 쓸 때 더 맞다
  ss           클래스별 support n장 랜덤 샘플의 평균 (class-balanced 효과)
  medoid       같은 클래스 샘플들과의 거리 합이 최소인 실제 샘플
  kmeans_multi 클래스별 k-means로 prototype 여러 개 (query는 최소거리로 판정)
  reference    따로 준비한 **대표 이미지**의 임베딩을 그대로 prototype으로 사용
               (제조사 카탈로그 이미지처럼 그 시스템을 가장 잘 보여주는 사진)

거리 계산은 scoring.py가 담당한다.
출처: implantsj src/evaluation.py, pipelines/eval/evaluate_metric_diagnosis.py,
      tools/improve_unseen_mahalanobis.py (multi_prototype)
"""

from __future__ import annotations

import hashlib
from typing import Dict, Sequence

import numpy as np

PROTOTYPE_METHODS = ("mean", "ss", "medoid", "kmeans_multi", "reference")


def _label_rng(seed: int, label: str) -> np.random.Generator:
    label_seed = int(hashlib.sha256(f"{seed}:{label}".encode()).hexdigest()[:16], 16)
    return np.random.default_rng(label_seed)


def _mean(block: np.ndarray, spherical: bool) -> np.ndarray:
    """평균 prototype. spherical이면 구면 평균(정규화 → 평균 → 재정규화).

    그냥 산술평균을 내면 노름이 큰 샘플이 방향을 독점한다. cosine 거리는
    방향만 보므로, 각 벡터를 단위길이로 맞춘 뒤 평균 내는 것이 맞다.
    출처: implantys "metric learning" prototype.py
    """
    if not spherical:
        return block.mean(axis=0, keepdims=True)
    unit = block / np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-12)
    center = unit.mean(axis=0, keepdims=True)
    return center / np.maximum(np.linalg.norm(center, axis=1, keepdims=True), 1e-12)


def build_prototypes(
    features: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[str],
    method: str = "mean",
    *,
    seed: int = 42,
    ss_count: int = 5,
    kmeans_clusters: int = 3,
    reference: Dict[str, np.ndarray] | None = None,
    reference_reduce: str = "mean",
    spherical_mean: bool = False,
    log=print,
) -> Dict[str, np.ndarray]:
    """클래스 → [n_protos, D] 매핑.

    method="reference"면 reference[label]의 임베딩을 prototype으로 쓴다.
    대표 이미지가 없는 클래스는 갤러리 평균으로 대체하고 그 사실을 로그에 남긴다
    (조용히 빠지면 그 클래스만 다른 기준으로 판정되는 걸 눈치채지 못한다).
    """
    if method not in PROTOTYPE_METHODS:
        raise ValueError(f"prototype method must be one of {PROTOTYPE_METHODS}")
    if method == "reference" and not reference:
        raise ValueError(
            "prototype method가 reference인데 대표 이미지 임베딩이 없습니다. "
            "eval.prototype.reference_dir 을 지정하세요."
        )
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels).astype(str)
    protos: Dict[str, np.ndarray] = {}
    fell_back: list[str] = []

    for label in classes:
        idx = np.flatnonzero(labels == label)
        if len(idx) == 0:
            raise ValueError(f"prototype source has no samples for label {label!r}")
        member = features[idx]

        if method == "reference":
            block = reference.get(label)
            if block is None or len(block) == 0:
                fell_back.append(label)                    # 대표 이미지 없음 → 갤러리 평균
                protos[label] = _mean(member, spherical_mean)
            elif reference_reduce == "each":
                protos[label] = np.asarray(block, dtype=np.float64)   # 여러 장을 각각 prototype으로
            else:
                protos[label] = _mean(np.asarray(block, dtype=np.float64), spherical_mean)
        elif method == "mean":
            protos[label] = _mean(member, spherical_mean)
        elif method == "ss":
            rng = _label_rng(seed, label)
            chosen = rng.choice(len(member), min(ss_count, len(member)), replace=False)
            protos[label] = _mean(member[chosen], spherical_mean)
        elif method == "medoid":
            dist = np.linalg.norm(member[:, None, :] - member[None, :, :], axis=2)
            protos[label] = member[dist.sum(axis=1).argmin()][None, :]
        else:  # kmeans_multi
            from sklearn.cluster import KMeans

            k = min(kmeans_clusters, len(member))
            centers = (
                KMeans(n_clusters=k, random_state=seed, n_init=10).fit(member).cluster_centers_
                if k > 1
                else member.mean(axis=0, keepdims=True)
            )
            protos[label] = np.asarray(centers, dtype=np.float64)

    if fell_back:
        log(f"    대표 이미지가 없어 갤러리 평균으로 대체한 클래스 {len(fell_back)}개: {fell_back}")
    return protos
