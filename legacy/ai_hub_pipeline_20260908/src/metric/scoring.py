"""임베딩 변환 + 거리 채점 — 전부 yaml 옵션.

[1] 임베딩 변환 (eval.transform.mode)
      none    변환 없음
      l2norm  L2 정규화 (spectral 계열은 내부에서 항상 정규화함)
      pca     PCA 차원 축소 후 거리 계산 (pca_dim, pca_balanced_fit)

[2] 거리 (eval.distances)
      cosine          1 - cos
      euclidean       L2
      mahalanobis     pooled within-class 공분산 기반 (아래 [3]에서 추정 방식 선택)
      fractional      고유값 스펙트럼 가중 거리 (weights = (λ/λ̄)^-alpha)
      centered_angular 전역 중심 제거 + 스펙트럼 가중 후 코사인

[3] mahalanobis 공분산 추정 (eval.mahalanobis.covariance)
      ledoit_wolf     Ledoit-Wolf shrinkage (기본)
      oas             Oracle Approximating Shrinkage
      diagonal        대각 공분산만
      class_balanced  클래스별 샘플 수를 맞춰 샘플링 후 Ledoit-Wolf
      rda             pooled ↔ 클래스별 공분산 블렌딩 + logdet 보정 (클래스별 precision)

모든 score는 "작을수록 가까움"으로 통일.
출처: implantsj tools/{evaluate_unseen,improve_unseen}_mahalanobis.py,
      tools/evaluate_covaware_target_pool_5shot.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np

DISTANCES = ("cosine", "euclidean", "mahalanobis", "fractional", "centered_angular")
COVARIANCE_ESTIMATORS = ("ledoit_wolf", "oas", "diagonal", "class_balanced", "rda")
TRANSFORMS = ("none", "l2norm", "pca")


def normalize_rows(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    return features / np.maximum(
        np.linalg.norm(features, axis=1, keepdims=True), np.finfo(np.float64).eps
    )


def _present_classes(labels: np.ndarray, classes: Sequence[str]) -> list:
    """fit 데이터에 실제로 등장하는 클래스만.

    closed27_open15에서는 변환/공분산을 closed 클래스로만 fit하는데 classes에는
    open 클래스도 들어 있다. 그대로 세면 표본 수가 0인 클래스가 섞여
    balanced 표본이 통째로 비거나 평균이 nan이 된다.
    """
    present = [c for c in classes if int((labels == c).sum()) > 0]
    if not present:
        raise ValueError("fit 데이터에 classes 중 어떤 클래스도 없습니다")
    return present


def _class_means(features: np.ndarray, labels: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    """classes 순서대로 클래스 평균. 표본이 없는 클래스는 전체 평균으로 대신한다.

    반환 배열은 항상 classes와 같은 길이라 인덱스가 어긋나지 않는다.
    (closed27_open15에서는 open 클래스가 fit 데이터에 없다 — 그대로 두면 nan이 된다)
    """
    _present_classes(labels, classes)          # 하나도 없으면 여기서 중단
    fallback = features.mean(axis=0)
    return np.stack([features[labels == c].mean(axis=0) if (labels == c).any() else fallback
                     for c in classes])


def _balanced_indices(labels: np.ndarray, classes: Sequence[str], seed: int) -> np.ndarray:
    """클래스마다 같은 장수를 뽑는다 (많은 클래스가 통계를 지배하지 않도록)."""
    rng = np.random.default_rng(seed)
    present = _present_classes(labels, classes)
    n = min(int((labels == c).sum()) for c in present)
    return np.concatenate([
        rng.choice(np.flatnonzero(labels == c), n, replace=False) for c in present
    ])


# =====================================================================
# [1] 임베딩 변환
# =====================================================================

@dataclass
class EmbeddingTransform:
    """fit 데이터로 변환을 학습하고 gallery/query에 동일하게 적용."""

    mode: str = "none"
    pca_dim: int = 64
    pca_balanced_fit: bool = False
    seed: int = 42
    _pca: object | None = field(default=None, repr=False)

    def fit(self, features: np.ndarray, labels: np.ndarray, classes: Sequence[str]) -> "EmbeddingTransform":
        if self.mode not in TRANSFORMS:
            raise ValueError(f"transform mode must be one of {TRANSFORMS}")
        if self.mode == "pca":
            from sklearn.decomposition import PCA

            fit_x = features
            if self.pca_balanced_fit:
                fit_x = features[_balanced_indices(labels, classes, self.seed)]
            dim = min(self.pca_dim, min(fit_x.shape))
            self._pca = PCA(n_components=dim, svd_solver="randomized", random_state=self.seed).fit(fit_x)
        return self

    def apply(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if self.mode == "l2norm":
            return normalize_rows(features)
        if self.mode == "pca":
            if self._pca is None:
                raise RuntimeError("PCA transform이 fit되지 않았습니다")
            return np.asarray(self._pca.transform(features), dtype=np.float64)
        return features

    def describe(self) -> dict:
        out = {"mode": self.mode}
        if self.mode == "pca":
            out.update({"pca_dim": self.pca_dim, "pca_balanced_fit": self.pca_balanced_fit})
        return out


# =====================================================================
# [2][3] 거리 채점기
# =====================================================================

@dataclass
class Scorer:
    """거리 파라미터를 fit 데이터(=학습 라벨 임베딩)로 추정하고 query를 채점."""

    distance: str = "cosine"
    covariance: str = "ledoit_wolf"
    rda_class_weight: float = 0.75      # 0=pooled만, 1=클래스별 공분산만
    rda_logdet_weight: float = 1.0
    spectral_alpha: float = 0.5
    seed: int = 42
    # fit 결과
    _precision: np.ndarray | None = field(default=None, repr=False)   # [D,D] 또는 [C,D,D]
    _logdets: np.ndarray | None = field(default=None, repr=False)     # rda용 [C]
    _eigvecs: np.ndarray | None = field(default=None, repr=False)
    _eigvals: np.ndarray | None = field(default=None, repr=False)
    _mean_eigval: float = 0.0
    _global_center: np.ndarray | None = field(default=None, repr=False)
    _classes: list[str] = field(default_factory=list, repr=False)

    def fit(self, features: np.ndarray, labels: np.ndarray, classes: Sequence[str]) -> "Scorer":
        if self.distance not in DISTANCES:
            raise ValueError(f"distance must be one of {DISTANCES}")
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels).astype(str)
        self._classes = list(classes)

        if self.distance == "mahalanobis":
            self._fit_mahalanobis(features, labels, classes)
        elif self.distance in ("fractional", "centered_angular"):
            self._fit_spectral(features, labels, classes)
        return self

    # ---- mahalanobis 공분산 추정 ----
    def _fit_mahalanobis(self, features, labels, classes) -> None:
        if self.covariance not in COVARIANCE_ESTIMATORS:
            raise ValueError(f"covariance must be one of {COVARIANCE_ESTIMATORS}")
        means = _class_means(features, labels, classes)
        index = {c: i for i, c in enumerate(classes)}
        own = np.asarray([index[c] for c in labels])
        residuals = features - means[own]

        if self.covariance == "diagonal":
            variance = residuals.var(axis=0, ddof=1)
            self._precision = np.diag(1.0 / np.maximum(variance, np.median(variance) * 1e-4))
            return
        if self.covariance == "class_balanced":
            from sklearn.covariance import LedoitWolf

            chosen = _balanced_indices(labels, classes, self.seed)
            self._precision = LedoitWolf().fit(residuals[chosen]).precision_.astype(np.float64)
            return
        if self.covariance == "oas":
            from sklearn.covariance import OAS

            self._precision = OAS().fit(residuals).precision_.astype(np.float64)
            return

        from sklearn.covariance import LedoitWolf

        pooled = LedoitWolf().fit(residuals).covariance_.astype(np.float64)
        if self.covariance == "ledoit_wolf":
            self._precision = np.linalg.inv(pooled)
            return

        # rda: pooled ↔ 클래스별 공분산 블렌딩 + logdet 보정
        w = float(self.rda_class_weight)
        if not 0.0 <= w <= 1.0:
            raise ValueError("rda.class_weight must be in [0, 1]")
        # 표본이 없는 클래스(open)는 pooled 공분산을 그대로 쓴다 — 추정할 근거가 없다
        class_covs = np.stack([
            LedoitWolf().fit(features[labels == c] - means[i]).covariance_
            if int((labels == c).sum()) >= 2 else pooled
            for i, c in enumerate(classes)
        ]).astype(np.float64)
        covs = (1.0 - w) * pooled[None] + w * class_covs
        scales = np.trace(covs, axis1=1, axis2=2) / covs.shape[1]
        covs += np.eye(covs.shape[1])[None] * np.maximum(scales * 1e-6, 1e-12)[:, None, None]
        self._precision = np.stack([np.linalg.inv(c) for c in covs])
        self._logdets = np.asarray([
            np.log(np.maximum(np.linalg.eigvalsh(c), np.finfo(float).tiny)).sum() for c in covs
        ])

    # ---- spectral (fractional / centered_angular) ----
    def _fit_spectral(self, features, labels, classes) -> None:
        from sklearn.covariance import LedoitWolf

        normalized = normalize_rows(features)
        means = _class_means(normalized, labels, classes)
        index = {c: i for i, c in enumerate(classes)}
        own = np.asarray([index[c] for c in labels])
        cov = LedoitWolf().fit(normalized - means[own]).covariance_
        eigvals, eigvecs = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
        scale = max(float(np.maximum(eigvals, 0.0).mean()), np.finfo(np.float64).tiny)
        self._eigvals = np.maximum(eigvals, scale * 1e-6)
        self._eigvecs = eigvecs
        self._mean_eigval = float(self._eigvals.mean())
        self._global_center = normalized.mean(axis=0)

    # ---- 채점: [N, C] (작을수록 가까움) ----
    def score(self, query: np.ndarray, protos: Dict[str, np.ndarray]) -> np.ndarray:
        query = np.asarray(query, dtype=np.float64)
        columns = []
        for class_index, label in enumerate(self._classes):
            centers = np.asarray(protos[label], dtype=np.float64)   # [k, D]
            scores = self._score_one(query, centers, class_index)
            columns.append(scores.min(axis=1))                      # multi-prototype: 최소거리
        return np.stack(columns, axis=1)

    def _score_one(self, query: np.ndarray, centers: np.ndarray, class_index: int) -> np.ndarray:
        if self.distance == "cosine":
            return 1.0 - normalize_rows(query) @ normalize_rows(centers).T

        if self.distance == "euclidean":
            return np.linalg.norm(query[:, None, :] - centers[None, :, :], axis=2)

        if self.distance == "mahalanobis":
            delta = query[:, None, :] - centers[None, :, :]
            precision = self._precision
            if precision.ndim == 3:                                  # rda: 클래스별
                precision = precision[class_index]
            squared = np.maximum(np.einsum("nkd,de,nke->nk", delta, precision, delta), 0.0)
            scores = np.sqrt(squared)
            if self._logdets is not None:                            # rda logdet 보정
                scores = scores + self.rda_logdet_weight * self._logdets[class_index]
            return scores

        if self.distance == "fractional":
            delta = normalize_rows(query)[:, None, :] - normalize_rows(centers)[None, :, :]
            projected = np.einsum("nkd,de->nke", delta, self._eigvecs)
            weights = (self._eigvals / self._mean_eigval) ** (-self.spectral_alpha)
            return np.einsum("nkd,d,nkd->nk", projected, weights, projected)

        # centered_angular
        weights = (self._eigvals / self._mean_eigval) ** (-self.spectral_alpha / 2.0)
        q = normalize_rows((normalize_rows(query) - self._global_center) @ self._eigvecs * weights)
        c = normalize_rows((normalize_rows(centers) - self._global_center) @ self._eigvecs * weights)
        return 1.0 - q @ c.T

    def describe(self) -> dict:
        out = {"distance": self.distance}
        if self.distance == "mahalanobis":
            out["covariance"] = self.covariance
            if self.covariance == "rda":
                out["rda_class_weight"] = self.rda_class_weight
                out["rda_logdet_weight"] = self.rda_logdet_weight
        if self.distance in ("fractional", "centered_angular"):
            out["spectral_alpha"] = self.spectral_alpha
        return out


def build_scorer(distance: str, eval_cfg: dict, seed: int = 42) -> Scorer:
    """yaml eval 섹션에서 Scorer 생성."""
    maha = eval_cfg.get("mahalanobis", {}) or {}
    rda = maha.get("rda", {}) or {}
    spectral = eval_cfg.get("spectral", {}) or {}
    return Scorer(
        distance=distance,
        covariance=maha.get("covariance", "ledoit_wolf"),
        rda_class_weight=float(rda.get("class_weight", 0.75)),
        rda_logdet_weight=float(rda.get("logdet_weight", 1.0)),
        spectral_alpha=float(spectral.get("alpha", 0.5)),
        seed=seed,
    )


def build_transform(eval_cfg: dict, seed: int = 42) -> EmbeddingTransform:
    """yaml eval.transform 섹션에서 EmbeddingTransform 생성."""
    cfg = eval_cfg.get("transform", {}) or {}
    return EmbeddingTransform(
        mode=cfg.get("mode", "none"),
        pca_dim=int(cfg.get("pca_dim", 64)),
        pca_balanced_fit=bool(cfg.get("pca_balanced_fit", False)),
        seed=seed,
    )
