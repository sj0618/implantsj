"""metric 모델: 백본 + projection 임베딩, loss 3종, 체크포인트.

- MetricModel : timm 백본 + MLP projection → L2 정규화 임베딩 (implantsj src/models.py)
- 백본 레지스트리: Stage 1에서 만든 백본 5종(vit/band-vit/dinov2/band-dinov2/cnn)
  체크포인트를 경로로 끌어와 초기화. 체크포인트가 없으면 timm pretrained로 시작.
- SupConLoss / BatchHardTripletLoss (implantsj src/metric_learning.py)
- ArcFaceHead (implantsj src/arcface.py — margin softmax)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .core import log


# =====================================================================
# 백본 레지스트리 (base.yaml의 backbones 섹션과 대응)
# =====================================================================

def resolve_backbone(name: str, registry: Dict[str, dict]) -> dict:
    """backbone 이름 → {timm, checkpoint, image_size} 해석."""
    if name not in registry:
        raise KeyError(
            f"backbone {name!r}이 레지스트리에 없습니다. "
            f"base.yaml backbones에 정의된 이름: {sorted(registry)}"
        )
    entry = dict(registry[name])
    entry.setdefault("checkpoint", None)
    entry.setdefault("image_size", 224)
    if "timm" not in entry:
        raise ValueError(f"backbone {name!r} 항목에 timm 모델명이 없습니다")
    return entry


class MetricModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        embedding_dim: int = 128,
        pretrained: bool = True,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        image_size: int | None = None,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        import timm

        kwargs = dict(pretrained=pretrained, num_classes=0,
                      drop_rate=dropout, drop_path_rate=drop_path_rate)
        # Stage 1 백본과 같은 해상도로 만들어야 pos_embed shape이 맞는다
        # (dinov2는 기본 518이라 224 체크포인트를 그냥 로드하면 깨진다).
        try:
            self.backbone = timm.create_model(model_name, img_size=image_size, **kwargs) \
                if image_size else timm.create_model(model_name, **kwargs)
        except TypeError:
            self.backbone = timm.create_model(model_name, **kwargs)
        if grad_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)   # 큰 모델 메모리 절약
        in_dim = getattr(self.backbone, "num_features", None)
        if in_dim is None:
            raise RuntimeError(f"Could not infer num_features for {model_name}")
        self.projector = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, embedding_dim),
        )

    def forward_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        z = self.projector(self.forward_features(x))
        return nn.functional.normalize(z, dim=1)


def build_metric_model(
    backbone_name: str,
    registry: Dict[str, dict],
    embedding_dim: int,
    dropout: float = 0.0,
    drop_path_rate: float = 0.0,
    grad_checkpointing: bool = False,
) -> Tuple[MetricModel, int]:
    """레지스트리 기반 MetricModel 생성 + Stage 1 백본 체크포인트 로드."""
    entry = resolve_backbone(backbone_name, registry)
    checkpoint = entry["checkpoint"]
    image_size = int(entry["image_size"])
    model = MetricModel(
        entry["timm"],
        embedding_dim=embedding_dim,
        pretrained=checkpoint is None,   # 체크포인트가 있으면 timm pretrained 생략
        dropout=dropout,
        drop_path_rate=drop_path_rate,
        image_size=image_size,
        grad_checkpointing=grad_checkpointing,
    )
    if checkpoint:
        loaded, skipped = load_matching_weights(model.backbone, checkpoint)
        log(f"backbone {backbone_name}: {checkpoint}에서 {loaded}개 텐서 로드 (skip {skipped})")
    else:
        log(f"backbone {backbone_name}: 체크포인트 미지정 → timm pretrained 사용")
    return model, image_size


# =====================================================================
# 체크포인트 (implantsj src/models.py)
# =====================================================================

def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def load_matching_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    strict: bool = False,
    min_coverage: float = 0.5,
) -> Tuple[int, int]:
    """shape이 맞는 텐서만 로드. metric 체크포인트의 backbone.* prefix도 처리."""
    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    state = unwrap_state_dict(ckpt)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint does not contain a state dict: {checkpoint_path}")
    model_state = model.state_dict()
    filtered = {}
    skipped = 0
    for k, v in state.items():
        kk = k
        for prefix in ("module.", "model."):
            if kk.startswith(prefix):
                kk = kk[len(prefix):]
        if kk in model_state and tuple(model_state[kk].shape) == tuple(v.shape):
            filtered[kk] = v
            continue
        if kk.startswith("backbone."):
            alt = kk[len("backbone."):]
            if alt in model_state and tuple(model_state[alt].shape) == tuple(v.shape):
                filtered[alt] = v
                continue
        skipped += 1
    loaded_numel = sum(int(model_state[key].numel()) for key in filtered)
    total_numel = sum(int(value.numel()) for value in model_state.values())
    coverage = loaded_numel / max(1, total_numel)
    if coverage < min_coverage:
        raise RuntimeError(
            f"Checkpoint coverage is only {coverage:.1%} ({len(filtered)}/{len(model_state)} tensors) "
            f"for {checkpoint_path}; required minimum is {min_coverage:.1%}"
        )
    model.load_state_dict(filtered, strict=strict)
    return len(filtered), skipped


def save_checkpoint(path: str | Path, model, optimizer, epoch: int, metrics: dict, args: dict, class_to_idx: Dict[str, int]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "metrics": metrics,
            "args": dict(args),
            "class_to_idx": class_to_idx,
        },
        path,
    )


# =====================================================================
# losses (implantsj src/metric_learning.py)
# =====================================================================

class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # features: [2B, D], labels: [2B]
        features = nn.functional.normalize(features, dim=1)
        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        mask = torch.eq(labels[:, None], labels[None, :]).float().to(features.device)
        self_mask = torch.eye(mask.size(0), device=features.device)
        mask = mask * (1.0 - self_mask)
        exp_logits = torch.exp(logits) * (1.0 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return -mean_log_prob_pos.mean()


def covariance_scores(
    support: torch.Tensor,
    support_y: torch.Tensor,
    query: torch.Tensor,
    classes: int,
    covariance_weight: float,
    beta: float,
    logdet_weight: float,
    covariance_mode: str = "legacy",
    covariance_shrinkage: float = 0.1,
) -> torch.Tensor:
    """episode 안에서 support로 공분산을 추정해 query를 Mahalanobis로 채점.

    covariance_mode:
      legacy          pooled ↔ 클래스별 '대각' 분산 블렌딩 (기본)
      pooled_diagonal pooled 대각 분산만 (logdet 항 무시)
      pooled_full     pooled 전체 공분산 + shrinkage
    """
    centers = torch.stack([support[support_y == index].mean(0) for index in range(classes)])
    residuals = support - centers[support_y]
    pooled_var = residuals.pow(2).mean(0)

    if covariance_mode == "pooled_full":
        if not 0.0 <= covariance_shrinkage < 1.0:
            raise ValueError("covariance_shrinkage must be in [0, 1)")
        covariance = residuals.T @ residuals / max(1, residuals.shape[0])
        mean_variance = covariance.diag().mean().clamp_min(beta)
        eye = torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)
        covariance = (1.0 - covariance_shrinkage) * covariance
        covariance = covariance + covariance_shrinkage * mean_variance * eye + beta * eye
        precision = torch.linalg.inv(covariance)
        delta = query[:, None, :] - centers[None, :, :]
        return torch.einsum("nkd,de,nke->nk", delta, precision, delta)

    class_var = torch.stack(
        [(support[support_y == index] - centers[index]).pow(2).mean(0) for index in range(classes)]
    )
    if covariance_mode == "pooled_diagonal":
        variance = pooled_var[None, :].expand(classes, -1)
        logdet_weight = 0.0
    elif covariance_mode == "legacy":
        variance = (1.0 - covariance_weight) * pooled_var[None, :] + covariance_weight * class_var
    else:
        raise ValueError(f"unknown covariance_mode: {covariance_mode}")
    variance = variance.clamp_min(beta)
    delta = query[:, None, :] - centers[None, :, :]
    scores = (delta.pow(2) / variance[None, :, :]).sum(-1)
    if logdet_weight:
        scores = scores + logdet_weight * variance.log().sum(1)[None, :]
    return scores


def covariance_loss(
    support, support_y, query, query_y, classes: int,
    covariance_weight: float, beta: float, logdet_weight: float,
    covariance_mode: str, covariance_shrinkage: float,
) -> torch.Tensor:
    """-score를 logit으로 쓰는 CE — Mahalanobis 거리에서 잘 분리되도록 직접 학습."""
    return nn.functional.cross_entropy(
        -covariance_scores(
            support, support_y, query, classes,
            covariance_weight, beta, logdet_weight,
            covariance_mode, covariance_shrinkage,
        ),
        query_y,
    )


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # torch.cdist의 역전파는 MPS(Apple Silicon)에 구현돼 있지 않아 학습이 죽는다.
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b 로 직접 계산하면 어느 장치에서도 돈다.
        square = embeddings.pow(2).sum(dim=1)
        dist = (square[:, None] + square[None, :]
                - 2.0 * embeddings @ embeddings.t()).clamp_min(0)
        # sqrt(0)의 기울기가 무한대라 eps를 더한 뒤 대각선을 0으로 되돌린다
        dist = (dist + 1e-12).sqrt()
        dist = dist - torch.diag_embed(torch.diagonal(dist))
        labels_equal = labels[:, None] == labels[None, :]
        eye = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
        pos_mask = labels_equal & ~eye
        neg_mask = ~labels_equal
        hardest_pos = torch.where(pos_mask, dist, torch.zeros_like(dist)).max(dim=1).values
        hardest_neg = torch.where(neg_mask, dist, torch.full_like(dist, 1e6)).min(dim=1).values
        return nn.functional.relu(hardest_pos - hardest_neg + self.margin).mean()


# =====================================================================
# ArcFace head (implantsj src/arcface.py)
# =====================================================================

class ArcFaceHead(nn.Module):
    """margin softmax: cos(θ+m)을 target 클래스에만 적용하고 s배 스케일."""

    def __init__(self, embedding_dim: int, num_classes: int, s: float = 30.0, m: float = 0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        # 수치 안정용 상수 (margin 경계 밖에서 단조성 보장)
        self._cos_m = math.cos(m)
        self._sin_m = math.sin(m)
        self._th = math.cos(math.pi - m)
        self._mm = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        cos = nn.functional.linear(
            nn.functional.normalize(embeddings, dim=1),
            nn.functional.normalize(self.weight, dim=1),
        )
        if labels is None:
            return cos * self.s
        cos = cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sin = torch.sqrt(1.0 - cos * cos)
        phi = cos * self._cos_m - sin * self._sin_m            # cos(θ+m)
        phi = torch.where(cos > self._th, phi, cos - self._mm)  # 경계 밖은 단조 대체
        onehot = torch.zeros_like(cos)
        onehot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = onehot * phi + (1.0 - onehot) * cos            # target만 margin 적용
        return logits * self.s
