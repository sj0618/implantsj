#!/usr/bin/env python3
"""
DINOv2 backbone + projection head + 분류기. extract.py / train.py 가 공유한다.

확정 사항
  backbone : DINOv2 ViT-B/14 (timm `vit_base_patch14_dinov2.lvd142m`)
  특징     : CLS 토큰 768d  (timm num_classes=0 + global_pool='token')
  입력     : 224x224  (14 x 16 = 224 이므로 패치 격자가 정확히 떨어진다)

**임베딩의 정의 (2026-07-30 변경)**

  M1 Base       : backbone CLS 토큰 그대로 (768d)
  M2~M7 (학습)  : backbone → **projection head** → 768d

왜 projection head 가 필요한가 —
처음에는 head 를 분류기(768→C)로만 두고 임베딩은 backbone 출력을 썼다. 그런데
frozen 조건(M2·M3)은 backbone 이 안 변하므로 임베딩이 M1 과 완전히 동일해졌고,
프로토타입 분류 결과가 소수점 4자리까지 M1 과 같아졌다. 6셀이 복사본이 되고
"head 학습이 도움되나"(Q1)를 물을 수 없었다. 학습이 임베딩 공간을 실제로
바꾸려면 임베딩 자체가 학습되는 층을 통과해야 한다.

왜 projection 이 선형이 아닌가 —
proj 가 Linear 이고 분류기도 Linear 이면 logits = W2(W1x+b1)+b2 가 되어
하나의 선형층과 등가다. CE 손실은 합성 W2W1 만 결정하고 W1 자체는
결정하지 않는다(underdetermined) → 임베딩 공간이 임의로 정해진다.
사이에 GELU 를 넣어 proj 출력이 손실에 실제로 구속되게 한다.

출력 차원은 768 로 유지한다. M1(768d)과 차원이 같아야 "학습 때문에 좋아진 것"과
"차원이 달라서 생긴 차이"가 섞이지 않는다.
"""

import torch
import torch.nn as nn

MODEL_NAME = "vit_base_patch14_dinov2.lvd142m"
EMBED_DIM = 768
IMG_SIZE = 224


def build_backbone(pretrained=True, model_name=MODEL_NAME, img_size=IMG_SIZE):
    """CLS 토큰 768d 를 뱉는 backbone. num_classes=0 이라 분류기가 없다."""
    import timm

    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,          # 분류기 제거 → forward()가 pooled 특징 반환
        global_pool="token",    # 패치 평균이 아니라 CLS 토큰
        img_size=img_size,      # 사전학습 518 → 224로 위치 임베딩 보간
    )


class ProjectionHead(nn.Module):
    """768 → GELU → 768. 이 출력이 프로토타입 분류에 쓰이는 임베딩이다."""

    def __init__(self, dim=EMBED_DIM, hidden=EMBED_DIM, out=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out),
        )

    def forward(self, x):
        return self.net(x)


class Embedder(nn.Module):
    """이미지 → 임베딩. extract.py 가 쓰는 추론 전용 래퍼.

    proj 가 None 이면 backbone 특징을 그대로 임베딩으로 쓴다 (M1 Base).
    """

    def __init__(self, backbone, proj=None):
        super().__init__()
        self.backbone = backbone
        self.proj = proj

    def forward(self, x):
        f = self.backbone(x)
        return f if self.proj is None else self.proj(f)


class ImplantModel(nn.Module):
    """backbone → proj → 분류기.

    forward 는 (emb, logits) 를 돌려준다.
      emb    : 프로토타입 분류와 SupCon 에 쓰인다 (proj 출력)
      logits : CE 손실과 softmax 참조값에 쓰인다
    """

    def __init__(self, num_classes, pretrained=True, freeze_backbone=False):
        super().__init__()
        self.backbone = build_backbone(pretrained=pretrained)
        self.proj = ProjectionHead()
        self.head = nn.Linear(EMBED_DIM, num_classes)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()  # frozen 조건에서 dropout/droppath 비활성
        return self

    def forward(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                f = self.backbone(x)
        else:
            f = self.backbone(x)
        emb = self.proj(f)
        return emb, self.head(emb)


def pick_device(prefer=None):
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
