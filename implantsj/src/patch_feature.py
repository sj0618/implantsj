"""VLM feature 를 ViT 의 패치 토큰으로 주입하는 실험용 모듈.

기존 `pipelines/vlm/train_vlm_fusion_5feature.py` 는 late fusion 이다.
이미지 로짓과 feature 로짓을 각각 낸 뒤 더한다. 그래서 feature 가 이미지의
"어디"와 관련되는지 transformer 가 볼 기회가 없다.

여기서는 feature 를 **패치 토큰으로 붙인다**. VLM 이 뽑은 feature 하나가
토큰 하나가 되어 이미지 패치들과 같은 시퀀스에 들어가고, 첫 블록부터
self-attention 으로 서로를 본다.

    [CLS] [img patch 1..196] [feat 1..21]
      |         |                 |
      +---------+-----------------+   <- 모든 블록에서 상호 attention

입력 feature 는 aihub_all_13feature_21reason_vlm.jsonl 형식을 그대로 쓴다.
각 레코드의 `feature_vector` 는 21개 feature x 3개 성분(value,
assessable_mask, confidence) = 63 차원이고 전부 [0,1] 이다. 21개 feature
각각이 토큰 하나가 되며, 3차원 성분이 그 토큰의 입력이 된다.

feature 가 없는 이미지가 섞여 있어도 동작한다(학습 데이터 전체에 VLM 을
돌리지 못한 경우). 그런 샘플은 학습되는 null 토큰을 대신 받고, 커버리지가
얼마인지 학습 시작 전에 출력한다.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

# =====================================================================
# 1. feature 파일 로딩
# =====================================================================

@dataclass
class FeatureTable:
    """VLM feature 모음. 이미지 경로로 조회한다."""
    names: List[str]              # feature 그룹 이름 21개
    components: List[str]         # 성분 이름 3개 (value/assessable_mask/confidence)
    lookup: Dict[str, np.ndarray] # 조회키 -> (F, C) float32

    @property
    def num_features(self) -> int:
        return len(self.names)

    @property
    def dims_per_feature(self) -> int:
        return len(self.components)


def _lookup_keys(rec: dict) -> List[str]:
    """한 레코드가 매칭될 수 있는 키들. 경로 표기가 달라도 찾히게 한다."""
    keys: List[str] = []
    for field in ("path", "image_path", "roi_path", "file", "filename", "id", "image_id"):
        raw = rec.get(field)
        if raw is None:
            continue
        s = str(raw)
        p = Path(s)
        keys.extend([s, p.name, p.stem])
    return keys


def load_feature_table(path: str | Path) -> FeatureTable:
    """jsonl 을 읽어 FeatureTable 을 만든다.

    `feature_vector` 와 `feature_vector_names` 를 그대로 쓴다. 이름은
    `<그룹>.<성분>` 형식이라 그룹별로 묶어 (F, C) 행렬로 만든다.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"feature file not found: {path}")

    names: Optional[List[str]] = None
    comps: Optional[List[str]] = None
    lookup: Dict[str, np.ndarray] = {}
    n_rows = 0
    n_skipped = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            vec = rec.get("feature_vector")
            vnames = rec.get("feature_vector_names")
            if vec is None or vnames is None:
                n_skipped += 1
                continue
            if len(vec) != len(vnames):
                raise ValueError(
                    f"{path}:{line_no}: feature_vector({len(vec)}) 와 "
                    f"feature_vector_names({len(vnames)}) 길이가 다르다."
                )

            # <그룹>.<성분> 을 그룹 순서 유지하며 묶는다.
            grouped: "OrderedDict[str, OrderedDict[str, float]]" = OrderedDict()
            for name, value in zip(vnames, vec):
                base, _, comp = str(name).rpartition(".")
                if not base:      # 성분 구분이 없으면 단일 성분으로 취급
                    base, comp = str(name), "value"
                grouped.setdefault(base, OrderedDict())[comp] = float(value)

            row_names = list(grouped.keys())
            row_comps = list(next(iter(grouped.values())).keys())
            if names is None:
                names, comps = row_names, row_comps
            elif row_names != names or row_comps != comps:
                raise ValueError(
                    f"{path}:{line_no}: feature 구성이 첫 레코드와 다르다. "
                    f"모든 레코드의 feature 이름과 순서가 같아야 한다."
                )

            mat = np.zeros((len(names), len(comps)), dtype=np.float32)
            for i, base in enumerate(names):
                for j, comp in enumerate(comps):
                    mat[i, j] = grouped[base].get(comp, 0.0)

            for key in _lookup_keys(rec):
                lookup[key] = mat
            n_rows += 1

    if names is None or comps is None:
        raise ValueError(f"{path}: 사용할 수 있는 feature_vector 레코드가 없다.")

    print(f"[features] {path.name}: {n_rows} records, "
          f"{len(names)} features x {len(comps)} components"
          + (f", skipped {n_skipped}" if n_skipped else ""))
    return FeatureTable(names=names, components=comps, lookup=lookup)


def find_feature(table: FeatureTable, image_path: str) -> Optional[np.ndarray]:
    """이미지 경로로 feature 를 찾는다. 전체경로 -> 파일명 -> 확장자 뺀 이름 순."""
    p = Path(str(image_path))
    for key in (str(image_path), p.name, p.stem):
        hit = table.lookup.get(key)
        if hit is not None:
            return hit
    return None


# =====================================================================
# 2. Dataset
# =====================================================================

# feature 를 어떻게 넣을지. 통제 실험을 위해 대조군을 함께 둔다.
FEATURE_MODES = (
    "patch",    # 정상: feature 를 패치 토큰으로 주입
    "none",     # 대조군: 이미지만 (feature 토큰 자체를 안 붙임)
    "zero",     # 대조군: 토큰은 붙이되 입력을 0 으로 (늘어난 파라미터 효과만 측정)
    "shuffle",  # 대조군: feature 를 다른 이미지 것과 섞음 (진짜 쓰는지 검증)
)


class PatchFeatureDataset(Dataset):
    """이미지 + VLM feature 를 함께 내주는 데이터셋.

    반환: (image, label, feature(F,C), has_feature(bool), path)
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        split: Optional[str],
        class_to_idx: Dict[str, int],
        table: Optional[FeatureTable],
        transform=None,
        feature_mode: str = "patch",
        shuffle_seed: int = 42,
    ):
        from src.data import load_manifest  # 기존 manifest 로더 재사용

        if feature_mode not in FEATURE_MODES:
            raise ValueError(f"feature_mode must be one of {FEATURE_MODES}, got {feature_mode!r}")

        self.df = load_manifest(manifest_csv, split=split)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.feature_mode = feature_mode
        self.table = table

        if table is None:
            self.num_features, self.dims_per_feature = 0, 0
        else:
            self.num_features, self.dims_per_feature = table.num_features, table.dims_per_feature

        # 각 행의 feature 를 미리 찾아둔다(매 __getitem__ 마다 찾지 않도록).
        self._feats: List[Optional[np.ndarray]] = []
        if table is not None and feature_mode != "none":
            for path in self.df["path"].astype(str).tolist():
                self._feats.append(find_feature(table, path))
        else:
            self._feats = [None] * len(self.df)

        # shuffle 대조군: feature 가 있는 것끼리 자리를 섞는다(시드 고정).
        if feature_mode == "shuffle":
            have = [i for i, f in enumerate(self._feats) if f is not None]
            if have:
                rng = np.random.default_rng(shuffle_seed)
                perm = rng.permutation(len(have))
                donors = [self._feats[have[j]] for j in perm]
                for slot, donor in zip(have, donors):
                    self._feats[slot] = donor

        self.n_with = sum(1 for f in self._feats if f is not None)

    def coverage(self) -> Tuple[int, int]:
        """(feature 가 있는 샘플 수, 전체 샘플 수)"""
        return self.n_with, len(self.df)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = str(row["path"])
        y = self.class_to_idx[str(row["label"])]

        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        F, C = self.num_features, self.dims_per_feature
        feat = self._feats[idx]
        if feat is None or self.feature_mode in ("none", "zero"):
            # 'zero' 는 토큰은 붙이되 입력을 0 으로 둔다 -> has_feature=True
            # 'none' 은 토큰을 안 붙이므로 값이 쓰이지 않는다.
            has = self.feature_mode == "zero"
            feat_t = torch.zeros(max(F, 0), max(C, 0), dtype=torch.float32)
        else:
            has = True
            feat_t = torch.from_numpy(np.ascontiguousarray(feat))

        return img, y, feat_t, torch.tensor(has), path


# =====================================================================
# 3. 모델
# =====================================================================

class FeatureTokenizer(nn.Module):
    """(B, F, C) feature 를 (B, F, D) 패치 토큰으로 바꾼다.

    feature 마다 성질이 다르므로 공유 Linear 하나가 아니라 feature 별로
    독립적인 projection 을 준다(weight shape (F, C, D)). 여기에 "내가 몇 번
    feature 인지" 알려주는 학습 가능한 type embedding 을 더한다. 이미지
    패치가 위치 임베딩으로 자리를 구분하는 것과 같은 역할이다.
    """

    def __init__(self, num_features: int, dims_per_feature: int, embed_dim: int):
        super().__init__()
        self.num_features = num_features
        self.dims_per_feature = dims_per_feature
        self.weight = nn.Parameter(torch.empty(num_features, dims_per_feature, embed_dim))
        self.bias = nn.Parameter(torch.zeros(num_features, embed_dim))
        self.type_embed = nn.Parameter(torch.zeros(num_features, embed_dim))
        # VLM feature 가 없는 이미지가 대신 받는 토큰.
        self.null_token = nn.Parameter(torch.zeros(num_features, embed_dim))
        nn.init.trunc_normal_(self.weight, std=0.02)
        nn.init.trunc_normal_(self.type_embed, std=0.02)
        nn.init.trunc_normal_(self.null_token, std=0.02)

    def forward(self, feat: torch.Tensor, has_feat: torch.Tensor) -> torch.Tensor:
        # feat (B,F,C) x weight (F,C,D) -> (B,F,D)
        tok = torch.einsum("bfc,fcd->bfd", feat, self.weight) + self.bias
        tok = tok + self.type_embed
        null = self.null_token.unsqueeze(0).expand_as(tok)
        return torch.where(has_feat.view(-1, 1, 1).bool(), tok, null)


class ViTWithFeaturePatches(nn.Module):
    """timm ViT 의 토큰 시퀀스 뒤에 feature 토큰을 이어붙인 모델.

    timm 의 forward_features 를 그대로 쓰지 않고 같은 순서를 따라가되
    blocks 직전에 feature 토큰을 concat 한다:

        patch_embed -> _pos_embed(cls+pos) -> patch_drop
          -> [feature 토큰 concat]  -> norm_pre -> blocks -> norm -> head

    patch_drop 뒤에 붙이는 이유는, patch_drop 이 켜져 있을 때 feature 토큰이
    무작위로 버려지지 않게 하기 위해서다.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        num_features: int,
        dims_per_feature: int,
        pretrained: bool = True,
        feature_mode: str = "patch",
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        import timm

        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        # 패치 토큰을 이어붙이는 구조라 ViT 계열에서만 성립한다.
        missing = [a for a in ("patch_embed", "_pos_embed", "blocks", "norm", "forward_head")
                   if not hasattr(self.vit, a)]
        if missing:
            raise ValueError(
                f"'{model_name}' 은 패치 토큰 주입을 지원하지 않는다 (없는 속성: {missing}). "
                f"vit_* 계열 모델을 쓰거나, feature 를 쓰지 않는 일반 학습은 "
                f"pipelines/train/train_supervised.py 를 쓴다."
            )

        self.feature_mode = feature_mode
        self.num_features = num_features
        self.embed_dim = self.vit.embed_dim
        self.tokenizer = (
            None if feature_mode == "none"
            else FeatureTokenizer(num_features, dims_per_feature, self.embed_dim)
        )

    def forward(self, img: torch.Tensor, feat: torch.Tensor, has_feat: torch.Tensor) -> torch.Tensor:
        v = self.vit
        x = v.patch_embed(img)
        x = v._pos_embed(x)      # [CLS] + pos embed
        x = v.patch_drop(x)

        if self.tokenizer is not None:
            x = torch.cat([x, self.tokenizer(feat, has_feat)], dim=1)

        x = v.norm_pre(x)
        x = v.blocks(x)
        x = v.norm(x)
        return v.forward_head(x)


# =====================================================================
# 4. 저장 안전장치
# =====================================================================

def assert_finite_state_dict(model: nn.Module, context: str) -> None:
    """NaN/Inf 가 섞인 checkpoint 를 저장하지 않도록 막는다.

    학습이 발산하면 지표는 그럴듯한데 가중치는 망가진 checkpoint 가
    best 로 저장될 수 있다. 그런 파일을 나중에 불러 평가하면 원인을 찾기
    어려우므로 저장 직전에 거른다.
    """
    bad = [k for k, v in model.state_dict().items()
           if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all()]
    if bad:
        raise RuntimeError(
            f"{context}: {len(bad)} model tensors are NaN/Inf (e.g. {bad[:3]}). "
            f"Refusing to save a corrupt checkpoint. Lower --lr or raise --batch-size and re-run."
        )
