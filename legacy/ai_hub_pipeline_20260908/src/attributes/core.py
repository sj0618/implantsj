"""attributes 패키지 공용 — 데이터/모델/유틸. 다른 패키지를 import하지 않는다.

모델 구성 2가지 (yaml의 train.mode):
  shared    백본 1개 + feature마다 head  → 13 feature를 동시에 학습 (빠름, 표현 공유)
  separate  feature마다 백본+head를 통째로 하나씩 → 따로 학습 (간섭 없음, 13배 비쌈)
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image


# =====================================================================
# utils
# =====================================================================

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def seed_everything(seed: int = 42) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =====================================================================
# manifest / dataset
# =====================================================================

def read_manifest(path: str | Path, split: Optional[str] = None) -> pd.DataFrame:
    manifest_path = Path(path).expanduser().resolve()
    frame = pd.read_csv(manifest_path)
    if not {"path", "label"}.issubset(frame.columns):
        raise ValueError(f"manifest {manifest_path} must contain path,label")
    frame = frame.copy()
    frame["path"] = frame["path"].map(
        lambda v: str((Path(str(v)).expanduser() if Path(str(v)).is_absolute()
                       else manifest_path.parent / str(v)).resolve())
    )
    frame["label"] = frame["label"].astype(str).str.strip()
    if split is not None:
        if "split" not in frame.columns:
            raise ValueError(f"manifest {manifest_path} has no split column")
        frame = frame[frame["split"].astype(str) == split]
    return frame.reset_index(drop=True)


def build_transforms(image_size: int, train: bool):
    from torchvision import transforms

    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(), normalize,
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), normalize,
    ])


class AttributeDataset:
    """이미지 → (tensor, class_index, path). feature 정답은 class_index로 조회한다."""

    def __init__(self, frame: pd.DataFrame, class_to_idx: Dict[str, int],
                 image_size: int, *, train: bool):
        # spec table에 못 붙은 클래스는 애초에 제외 (정답이 없으므로)
        self.frame = frame[frame["label"].isin(class_to_idx)].reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = build_transforms(image_size, train)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row["path"]) as source:
            image = self.transform(source.convert("RGB"))
        return image, self.class_to_idx[str(row["label"])], str(row["path"])


def make_loader(dataset, batch_size: int, num_workers: int, *, shuffle: bool):
    import torch

    kwargs = {"batch_size": batch_size, "shuffle": shuffle, "num_workers": num_workers,
              "pin_memory": torch.cuda.is_available()}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return torch.utils.data.DataLoader(dataset, **kwargs)


# =====================================================================
# 모델
# =====================================================================

def _create_backbone(timm_model: str, image_size: int, *, pretrained: bool,
                     dropout: float, drop_path_rate: float, checkpoint: str | None):
    """timm 백본(num_classes=0) 생성 + 선택적 체크포인트 로드."""
    import timm
    import torch

    kwargs = dict(pretrained=pretrained and not checkpoint, num_classes=0,
                  drop_rate=dropout, drop_path_rate=drop_path_rate)
    try:
        backbone = timm.create_model(timm_model, img_size=image_size, **kwargs)
    except TypeError:
        backbone = timm.create_model(timm_model, **kwargs)   # CNN 등은 img_size 미지원

    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        for key in ("model_state_dict", "state_dict", "model"):
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                state = state[key]
                break
        target = backbone.state_dict()
        filtered = {}
        for k, v in state.items():
            kk = k
            for prefix in ("module.", "model.", "backbone."):
                if kk.startswith(prefix):
                    kk = kk[len(prefix):]
            if kk in target and tuple(target[kk].shape) == tuple(v.shape):
                filtered[kk] = v
        backbone.load_state_dict(filtered, strict=False)
        log(f"  backbone 체크포인트에서 {len(filtered)}/{len(target)} 텐서 로드: {checkpoint}")
    return backbone


def build_feature_model(spec_entry: dict, attribute: str, num_values: int, *,
                        dropout: float = 0.1, drop_path_rate: float = 0.1,
                        pretrained: bool = True):
    """feature 하나를 담당하는 **독립 모델** (백본 + head) — train.mode: separate.

    feature마다 백본을 통째로 하나씩 갖는다. 공유 백본 방식은 build_shared_model.
    forward_features / forward_heads 는 두 모델 구성이 같은 인터페이스를 갖게 한다
    (학습 루프·임베딩 저장이 mode 를 몰라도 되게).
    """
    import torch.nn as nn

    class FeatureModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attribute = attribute
            self.attributes = [attribute]
            self.backbone = _create_backbone(
                spec_entry["timm"], int(spec_entry.get("image_size", 224)),
                pretrained=pretrained, dropout=dropout, drop_path_rate=drop_path_rate,
                checkpoint=spec_entry.get("checkpoint"),
            )
            in_dim = getattr(self.backbone, "num_features", None)
            if in_dim is None:
                raise RuntimeError(f"num_features를 알 수 없습니다: {spec_entry['timm']}")
            self.head = nn.Linear(in_dim, num_values)

        def forward_features(self, x):
            return self.backbone(x)

        def forward_heads(self, features):
            return {self.attribute: self.head(features)}

        def forward(self, x):
            return self.forward_heads(self.forward_features(x))

    return FeatureModel()


def build_shared_model(spec_entry: dict, head_sizes: Dict[str, int], *,
                       dropout: float = 0.1, drop_path_rate: float = 0.1,
                       pretrained: bool = True):
    """백본 1개 + feature마다 head (multi-head) — train.mode: shared.

    forward(x) 는 {feature: logits} 를 한 번에 돌려주므로 masked_cross_entropy 가
    모든 feature 의 loss 를 합쳐 한 번의 backward 로 학습한다.
    """
    import torch.nn as nn

    if not head_sizes:
        raise ValueError("shared 모델은 head 가 하나 이상 필요합니다")

    class SharedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attributes = list(head_sizes)
            self.backbone = _create_backbone(
                spec_entry["timm"], int(spec_entry.get("image_size", 224)),
                pretrained=pretrained, dropout=dropout, drop_path_rate=drop_path_rate,
                checkpoint=spec_entry.get("checkpoint"),
            )
            in_dim = getattr(self.backbone, "num_features", None)
            if in_dim is None:
                raise RuntimeError(f"num_features를 알 수 없습니다: {spec_entry['timm']}")
            self.heads = nn.ModuleDict({a: nn.Linear(in_dim, int(n)) for a, n in head_sizes.items()})

        def forward_features(self, x):
            return self.backbone(x)

        def forward_heads(self, features):
            return {a: head(features) for a, head in self.heads.items()}

        def forward(self, x):
            return self.forward_heads(self.forward_features(x))

    return SharedModel()


def attribute_class_weights(target, attributes: Sequence[str], vocab: Dict[str, list],
                            class_index, device):
    """feature마다 값 빈도의 역수를 가중치로. 드문 값이 loss에서 밀리지 않게 한다."""
    import torch

    weights = {}
    for ai, a in enumerate(attributes):
        values = target[class_index, ai]
        counts = np.bincount(values[values >= 0], minlength=len(vocab[a])).astype(np.float32)
        w = 1.0 / np.maximum(counts, 1.0)
        weights[a] = torch.tensor(w / w.sum() * len(w), dtype=torch.float32, device=device)
    return weights


def masked_cross_entropy(logits: Dict, target, attributes: Sequence[str], smoothing: float,
                         weights: Dict | None = None):
    """unknown(-1)은 loss에서 제외 — 정답을 모르는 칸을 틀렸다고 셀 수 없다."""
    import torch.nn.functional as functional

    total = None
    heads = sum(1 for a in attributes if a in logits)   # 분모는 head 수로 고정 — 배치마다 unknown 이
    for ai, a in enumerate(attributes):                  # 달라도 loss 스케일(실효 lr)이 흔들리지 않게
        if a not in logits:
            continue
        t = target[:, ai]
        keep = t >= 0
        if not bool(keep.any()):
            continue
        loss = functional.cross_entropy(logits[a][keep], t[keep], label_smoothing=smoothing,
                                        weight=(weights or {}).get(a))
        total = loss if total is None else total + loss
    if total is None:
        raise RuntimeError("배치 전체가 unknown이라 loss를 계산할 수 없습니다")
    return total / max(heads, 1)
