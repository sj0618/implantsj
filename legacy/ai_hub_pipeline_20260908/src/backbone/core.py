"""backbone 패키지 공용 — 데이터/모델/attention/저장. 다른 패키지를 import하지 않는다.

band-* 백본의 핵심: 이미지와 band mask를 같은 기하로 augment한 뒤,
CLS→patch attention을 band 쪽으로 유도하는 KL prior loss를 CE에 더한다.
이 로직은 ViT 구조(patch_embed.grid_size, blocks[].attn.attn_drop, num_prefix_tokens)에만
의존하므로 ViT와 DINOv2에 동일하게 적용된다.

manifest 계약: CSV에 path,label,split (+ band-* 백본은 mask_path).
                mask_path는 Stage 0의 make_band_masks.py 산출물.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


# =====================================================================
# utils
# =====================================================================

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(path: str | Path, obj: dict) -> None:
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

def pad_to_square(image: Image.Image, fill=0) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    h, v = side - width, side - height
    return ImageOps.expand(image, border=(h // 2, v // 2, h - h // 2, v - v // 2), fill=fill)


def read_manifest(path: str | Path, *, require_mask: bool = False,
                  mask_column: str = "mask_path") -> pd.DataFrame:
    manifest_path = Path(path).expanduser().resolve()
    frame = pd.read_csv(manifest_path)
    if not {"path", "label", "split"}.issubset(frame.columns):
        raise ValueError(f"manifest {manifest_path} must contain path,label,split")

    def resolve(value, optional=False):
        if pd.isna(value) or not str(value).strip():
            if optional:
                return ""
            raise ValueError(f"empty path in manifest {manifest_path}")
        p = Path(str(value)).expanduser()
        return str((p if p.is_absolute() else manifest_path.parent / p).resolve())

    frame = frame.copy()
    frame["path"] = frame["path"].map(resolve)
    frame["label"] = frame["label"].astype(str).str.strip()
    frame["split"] = frame["split"].astype(str)
    if mask_column in frame.columns:
        frame[mask_column] = frame[mask_column].map(lambda v: resolve(v, optional=True))
    if require_mask:
        if mask_column not in frame.columns:
            raise ValueError(
                f"band-* 백본은 manifest에 {mask_column!r} 컬럼이 필요합니다 "
                f"(Stage 0의 make_band_masks.py 산출물을 쓰세요): {manifest_path}"
            )
        missing = [v for v in frame[mask_column] if not v or not Path(v).is_file()]
        if missing:
            raise FileNotFoundError(
                f"band mask가 없는 이미지 {len(missing)}건 (예: {missing[:3]})"
            )
    return frame.reset_index(drop=True)


class BackboneDataset:
    """이미지(+band mask)를 같은 기하로 augment. mask는 patch grid로 pooling해 반환."""

    def __init__(
        self,
        frame: pd.DataFrame,
        class_to_idx: Dict[str, int],
        image_size: int,
        *,
        train: bool,
        patch_grid: tuple[int, int] | None = None,
        mask_column: str = "mask_path",
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.image_size = image_size
        self.train = train
        self.patch_grid = patch_grid          # None이면 mask를 반환하지 않음 (band 미사용)
        self.mask_column = mask_column
        self.mean, self.std = tuple(mean), tuple(std)
        if train:
            from torchvision import transforms

            self.color_jitter = transforms.ColorJitter(brightness=0.15, contrast=0.15)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        import torch.nn.functional as functional
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as vision

        row = self.frame.iloc[index]
        with Image.open(row["path"]) as source:
            image = source.convert("RGB")
        image = pad_to_square(image, fill=0)
        size = [self.image_size, self.image_size]

        band = None
        if self.patch_grid is not None:
            with Image.open(row[self.mask_column]) as source:
                band = source.convert("L")
            if band.width != band.height:
                band = pad_to_square(band, fill=0)
            band = vision.resize(band, size, interpolation=InterpolationMode.NEAREST)

        image = vision.resize(image, size, interpolation=InterpolationMode.BILINEAR)

        if self.train:
            # 이미지와 mask에 동일한 기하 변환을 적용해야 attention prior가 맞는다
            if random.random() < 0.5:
                image = vision.hflip(image)
                if band is not None:
                    band = vision.hflip(band)
            angle = random.uniform(-10.0, 10.0)
            image = vision.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            if band is not None:
                band = vision.rotate(band, angle, interpolation=InterpolationMode.NEAREST)
            image = self.color_jitter(image)

        image_tensor = vision.normalize(vision.to_tensor(image), self.mean, self.std)
        label = self.class_to_idx[str(row["label"])]
        if band is None:
            return image_tensor, label
        band_patches = functional.adaptive_avg_pool2d(
            vision.to_tensor(band).unsqueeze(0), self.patch_grid
        ).flatten()
        return image_tensor, label, band_patches


def make_loader(dataset, batch_size: int, num_workers: int, *, shuffle: bool):
    import torch

    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return torch.utils.data.DataLoader(dataset, **kwargs)


# =====================================================================
# 모델 생성 + finetune 범위 + attention 캡처
# =====================================================================

@dataclass(frozen=True)
class BackboneSpec:
    """백본 하나의 정의. name은 base.yaml backbones 레지스트리 키와 같다."""

    name: str
    timm_model: str
    use_band: bool
    image_size: int = 224

    @classmethod
    def from_registry(cls, name: str, registry: Dict[str, dict]) -> "BackboneSpec":
        if name not in registry:
            raise KeyError(
                f"backbone {name!r}이 레지스트리에 없습니다. 정의된 이름: {sorted(registry)}"
            )
        entry = registry[name]
        if "timm" not in entry:
            raise ValueError(f"backbone {name!r} 항목에 timm 모델명이 없습니다")
        return cls(
            name=name,
            timm_model=entry["timm"],
            use_band=name.startswith("band-"),   # band-* 이면 band-attention 학습
            image_size=int(entry.get("image_size", 224)),
        )


def create_model(spec: BackboneSpec, num_classes: int, *, pretrained: bool = True,
                 dropout: float = 0.1, drop_path_rate: float = 0.1,
                 grad_checkpointing: bool = False):
    import timm

    kwargs = dict(pretrained=pretrained, num_classes=num_classes,
                  drop_rate=dropout, drop_path_rate=drop_path_rate)
    # dinov2(patch14) 등 기본 해상도가 다른 ViT는 img_size로 맞춰준다.
    # img_size를 받지 않는 아키텍처(CNN 등)는 인자 없이 재시도.
    try:
        model = timm.create_model(spec.timm_model, img_size=spec.image_size, **kwargs)
    except TypeError:
        log(f"  {spec.timm_model}: img_size 인자 미지원 → 모델 기본 해상도 사용")
        model = timm.create_model(spec.timm_model, **kwargs)
    if grad_checkpointing:
        # large/giant 백본의 활성값 메모리를 줄인다 (속도는 ~20-30% 느려짐)
        if hasattr(model, "set_grad_checkpointing"):
            model.set_grad_checkpointing(True)
            log("  gradient checkpointing 켬")
        else:
            log(f"  {spec.timm_model}: gradient checkpointing 미지원 → 건너뜀")
    return model


def describe_model(model, spec: BackboneSpec) -> dict:
    """백본 규모 요약 — 큰 모델을 쓸 때 배치/메모리 감을 잡기 위해."""
    blocks = getattr(model, "blocks", None)
    return {
        "backbone": spec.name,
        "timm_model": spec.timm_model,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "embed_dim": int(getattr(model, "num_features", 0)),
        "depth": len(blocks) if blocks is not None else None,
        "image_size": spec.image_size,
    }


def apply_finetune_mode(model, mode: str, trainable_blocks: int) -> dict:
    """전체 finetuning ↔ 마지막 n개 block만 학습(나머지 frozen).

    mode: full | partial
    """
    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {"mode": "full", "trainable_blocks": None,
                "trainable_parameters": int(trainable)}

    if mode != "partial":
        raise ValueError("finetune_mode must be full or partial")
    blocks = getattr(model, "blocks", None)
    if blocks is None or not len(blocks):
        raise TypeError("partial finetuning requires a ViT-style model with .blocks")
    if trainable_blocks <= 0:
        raise ValueError("partial.trainable_blocks must be positive")

    for p in model.parameters():          # 일단 전부 동결
        p.requires_grad = False
    n = min(trainable_blocks, len(blocks))
    for block in blocks[-n:]:             # 마지막 n개 block만 해제
        for p in block.parameters():
            p.requires_grad = True
    # 최종 norm과 분류 head는 항상 학습 (frozen이면 학습이 되지 않음)
    for attr in ("norm", "fc_norm", "head_norm", "head"):
        module = getattr(model, attr, None)
        if module is not None and hasattr(module, "parameters"):
            for p in module.parameters():
                p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"mode": "partial", "trainable_blocks": n, "total_blocks": len(blocks),
            "trainable_parameters": int(trainable)}


def patch_grid_size(model) -> tuple[int, int]:
    grid = getattr(getattr(model, "patch_embed", None), "grid_size", None)
    if grid is None:
        raise TypeError("band attention requires a ViT-style model with patch_embed.grid_size")
    if isinstance(grid, int):
        return grid, grid
    if len(grid) != 2:
        raise ValueError(f"expected a 2D patch grid, got {grid!r}")
    return int(grid[0]), int(grid[1])


def attach_attention_capture(model, num_blocks: int = 6):
    """마지막 num_blocks개 block의 attention 확률을 forward hook으로 캡처."""
    blocks = getattr(model, "blocks", None)
    if blocks is None or not len(blocks):
        raise TypeError("band attention requires a ViT-style model with transformer blocks")
    if num_blocks <= 0:
        raise ValueError("attention_blocks must be positive")

    for handle in getattr(model, "_band_attention_handles", []):
        handle.remove()
    stores: list[dict[str, Any]] = []
    handles = []
    for block in blocks[-min(num_blocks, len(blocks)):]:
        attention = getattr(block, "attn", None)
        attention_drop = getattr(attention, "attn_drop", None)
        if attention_drop is None:
            raise TypeError("band attention requires blocks with attn.attn_drop")
        if hasattr(attention, "fused_attn"):
            attention.fused_attn = False       # fused 경로는 확률을 노출하지 않음
        store: dict[str, Any] = {}

        def capture(_module, inputs, _output, *, target=store):
            target["attention"] = inputs[0]

        handles.append(attention_drop.register_forward_hook(capture))
        stores.append(store)
    model._band_attention_stores = stores
    model._band_attention_handles = handles
    return model


def get_cls_patch_attention(model):
    """캡처한 block들의 CLS→patch attention을 head/block 평균 후 합=1로 정규화."""
    import torch

    stores = getattr(model, "_band_attention_stores", None)
    if not stores:
        raise RuntimeError("attention capture is not attached to this model")
    prefix_tokens = int(getattr(model, "num_prefix_tokens", 1))
    values = []
    for store in stores:
        attention = store.get("attention")
        if attention is None:
            raise RuntimeError("run a model forward pass before reading attention")
        if attention.ndim != 4:
            raise ValueError(f"expected [batch, heads, tokens, tokens], got {attention.shape}")
        values.append(attention.mean(dim=1)[:, 0, prefix_tokens:])
    result = torch.stack(values).mean(dim=0)
    return result / result.sum(dim=1, keepdim=True).clamp_min(1e-8)


def band_attention_loss(model, band_masks, alpha: float):
    """CLS→patch attention을 band 쪽으로 유도하는 KL prior loss."""
    import torch.nn.functional as functional

    attention = get_cls_patch_attention(model).float()
    band_masks = band_masks.float()
    if attention.shape != band_masks.shape:
        raise ValueError(
            f"attention/mask patch 수 불일치: attention={tuple(attention.shape)}, "
            f"mask={tuple(band_masks.shape)}"
        )
    prior = 1.0 + alpha * band_masks
    prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return functional.kl_div(attention.clamp_min(1e-8).log(), prior, reduction="batchmean")


# =====================================================================
# 백본 저장 — 다른 스테이지가 끌어다 쓰는 산출물
# =====================================================================

HEAD_PREFIXES = ("head.", "fc.", "classifier.")


def save_backbone(path: str | Path, model, spec: BackboneSpec, metadata: dict) -> Path:
    """분류 head를 제외한 백본 weight만 저장.

    Stage 2(metric) / Stage 3(classifier)은 이 파일을 num_classes=0 백본에 로드한다.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith(HEAD_PREFIXES)
    }
    torch.save(
        {
            "model_state_dict": state,
            "backbone": spec.name,
            "timm_model": spec.timm_model,
            "image_size": spec.image_size,
            "use_band": spec.use_band,
            **metadata,
        },
        path,
    )
    return path


def class_weights(frame: pd.DataFrame, classes: Sequence[str], *, split: str = "train"):
    """클래스 빈도의 역수를 가중치로. 이미지가 적은 클래스가 loss에서 밀리지 않게 한다.

    합이 클래스 수가 되도록 정규화하므로 전체 loss 스케일은 유지된다.
    출처: implantsj src/data.py 의 class_weights_from_manifest
    """
    import torch

    counts = frame[frame["split"] == split]["label"].astype(str).value_counts().to_dict()
    weights = np.asarray([1.0 / max(1, counts.get(c, 0)) for c in classes], dtype=np.float32)
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(y_true, y_pred, class_names: Sequence[str]) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    labels = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }
