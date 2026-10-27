"""Fine-tuned SAM2 + Adapter + U-Net inference without writing mask files."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SegmenterConfig:
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_t.yaml"
    # 기본값은 프로젝트 상대경로. 실제 경로는 yaml(04_masked_vit.yaml의 sam.*)에서 주고,
    # 평가 때는 체크포인트 메타데이터 또는 --sam2-checkpoint / --sam-model-checkpoint로 덮어쓴다.
    sam2_checkpoint: str = "checkpoints/sam2/sam2.1_hiera_tiny.pt"
    model_checkpoint: str = "checkpoints/best_sam2.1_tiny_adapter_unet.pth"
    input_size: int = 512
    threshold: float = 0.5
    morph_kernel_size: int = 3
    batch_size: int = 2

    def validate(self) -> None:
        for name in ("sam2_checkpoint", "model_checkpoint"):
            path = Path(getattr(self, name)).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"{name}: {path}")
        if self.input_size <= 0 or self.batch_size <= 0:
            raise ValueError("segmenter input_size and batch_size must be positive")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("segmenter threshold must be between zero and one")
        if self.morph_kernel_size <= 0:
            raise ValueError("morph_kernel_size must be positive")

    def metadata(self) -> dict:
        payload = asdict(self)
        payload["sam2_checkpoint"] = str(Path(self.sam2_checkpoint).resolve())
        payload["model_checkpoint"] = str(Path(self.model_checkpoint).resolve())
        payload["preprocessing"] = "direct_resize_512_rgb_float_0_1"
        payload["source_implementation"] = "make_mask/SAM_mask_produce_util.py"
        return payload


def _extract_features(features):
    if isinstance(features, dict):
        if "vision_features" in features:
            features = features["vision_features"]
        elif "backbone_fpn" in features:
            features = features["backbone_fpn"][-1]
        else:
            features = list(features.values())[-1]
    if isinstance(features, (list, tuple)):
        features = features[-1]
    return features


def _model_classes():
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    class Adapter(nn.Module):
        def __init__(self, in_channels: int = 256, out_channels: int = 256):
            super().__init__()
            self.adapter = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, value):
            return self.adapter(value)

    class UNetDecoder(nn.Module):
        def __init__(self, in_channels: int = 256):
            super().__init__()
            
            self.up1 = self.block(in_channels, 128)
            self.up2 = self.block(128, 64)
            self.up3 = self.block(64, 32)
            self.up4 = self.block(32, 16)
            self.final = nn.Conv2d(16, 1, kernel_size=1)

        @staticmethod
        def block(in_channels: int, out_channels: int):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, value):
            value = self.up1(value)
            value = self.up2(value)
            value = self.up3(value)
            value = self.up4(value)
            return self.final(value)

    class SAM2AdapterUNet(nn.Module):
        def __init__(self, sam2_model, output_size: int):
            super().__init__()
            self.sam2_encoder = sam2_model.image_encoder
            self.adapter = Adapter(256, 256)
            self.decoder = UNetDecoder(256)
            self.output_size = output_size
            for parameter in self.sam2_encoder.parameters():
                parameter.requires_grad = False

        def forward(self, value):
            with torch.no_grad():
                features = _extract_features(self.sam2_encoder(value))
            value = self.adapter(features)
            value = self.decoder(value)
            return functional.interpolate(
                value,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )

    return SAM2AdapterUNet


class RuntimeSegmenter:
    """Load the frozen segmentation model once and predict masks in batches."""

    def __init__(self, config: SegmenterConfig, device: str):
        import torch
        from sam2.build_sam import build_sam2

        config.validate()
        self.config = config
        self.device = device
        sam2 = build_sam2(
            config.sam2_config,
            str(Path(config.sam2_checkpoint).resolve()),
            device=device,
        )
        model_class = _model_classes()
        self.model = model_class(sam2, config.input_size).to(device)
        try:
            state = torch.load(
                Path(config.model_checkpoint).resolve(),
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            state = torch.load(
                Path(config.model_checkpoint).resolve(), map_location="cpu"
            )
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def _tensor(self, image: Image.Image):
        import torch

        rgb = np.asarray(image.convert("RGB"))
        resized = cv2.resize(
            rgb,
            (self.config.input_size, self.config.input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        value = resized.astype(np.float32) / 255.0
        return torch.from_numpy(value).permute(2, 0, 1).contiguous()

    def _postprocess(self, probability: np.ndarray, size: tuple[int, int]):
        width, height = size
        probability = cv2.resize(
            probability, (width, height), interpolation=cv2.INTER_LINEAR
        )
        mask = (probability > self.config.threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.morph_kernel_size, self.config.morph_kernel_size),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        filled = np.zeros_like(mask)
        if contours:
            cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(filled)
        if count <= 1:
            return np.zeros_like(mask)
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return np.where(labels == largest, 255, 0).astype(np.uint8)

    def predict(self, images: Sequence[Image.Image]) -> list[np.ndarray]:
        import torch

        if not images:
            return []
        tensors = torch.stack([self._tensor(image) for image in images]).to(
            self.device
        )
        with torch.inference_mode():
            probabilities = torch.sigmoid(self.model(tensors)).cpu().numpy()[:, 0]
        return [
            self._postprocess(probability, image.size)
            for probability, image in zip(probabilities, images)
        ]

    def close(self) -> None:
        import gc
        import torch

        self.model.to("cpu")
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

