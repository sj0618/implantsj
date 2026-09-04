"""Convert a predicted implant mask into a tight boundary-band-only image."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage


@dataclass(frozen=True)
class BandCropConfig:
    working_size: int = 224
    outer_radius: int = 15
    inner_radius: int = 15
    crop_margin: int = 4
    output_size: int = 224

    def validate(self) -> None:
        if self.working_size <= 0 or self.output_size <= 0:
            raise ValueError("band working/output sizes must be positive")
        if min(self.outer_radius, self.inner_radius, self.crop_margin) < 0:
            raise ValueError("band radii and crop margin must be non-negative")

    def metadata(self) -> dict:
        return {
            **asdict(self),
            "definition": "dilation(mask, outer) AND NOT erosion(mask, inner)",
            "background": "black_zero",
            "crop": "tight_band_bbox_then_square_pad",
        }


def ellipse(radius: int) -> np.ndarray:
    if radius == 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x / radius) ** 2 + (y / radius) ** 2 <= 1.0


def _square_pad(image: Image.Image, fill=0) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    horizontal = side - width
    vertical = side - height
    return ImageOps.expand(
        image,
        border=(
            horizontal // 2,
            vertical // 2,
            horizontal - horizontal // 2,
            vertical - vertical // 2,
        ),
        fill=fill,
    )


INPUT_MODES = ("band", "original")


def _band_geometry(
    image: Image.Image,
    mask: np.ndarray,
    config: BandCropConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Square-pad/resize image+mask to working size and compute the band."""
    config.validate()
    if mask.ndim != 2 or mask.shape != (image.height, image.width):
        raise ValueError(
            f"mask/image mismatch: mask={mask.shape}, image={image.size[::-1]}"
        )
    padded_image = _square_pad(image.convert("RGB"), fill=0).resize(
        (config.working_size, config.working_size), Image.Resampling.BILINEAR
    )
    padded_mask = _square_pad(Image.fromarray(mask, mode="L"), fill=0).resize(
        (config.working_size, config.working_size), Image.Resampling.NEAREST
    )
    binary = np.asarray(padded_mask) > 127
    outer = ndimage.binary_dilation(binary, structure=ellipse(config.outer_radius))
    inner = ndimage.binary_erosion(binary, structure=ellipse(config.inner_radius))
    band = np.logical_and(outer, np.logical_not(inner))
    rgb = np.asarray(padded_image, dtype=np.uint8)
    return rgb, binary, band


def _band_bbox(band: np.ndarray, config: BandCropConfig) -> tuple[int, int, int, int]:
    ys, xs = np.where(band)
    margin = config.crop_margin
    left = max(0, int(xs.min()) - margin)
    top = max(0, int(ys.min()) - margin)
    right = min(config.working_size, int(xs.max()) + 1 + margin)
    bottom = min(config.working_size, int(ys.max()) + 1 + margin)
    return left, top, right, bottom


def _finalize(pixels: np.ndarray, config: BandCropConfig, resample) -> np.ndarray:
    mode = "RGB" if pixels.ndim == 3 else "L"
    out = _square_pad(Image.fromarray(pixels, mode=mode), fill=0).resize(
        (config.output_size, config.output_size), resample
    )
    return np.asarray(out, dtype=np.uint8)


def create_input_crop(
    image: Image.Image,
    mask: np.ndarray,
    config: BandCropConfig,
    input_mode: str = "band",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build a classifier input plus an aligned band mask at output size.

    input_mode:
      band     -- band-only pixels, tight band bbox crop (training input).
      original -- unmasked pixels, same band bbox crop, so the implant sits
                  at the identical position/scale as the training input.
    Returns (crop_rgb_uint8, band_mask_uint8, stats). band_mask marks the
    boundary band in the SAME geometry as the returned crop, for measuring
    how much attention falls on the band.
    """
    if input_mode not in INPUT_MODES:
        raise ValueError(f"input_mode must be one of {INPUT_MODES}")
    rgb, binary, band = _band_geometry(image, mask, config)
    mask_ratio = float(binary.mean())
    band_ratio = float(band.mean())

    if not binary.any():
        raise ValueError("predicted segmentation mask is empty")
    if not band.any():
        raise ValueError("boundary band is empty")
    left, top, right, bottom = _band_bbox(band, config)

    if input_mode == "band":
        pixels = np.zeros_like(rgb)
        pixels[band] = rgb[band]
    else:  # original
        pixels = rgb
    crop = _finalize(pixels[top:bottom, left:right], config, Image.Resampling.BILINEAR)
    band_out = _finalize(
        (band[top:bottom, left:right].astype(np.uint8) * 255),
        config,
        Image.Resampling.NEAREST,
    )
    stats = {
        "mask_area_ratio": mask_ratio,
        "band_area_ratio": band_ratio,
        "crop_left": left,
        "crop_top": top,
        "crop_right": right,
        "crop_bottom": bottom,
        "crop_width": right - left,
        "crop_height": bottom - top,
    }
    return crop, band_out, stats


def create_band_only_crop(
    image: Image.Image,
    mask: np.ndarray,
    config: BandCropConfig,
) -> tuple[np.ndarray, dict]:
    """Backward-compatible band-only crop (training path)."""
    crop, _, stats = create_input_crop(image, mask, config, input_mode="band")
    return crop, stats
