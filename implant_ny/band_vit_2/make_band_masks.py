"""Create boundary-band masks and add their paths to a training manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from .data import pad_to_square


def ellipse(radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("band radii must be non-negative")
    if radius == 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x / radius) ** 2 + (y / radius) ** 2 <= 1.0


def create_band_masks(args: argparse.Namespace) -> Path:
    manifest_path = Path(args.manifest).expanduser().resolve()
    mask_root = Path(args.mask_root).expanduser().resolve()
    output_root = Path(args.out_root).expanduser().resolve()
    output_manifest = Path(args.out_manifest).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    skip_splits = {
        value.strip().lower()
        for value in args.skip_splits.split(",")
        if value.strip()
    }
    mask_index = {
        path.name[: -len("_pred.png")]: path
        for path in mask_root.rglob("*_pred.png")
    }
    outer_kernel = ellipse(args.outer)
    inner_kernel = ellipse(args.inner)

    frame = pd.read_csv(manifest_path)
    if not {"path", "label"}.issubset(frame.columns):
        raise ValueError("manifest must contain path,label")
    band_paths: list[str] = []
    missing = 0
    skipped = 0
    for row in frame.itertuples(index=False):
        split = str(getattr(row, "split", "")).strip().lower()
        if split in skip_splits:
            band_paths.append("")
            skipped += 1
            continue
        image_path = Path(str(row.path))
        source_mask = mask_index.get(image_path.stem)
        if source_mask is None:
            band_paths.append("")
            missing += 1
            continue
        try:
            with Image.open(source_mask) as source:
                mask = pad_to_square(source.convert("L"), fill=0).resize(
                    (args.size, args.size), resample=Image.Resampling.NEAREST
                )
        except (OSError, ValueError):
            band_paths.append("")
            missing += 1
            continue
        binary = np.asarray(mask) > 127
        outer = ndimage.binary_dilation(binary, structure=outer_kernel)
        inner = ndimage.binary_erosion(binary, structure=inner_kernel)
        band = np.logical_and(outer, np.logical_not(inner)).astype(np.uint8) * 255
        band_image = Image.fromarray(band, mode="L")
        label = str(row.label).replace("/", "_")
        output_path = output_root / (
            f"{split}_{label}_{image_path.stem}_band.png"
        )
        band_image.save(output_path)
        band_paths.append(str(output_path.resolve()))

    frame[args.mask_column] = band_paths
    frame.to_csv(output_manifest, index=False)
    print(
        f"wrote={output_manifest} generated={len(frame) - missing - skipped} "
        f"missing={missing} skipped={skipped}"
    )
    return output_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--outer", type=int, default=15)
    parser.add_argument("--inner", type=int, default=15)
    parser.add_argument("--mask-column", default="mask_path")
    parser.add_argument(
        "--skip-splits",
        default="test",
        help="comma-separated splits that do not need band masks",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    create_band_masks(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
