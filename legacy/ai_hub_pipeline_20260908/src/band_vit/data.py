"""Manifest datasets and loaders for supervised ViT classification."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps


def square_padding(size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Return centered left/top/right/bottom padding for a square canvas."""
    width, height = size
    side = max(width, height)
    horizontal = side - width
    vertical = side - height
    left = horizontal // 2
    top = vertical // 2
    return left, top, horizontal - left, vertical - top


def pad_to_square(image: Image.Image, fill=0) -> Image.Image:
    return ImageOps.expand(image, border=square_padding(image.size), fill=fill)


class SquarePad:
    def __init__(self, fill=0) -> None:
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        return pad_to_square(image, fill=self.fill)


def _resolve_path(value, manifest_path: Path, *, optional: bool = False) -> str:
    if pd.isna(value) or not str(value).strip():
        if optional:
            return ""
        raise ValueError(f"empty path in manifest {manifest_path}")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return str(path.resolve())


def read_manifest(path: str | Path) -> pd.DataFrame:
    manifest_path = Path(path).expanduser().resolve()
    frame = pd.read_csv(manifest_path)
    if not {"path", "label"}.issubset(frame.columns):
        raise ValueError(f"manifest {manifest_path} must contain path,label")
    if frame.empty:
        raise ValueError(f"manifest {manifest_path} is empty")

    frame = frame.copy()
    frame["path"] = frame["path"].map(
        lambda value: _resolve_path(value, manifest_path)
    )
    frame["label"] = frame["label"].astype(str).str.strip()
    if frame["label"].eq("").any():
        raise ValueError(f"manifest {manifest_path} contains empty labels")
    for column in (
        name for name in frame.columns if name != "path" and name.endswith("_path")
    ):
        frame[column] = frame[column].map(
            lambda value: _resolve_path(value, manifest_path, optional=True)
        )
    if frame["path"].duplicated().any():
        raise ValueError(f"manifest {manifest_path} contains duplicate image paths")
    return frame.reset_index(drop=True)


class ManifestDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        label_to_index: dict[str, int],
        transform,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.label_to_index = label_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row["path"]) as source:
            image = self.transform(source.convert("RGB"))
        label = str(row["label"])
        return image, self.label_to_index[label], label, str(row["path"])


class BandManifestDataset(ManifestDataset):
    """Training dataset with synchronized image and boundary-mask geometry."""

    def __init__(
        self,
        frame: pd.DataFrame,
        label_to_index: dict[str, int],
        image_size: int,
        patch_grid: tuple[int, int],
        *,
        mask_column: str = "mask_path",
        strong_aug: bool = True,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__(frame, label_to_index, transform=None)
        if mask_column not in self.frame.columns:
            raise ValueError(
                f"band training requires a {mask_column!r} column "
                "in the training manifest"
            )
        missing = [
            str(value)
            for value in self.frame[mask_column]
            if not value or not Path(str(value)).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "band training requires one mask per training image; "
                f"missing={len(missing)}, examples={missing[:5]}"
            )

        from torchvision import transforms

        self.image_size = image_size
        self.patch_grid = patch_grid
        self.mask_column = mask_column
        self.strong_aug = strong_aug
        self.mean = mean
        self.std = std
        self.color_jitter = transforms.ColorJitter(brightness=0.15, contrast=0.15)

    def __getitem__(self, index: int):
        import torch.nn.functional as functional
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as vision

        row = self.frame.iloc[index]
        with Image.open(row["path"]) as source:
            image = source.convert("RGB")
        with Image.open(row[self.mask_column]) as source:
            band = source.convert("L")

        image = pad_to_square(image, fill=0)
        if band.width != band.height:
            band = pad_to_square(band, fill=0)
        size = [self.image_size, self.image_size]
        image = vision.resize(image, size, interpolation=InterpolationMode.BILINEAR)
        band = vision.resize(band, size, interpolation=InterpolationMode.NEAREST)
        if self.strong_aug:
            if random.random() < 0.5:
                image, band = vision.hflip(image), vision.hflip(band)
            angle = random.uniform(-10.0, 10.0)
            image = vision.rotate(
                image, angle, interpolation=InterpolationMode.BILINEAR
            )
            band = vision.rotate(
                band, angle, interpolation=InterpolationMode.NEAREST
            )
            image = self.color_jitter(image)

        image_tensor = vision.normalize(vision.to_tensor(image), self.mean, self.std)
        band_tensor = vision.to_tensor(band).unsqueeze(0)
        band_patches = functional.adaptive_avg_pool2d(
            band_tensor, self.patch_grid
        ).flatten()
        label = str(row["label"])
        return (
            image_tensor,
            self.label_to_index[label],
            label,
            str(row["path"]),
            band_patches,
        )


@dataclass(frozen=True)
class DataConfig:
    train_path: str
    valid_path: str
    test_path: str | None
    batch_size: int
    num_workers: int


@dataclass(frozen=True)
class PreparedManifests:
    frames: dict[str, pd.DataFrame]
    labels: list[str]
    label_to_index: dict[str, int]


@dataclass
class ExperimentData:
    labels: list[str]
    label_to_index: dict[str, int]
    train_loader: object
    valid_loader: object
    test_loader: object | None


def prepare_manifests(config: DataConfig) -> PreparedManifests:
    frames = {
        "train": read_manifest(config.train_path),
        "valid": read_manifest(config.valid_path),
    }
    if config.test_path:
        frames["test"] = read_manifest(config.test_path)

    labels = sorted(frames["train"]["label"].unique().tolist())
    label_to_index = {label: index for index, label in enumerate(labels)}
    missing_valid = sorted(set(labels) - set(frames["valid"]["label"]))
    if missing_valid:
        raise ValueError(
            f"validation must contain every training label; missing={missing_valid}"
        )
    for split in ("valid", "test"):
        if split not in frames:
            continue
        extra = sorted(set(frames[split]["label"]) - set(labels))
        if extra:
            raise ValueError(
                f"{split} contains labels absent from training; extra={extra}"
            )
    return PreparedManifests(frames, labels, label_to_index)


def _loader(dataset, batch_size: int, num_workers: int, *, shuffle: bool):
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


def create_data(
    config: DataConfig,
    prepared: PreparedManifests,
    train_transform,
    eval_transform,
    *,
    band_patch_grid: tuple[int, int] | None = None,
    band_image_size: int = 224,
    band_mask_column: str = "mask_path",
    band_strong_aug: bool = True,
    band_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    band_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> ExperimentData:
    datasets = {
        split: ManifestDataset(
            frame,
            prepared.label_to_index,
            train_transform if split == "train" else eval_transform,
        )
        for split, frame in prepared.frames.items()
    }
    if band_patch_grid is not None:
        datasets["train"] = BandManifestDataset(
            prepared.frames["train"],
            prepared.label_to_index,
            image_size=band_image_size,
            patch_grid=band_patch_grid,
            mask_column=band_mask_column,
            strong_aug=band_strong_aug,
            mean=band_mean,
            std=band_std,
        )
    return ExperimentData(
        labels=prepared.labels,
        label_to_index=prepared.label_to_index,
        train_loader=_loader(
            datasets["train"], config.batch_size, config.num_workers, shuffle=True
        ),
        valid_loader=_loader(
            datasets["valid"], config.batch_size, config.num_workers, shuffle=False
        ),
        test_loader=(
            _loader(
                datasets["test"], config.batch_size, config.num_workers, shuffle=False
            )
            if "test" in datasets
            else None
        ),
    )
