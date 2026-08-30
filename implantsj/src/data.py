from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _iter_images_under_class_folder(data_root: Path) -> List[dict]:
    rows: List[dict] = []
    for class_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        label = class_dir.name
        if label.lower() in {'train', 'valid', 'val', 'test', 'reference', 'references'}:
            continue
        for img in sorted(class_dir.rglob('*')):
            if img.is_file() and is_image(img):
                rows.append({'path': str(img), 'label': label, 'split': ''})
    return rows


def _iter_images_under_split_class_folder(data_root: Path) -> List[dict]:
    rows: List[dict] = []
    split_names = [('train', 'train'), ('valid', 'valid'), ('val', 'valid'), ('test', 'test')]
    for folder, split in split_names:
        split_dir = data_root / folder
        if not split_dir.exists():
            continue
        for class_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            label = class_dir.name
            for img in sorted(class_dir.rglob('*')):
                if img.is_file() and is_image(img):
                    rows.append({'path': str(img), 'label': label, 'split': split})
    return rows


def detect_layout(data_root: str | Path) -> str:
    root = Path(data_root)
    if any((root / s).is_dir() for s in ['train', 'valid', 'val', 'test']):
        return 'split_class_folder'
    return 'class_folder'


def build_manifest(
    data_root: str | Path,
    out_csv: str | Path,
    layout: str = 'auto',
    make_paths_absolute: bool = True,
) -> pd.DataFrame:
    root = Path(data_root).resolve()
    if layout == 'auto':
        layout = detect_layout(root)
    if layout == 'class_folder':
        rows = _iter_images_under_class_folder(root)
    elif layout == 'split_class_folder':
        rows = _iter_images_under_split_class_folder(root)
    else:
        raise ValueError(f'Unsupported layout: {layout}')
    if not rows:
        raise RuntimeError(f'No image files found under {root}')
    df = pd.DataFrame(rows)
    if make_paths_absolute:
        df['path'] = df['path'].map(lambda p: str(Path(p).resolve()))
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def make_stratified_splits(
    manifest_csv: str | Path,
    out_csv: str | Path,
    train: float = 8,
    valid: float = 1,
    test: float = 1,
    seed: int = 42,
    label_col: str = 'label',
) -> pd.DataFrame:
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(manifest_csv)
    if 'split' in df.columns and df['split'].fillna('').astype(str).str.len().gt(0).any():
        # Existing splits are intentionally overwritten only by this function.
        df = df.drop(columns=['split'])
    ratios = np.array([train, valid, test], dtype=float)
    ratios = ratios / ratios.sum()
    train_ratio, valid_ratio, test_ratio = ratios.tolist()

    labels = df[label_col].astype(str)
    try:
        train_df, temp_df = train_test_split(
            df,
            train_size=train_ratio,
            random_state=seed,
            stratify=labels,
            shuffle=True,
        )
        temp_labels = temp_df[label_col].astype(str)
        relative_valid = valid_ratio / (valid_ratio + test_ratio)
        valid_df, test_df = train_test_split(
            temp_df,
            train_size=relative_valid,
            random_state=seed,
            stratify=temp_labels,
            shuffle=True,
        )
    except ValueError as e:
        # For very tiny classes, sklearn stratification can fail.
        rng = np.random.default_rng(seed)
        indices = np.arange(len(df))
        rng.shuffle(indices)
        n_train = max(1, int(round(len(indices) * train_ratio)))
        n_valid = max(1, int(round(len(indices) * valid_ratio)))
        train_idx = indices[:n_train]
        valid_idx = indices[n_train:n_train + n_valid]
        test_idx = indices[n_train + n_valid:]
        train_df, valid_df, test_df = df.iloc[train_idx], df.iloc[valid_idx], df.iloc[test_idx]

    train_df = train_df.copy(); train_df['split'] = 'train'
    valid_df = valid_df.copy(); valid_df['split'] = 'valid'
    test_df = test_df.copy(); test_df['split'] = 'test'
    out_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    out_df = out_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    return out_df


def load_manifest(manifest_csv: str | Path, split: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    if 'path' not in df.columns or 'label' not in df.columns:
        raise ValueError('Manifest must have at least path,label columns.')
    if split is not None:
        if 'split' not in df.columns:
            raise ValueError('Manifest has no split column. Build splits first.')
        df = df[df['split'].astype(str) == split].copy()
    df['label'] = df['label'].astype(str)
    return df.reset_index(drop=True)


def label_mapping(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    classes = sorted(df['label'].astype(str).unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    return class_to_idx, idx_to_class


def build_transforms(image_size: int = 224, train: bool = False, strong_aug: bool = True):
    from torchvision import transforms

    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if train:
        aug = [
            transforms.Resize((image_size, image_size)),
        ]
        if strong_aug:
            aug.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ])
        aug.extend([transforms.ToTensor(), normalize])
        return transforms.Compose(aug)
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])


class ImageClassificationDataset:
    def __init__(
        self,
        manifest_csv: str | Path,
        split: Optional[str],
        class_to_idx: Optional[Dict[str, int]] = None,
        transform=None,
        return_path: bool = False,
    ):
        self.manifest_csv = str(manifest_csv)
        self.df = load_manifest(manifest_csv, split=split)
        if class_to_idx is None:
            class_to_idx, _ = label_mapping(load_manifest(manifest_csv, split=None))
        self.class_to_idx = class_to_idx
        self.idx_to_class = {i: c for c, i in class_to_idx.items()}
        self.transform = transform
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = Path(row['path'])
        label = str(row['label'])
        y = self.class_to_idx[label]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.return_path:
            return img, y, str(path)
        return img, y


def create_loader(
    manifest_csv: str | Path,
    split: str,
    class_to_idx: Dict[str, int],
    image_size: int,
    batch_size: int,
    num_workers: int,
    train: bool,
    strong_aug: bool = True,
    return_path: bool = False,
):
    import torch
    from torch.utils.data import DataLoader

    ds = ImageClassificationDataset(
        manifest_csv=manifest_csv,
        split=split,
        class_to_idx=class_to_idx,
        transform=build_transforms(image_size=image_size, train=train, strong_aug=strong_aug),
        return_path=return_path,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return loader


def class_weights_from_manifest(manifest_csv: str | Path, class_to_idx: Dict[str, int], split: str = 'train'):
    import torch
    df = load_manifest(manifest_csv, split=split)
    counts = df['label'].astype(str).value_counts().to_dict()
    weights = []
    for label, idx in sorted(class_to_idx.items(), key=lambda kv: kv[1]):
        weights.append(1.0 / max(1, counts.get(label, 0)))
    arr = np.array(weights, dtype=np.float32)
    arr = arr / arr.sum() * len(arr)
    return torch.tensor(arr, dtype=torch.float32)


def print_manifest_summary(manifest_csv: str | Path) -> None:
    df = pd.read_csv(manifest_csv)
    print('Manifest:', manifest_csv)
    print('Rows:', len(df))
    print('Labels:', sorted(df['label'].astype(str).unique().tolist()))
    print('Label counts:')
    print(df['label'].astype(str).value_counts().sort_index())
    if 'split' in df.columns:
        print('Split counts:')
        print(df['split'].astype(str).value_counts().sort_index())
        print('Split x label:')
        print(pd.crosstab(df['split'], df['label']))
