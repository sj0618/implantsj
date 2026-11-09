"""metric 패키지 공용 헬퍼 — manifest/데이터/샘플러/지표.

implantsj의 src/data.py, src/utils.py, src/metrics.py, src/experiment_data.py에서
metric learning에 필요한 부분만 흡수 (로직 동일). 다른 패키지를 import하지 않는다.

manifest 계약: CSV에 path,label,split 컬럼 (split ∈ {train,valid,test}).
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image


# =====================================================================
# utils
# =====================================================================

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def as_list(value):
    """설정값을 항상 리스트로 — 하나만 실험할 때 리스트를 벗겨 써도 되게 한다.

    `distances: cosine` 처럼 스칼라로 적어도 글자 단위로 순회되지 않는다.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


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
# manifest / dataset / loader
# =====================================================================

def load_manifest(manifest_csv: str | Path, split: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    if "path" not in df.columns or "label" not in df.columns:
        raise ValueError("Manifest must have at least path,label columns.")
    if split is not None:
        if "split" not in df.columns:
            raise ValueError("Manifest has no split column. Build splits first.")
        df = df[df["split"].astype(str) == split].copy()
    df["label"] = df["label"].astype(str)
    return df.reset_index(drop=True)


def label_mapping(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    classes = sorted(df["label"].astype(str).unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    return class_to_idx, idx_to_class


def build_transforms(image_size: int = 224, train: bool = False, strong_aug: bool = True):
    from torchvision import transforms

    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if train:
        aug = [transforms.Resize((image_size, image_size))]
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
        path = Path(row["path"])
        label = str(row["label"])
        # 갤러리/학습 클래스에 없는 라벨은 -1. open-set 평가는 정의상 학습에
        # 없던 클래스를 만나므로 KeyError 대신 -1로 처리한다.
        y = self.class_to_idx.get(label, -1)
        with Image.open(path) as opened:
            img = opened.convert("RGB")
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
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


# =====================================================================
# P×K batch sampler (metric loss용: P개 클래스 × K장씩)
# =====================================================================

class PKBatchSampler:
    """Sample P labels × K distinct images for metric-learning batches."""

    def __init__(
        self, labels: Sequence[str], batch_size: int, samples_per_class: int, seed: int
    ) -> None:
        if samples_per_class < 2:
            raise ValueError("samples_per_class must be >= 2")
        if batch_size % samples_per_class:
            raise ValueError("batch_size must be divisible by samples_per_class")
        self.batch_size = batch_size
        self.k = samples_per_class
        self.p = batch_size // samples_per_class
        self.seed = seed
        self.epoch = 0
        self.by_label: dict[str, list[int]] = {}
        for index, label in enumerate(labels):
            self.by_label.setdefault(str(label), []).append(index)
        too_small = {
            label: len(indices)
            for label, indices in self.by_label.items()
            if len(indices) < self.k
        }
        if too_small:
            raise ValueError(
                f"metric loss needs at least K distinct images per class; too small={too_small}"
            )
        if len(self.by_label) < self.p:
            raise ValueError(
                f"batch requires P={self.p} classes but data has {len(self.by_label)}; "
                "reduce batch_size or increase samples_per_class"
            )
        self.num_batches = max(1, math.ceil(len(labels) / batch_size))

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        class_names = sorted(self.by_label)
        class_pool = class_names.copy()
        rng.shuffle(class_pool)
        for _ in range(self.num_batches):
            if len(class_pool) >= self.p:
                selected_labels = class_pool[: self.p]
                class_pool = class_pool[self.p:]
            else:
                selected_labels = class_pool
                candidates = [label for label in class_names if label not in selected_labels]
                rng.shuffle(candidates)
                needed = self.p - len(selected_labels)
                selected_labels += candidates[:needed]
                class_pool = candidates[needed:]
            batch = [
                index
                for label in selected_labels
                for index in rng.sample(self.by_label[label], self.k)
            ]
            rng.shuffle(batch)
            yield batch


def resolve_pk_batch_size(labels, requested_batch_size: int, samples_per_class: int) -> int:
    if samples_per_class < 2:
        raise ValueError("samples_per_class must be at least 2")
    if requested_batch_size < samples_per_class or requested_batch_size % samples_per_class:
        raise ValueError(
            f"batch_size={requested_batch_size} must be a positive multiple of "
            f"samples_per_class={samples_per_class}"
        )
    class_count = len(set(str(label) for label in labels))
    if not class_count:
        raise ValueError("metric training data has no labels")
    return min(requested_batch_size, class_count * samples_per_class)


# =====================================================================
# metrics
# =====================================================================

def compute_metrics(y_true, y_pred, class_names: Sequence[str]) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(class_names)))
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }
    out["classification_report"] = classification_report(
        y_true, y_pred, labels=labels, target_names=list(class_names),
        zero_division=0, output_dict=True,
    )
    return out


def save_confusion_matrix(path_csv: str | Path, path_png: str | Path, y_true, y_pred, class_names) -> None:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    path_csv = Path(path_csv)
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(path_csv)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(max(6, len(class_names) * 0.9), max(5, len(class_names) * 0.8)))
        ax = fig.add_subplot(111)
        im = ax.imshow(cm)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion matrix")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path_png = Path(path_png)
        path_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_png, dpi=180)
        plt.close(fig)
    except Exception as e:
        print("Warning: could not save confusion matrix image:", repr(e))


def save_predictions(path_csv: str | Path, rows: list[dict]) -> None:
    path = Path(path_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
