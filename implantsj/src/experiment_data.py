"""Data modules used by one training experiment."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
from PIL import Image


def read_manifest(path: str | Path, scope: str) -> pd.DataFrame:
    manifest_path = Path(path).expanduser().resolve()
    frame = pd.read_csv(manifest_path)
    if not {"path", "label"}.issubset(frame.columns):
        raise ValueError(f"manifest {manifest_path} must contain path,label")
    frame = frame.copy()
    frame["path"] = frame["path"].map(
        lambda value: str(
            (
                Path(str(value)).expanduser()
                if Path(str(value)).is_absolute()
                else manifest_path.parent / str(value)
            ).resolve()
        )
    )
    if scope == "closed":
        if "set" not in frame.columns:
            raise ValueError("closed-scope training requires a manifest set column")
        frame = frame[frame["set"].astype(str).str.lower() == "closed"].copy()
    elif scope != "all":
        raise ValueError("scope must be closed or all")
    if frame.empty:
        raise ValueError(f"manifest {manifest_path} has no rows after scope={scope}")
    return frame.reset_index(drop=True)


class ManifestDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        label_to_index: dict[str, int],
        transform,
        two_views: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.label_to_index = label_to_index
        self.transform = transform
        self.two_views = two_views

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row.path) as image:
            image = image.convert("RGB")
            value = (
                (self.transform(image), self.transform(image))
                if self.two_views
                else self.transform(image)
            )
        return (
            value,
            self.label_to_index.get(str(row.label), -1),
            str(row.label),
            str(row.path),
        )


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
                selected_labels = class_pool[:self.p]
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


@dataclass(frozen=True)
class ManifestDataConfig:
    train_path: str
    valid_path: str
    scope: str
    batch_size: int
    samples_per_class: int
    num_workers: int
    seed: int


@dataclass
class ExperimentData:
    labels: list[str]
    label_to_index: dict[str, int]
    train_loader: object
    valid_loader: object
    support_eval_loader: object
    valid_eval_loader: object
    train_sampler: PKBatchSampler | None

    def set_epoch(self, epoch: int) -> None:
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)


def _loader(dataset, batch_size: int, num_workers: int, *, shuffle=False, sampler=None):
    import torch

    kwargs = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    if sampler is not None:
        return torch.utils.data.DataLoader(dataset, batch_sampler=sampler, **kwargs)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, **kwargs
    )


class ManifestDataModule:
    """Build loaders from explicit train/validation/test CSV paths."""

    def __init__(self, config: ManifestDataConfig) -> None:
        self.config = config
        self.frames: dict[str, pd.DataFrame] = {}
        self.labels: list[str] = []
        self.label_to_index: dict[str, int] = {}

    def prepare(self) -> None:
        self.frames = {
            "train": read_manifest(self.config.train_path, self.config.scope),
            "valid": read_manifest(self.config.valid_path, self.config.scope),
        }
        self.labels = sorted(
            self.frames["train"]["label"].astype(str).unique().tolist()
        )
        self.label_to_index = {label: index for index, label in enumerate(self.labels)}
        missing = sorted(
            set(self.labels) - set(self.frames["valid"]["label"].astype(str))
        )
        if missing:
            raise ValueError(
                f"validation must cover training labels; missing={missing}"
            )

    def build_loaders(
        self,
        train_transform,
        eval_transform,
        *,
        two_views: bool,
        balanced_train: bool,
        balanced_eval: bool,
    ) -> ExperimentData:
        if not self.frames:
            self.prepare()
        datasets = {
            "train": ManifestDataset(
                self.frames["train"],
                self.label_to_index,
                train_transform,
                two_views=two_views,
            ),
            "valid": ManifestDataset(
                self.frames["valid"],
                self.label_to_index,
                eval_transform,
                two_views=two_views,
            ),
        }
        eval_datasets = {
            split: ManifestDataset(
                self.frames[split], self.label_to_index, eval_transform
            )
            for split in ("train", "valid")
        }
        samplers = {
            split: PKBatchSampler(
                self.frames[split]["label"].astype(str).tolist(),
                self.config.batch_size,
                self.config.samples_per_class,
                self.config.seed + index,
            )
            for index, split in enumerate(("train", "valid"))
            if balanced_train and (split == "train" or balanced_eval)
        }
        return ExperimentData(
            labels=self.labels,
            label_to_index=self.label_to_index,
            train_loader=_loader(
                datasets["train"],
                self.config.batch_size,
                self.config.num_workers,
                shuffle=not balanced_train,
                sampler=samplers.get("train"),
            ),
            valid_loader=_loader(
                datasets["valid"],
                self.config.batch_size,
                self.config.num_workers,
                sampler=samplers.get("valid"),
            ),
            support_eval_loader=_loader(
                eval_datasets["train"], self.config.batch_size, self.config.num_workers
            ),
            valid_eval_loader=_loader(
                eval_datasets["valid"], self.config.batch_size, self.config.num_workers
            ),
            train_sampler=samplers.get("train"),
        )


DATA_MODULES = {"manifest": ManifestDataModule}


def create_data_module(name: str, config: ManifestDataConfig) -> ManifestDataModule:
    try:
        return DATA_MODULES[name](config)
    except KeyError as exc:
        raise KeyError(
            f"unknown data module {name!r}; available={sorted(DATA_MODULES)}"
        ) from exc
