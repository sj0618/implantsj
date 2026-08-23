"""Registered image transforms and the shared canonical Dataset/DataLoader."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from PIL import Image

from .schema import DependencyError, TaskConfigError, sha256_json


@dataclass(frozen=True)
class TransformSpec:
    factory: Callable[..., Any]
    deterministic: bool
    parameters: frozenset[str]
    input_type: str = "PIL|tensor"
    output_type: str = "PIL|tensor"
    requires_seed: bool = False


def _torchvision():
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise DependencyError("torchvision is required for image transforms") from exc
    return transforms


def _factory(name: str) -> Callable[..., Any]:
    return lambda **kwargs: getattr(_torchvision(), name)(**kwargs)


TRANSFORMS: dict[str, TransformSpec] = {
    "resize": TransformSpec(
        _factory("Resize"),
        True,
        frozenset({"size", "interpolation", "max_size", "antialias"}),
    ),
    "center_crop": TransformSpec(_factory("CenterCrop"), True, frozenset({"size"})),
    "random_resized_crop": TransformSpec(
        _factory("RandomResizedCrop"),
        False,
        frozenset({"size", "scale", "ratio", "interpolation", "antialias"}),
        requires_seed=True,
    ),
    "random_horizontal_flip": TransformSpec(
        _factory("RandomHorizontalFlip"),
        False,
        frozenset({"p"}),
        requires_seed=True,
    ),
    "random_vertical_flip": TransformSpec(
        _factory("RandomVerticalFlip"),
        False,
        frozenset({"p"}),
        requires_seed=True,
    ),
    "random_rotation": TransformSpec(
        _factory("RandomRotation"),
        False,
        frozenset({"degrees", "interpolation", "expand", "center", "fill"}),
        requires_seed=True,
    ),
    "color_jitter": TransformSpec(
        _factory("ColorJitter"),
        False,
        frozenset({"brightness", "contrast", "saturation", "hue"}),
        requires_seed=True,
    ),
    "grayscale": TransformSpec(
        _factory("Grayscale"), True, frozenset({"num_output_channels"})
    ),
    "to_tensor": TransformSpec(
        _factory("ToTensor"), True, frozenset(), "PIL", "tensor"
    ),
    "normalize": TransformSpec(
        _factory("Normalize"),
        True,
        frozenset({"mean", "std", "inplace"}),
        "tensor",
        "tensor",
    ),
}


def build_pipeline(
    transform_manifest: dict[str, Any], pipeline_id: str, *, evaluation: bool
) -> tuple[Any, str]:
    try:
        steps = transform_manifest["pipelines"][pipeline_id]
    except KeyError as exc:
        raise TaskConfigError(f"unknown transform pipeline {pipeline_id!r}") from exc
    built = []
    for index, step in enumerate(steps):
        name = step["name"]
        if name not in TRANSFORMS:
            raise TaskConfigError(
                f"pipeline {pipeline_id} step {index}: unknown transform {name!r}"
            )
        spec = TRANSFORMS[name]
        if evaluation and not spec.deterministic:
            raise TaskConfigError(
                f"pipeline {pipeline_id} step {index}: stochastic transform "
                f"{name!r} is forbidden for evaluation"
            )
        params = step.get("params", {})
        unknown = sorted(set(params) - spec.parameters)
        if unknown:
            raise TaskConfigError(
                f"pipeline {pipeline_id} step {index}: unknown parameters {unknown}"
            )
        try:
            built.append(spec.factory(**params))
        except Exception as exc:
            raise TaskConfigError(
                f"pipeline {pipeline_id} step {index}: invalid parameters: {exc}"
            ) from exc
    return _torchvision().Compose(built), sha256_json(steps)


class CanonicalDataset:
    """One Dataset for classification, contrastive, embedding, and evaluation."""

    def __init__(
        self,
        frame,
        label_map: dict[str, int],
        transform: Callable[[Image.Image], Any],
        views: int = 1,
        view_seed: int = 0,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.label_map = label_map
        if views < 1:
            raise TaskConfigError("view count must be positive")
        self.transform = transform
        self.views = views
        self.view_seed = view_seed

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        try:
            with Image.open(row.image_uri) as image:
                image = image.convert("RGB")
                value = (
                    tuple(
                        self._independent_view(image, index, view)
                        for view in range(self.views)
                    )
                    if self.views > 1
                    else self.transform(image)
                )
        except OSError as exc:
            raise RuntimeError(
                f"sample {row.sample_id}: failed to read {row.image_uri}: {exc}"
            ) from exc
        return value, self.label_map.get(str(row.label), -1), index

    def _independent_view(self, image: Image.Image, index: int, view: int):
        import torch

        seed = derived_seed(self.view_seed, index, view)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                random.seed(seed)
                np.random.seed(seed % (2**32))
                torch.manual_seed(seed)
                return self.transform(image.copy())
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


class ClassAwareBatchSampler:
    def __init__(
        self,
        labels: list[str],
        batch_size: int,
        samples_per_class: int,
        seed: int,
    ) -> None:
        if samples_per_class < 2 or batch_size % samples_per_class:
            raise TaskConfigError(
                "class_aware sampler needs samples_per_class >= 2 and "
                "batch_size divisible by samples_per_class"
            )
        self.batch_size = batch_size
        self.k = samples_per_class
        self.p = batch_size // samples_per_class
        self.seed = seed
        self.by_label: dict[str, list[int]] = {}
        for index, label in enumerate(labels):
            self.by_label.setdefault(str(label), []).append(index)
        too_small = {
            label: len(indices)
            for label, indices in self.by_label.items()
            if len(indices) < self.k
        }
        if too_small:
            raise TaskConfigError(
                f"class_aware sampler lacks samples per class: {too_small}"
            )
        if len(self.by_label) < self.p:
            raise TaskConfigError(
                f"class_aware batch needs {self.p} classes, has {len(self.by_label)}"
            )
        self.length = max(1, math_ceil(len(labels), batch_size))

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        rng = random.Random(self.seed)
        labels = sorted(self.by_label)
        for _ in range(self.length):
            batch = [
                index
                for label in rng.sample(labels, self.p)
                for index in rng.sample(self.by_label[label], self.k)
            ]
            rng.shuffle(batch)
            yield batch


def math_ceil(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def derived_seed(base_seed: int, *parts: Any) -> int:
    digest = hashlib.sha256(
        "\0".join([str(base_seed), *(str(part) for part in parts)]).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def _seed_worker(seed: int):
    def initialize(worker_id: int) -> None:
        value = derived_seed(seed, worker_id)
        random.seed(value)
        np.random.seed(value % (2**32))
        try:
            import torch

            torch.manual_seed(value)
        except ImportError:
            pass

    return initialize


def build_loader(
    frame,
    label_map: dict[str, int],
    transform,
    config: dict[str, Any],
    *,
    base_seed: int,
    stage_id: str,
    epoch: int,
    evaluation: bool,
    views: int = 1,
):
    try:
        import torch
    except ImportError as exc:
        raise DependencyError("PyTorch is required for task execution") from exc
    if evaluation:
        if config["shuffle"] or config["drop_last"]:
            raise TaskConfigError(
                f"stage {stage_id}: evaluation loader must set shuffle=false and drop_last=false"
            )
        if not isinstance(config["sampler"], str) or config["sampler"] not in {
            "none",
            "sequential",
        }:
            raise TaskConfigError(
                f"stage {stage_id}: evaluation loader sampler must be none or sequential"
            )
    elif config["shuffle"] and config["sampler"] != "none":
        raise TaskConfigError(
            f"stage {stage_id}: training shuffle and sampler are mutually exclusive"
        )
    if config["num_workers"] == 0 and config["persistent_workers"]:
        raise TaskConfigError(
            f"stage {stage_id}: persistent_workers requires num_workers > 0"
        )
    seed = derived_seed(base_seed, stage_id, epoch)
    dataset = CanonicalDataset(frame, label_map, transform, views=views, view_seed=seed)
    if config["num_workers"] == 0:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    kwargs = {
        "num_workers": config["num_workers"],
        "pin_memory": config["pin_memory"],
        "persistent_workers": config["persistent_workers"],
        "worker_init_fn": _seed_worker(seed),
        "generator": generator,
    }
    if config["sampler"] == "class_aware":
        raise TaskConfigError(
            "class_aware sampler must be an object with type and samples_per_class"
        )
    if isinstance(config["sampler"], dict):
        sampler = config["sampler"]
        if sampler.get("type") != "class_aware":
            raise TaskConfigError(f"unknown sampler {sampler!r}")
        unknown = sorted(set(sampler) - {"type", "samples_per_class"})
        if unknown:
            raise TaskConfigError(f"unknown class_aware sampler fields: {unknown}")
        batch_sampler = ClassAwareBatchSampler(
            frame["label"].astype(str).tolist(),
            config["batch_size"],
            int(sampler["samples_per_class"]),
            seed,
        )
        return torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, **kwargs
        )
    if config["sampler"] not in {"none", "sequential"}:
        raise TaskConfigError(f"unknown sampler {config['sampler']!r}")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        drop_last=config["drop_last"],
        **kwargs,
    )
