"""In-memory runtime band preprocessing and classification data loaders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from band_vit_2.data import DataConfig, PreparedManifests, prepare_manifests

from .band_preprocessing import BandCropConfig, create_band_only_crop
from .segmentation import RuntimeSegmenter


class CachedBandDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        images: list[np.ndarray],
        label_to_index: dict[str, int],
        transform,
    ) -> None:
        if len(frame) != len(images):
            raise ValueError("manifest/cache length mismatch")
        self.frame = frame.reset_index(drop=True)
        self.images = images
        self.label_to_index = label_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.fromarray(self.images[index], mode="RGB")
        label = str(row["label"])
        return (
            self.transform(image),
            self.label_to_index[label],
            label,
            str(row["path"]),
        )


@dataclass
class RuntimeBandData:
    labels: list[str]
    label_to_index: dict[str, int]
    train_loader: object
    valid_loader: object
    test_loader: object | None
    preprocessing_report: Path


def _load_images(paths: list[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as source:
            images.append(source.convert("RGB").copy())
    return images


def precompute_band_cache(
    frames: dict[str, pd.DataFrame],
    segmenter: RuntimeSegmenter,
    band_config: BandCropConfig,
    output_dir: str | Path,
) -> tuple[dict[str, list[np.ndarray]], Path]:
    """Generate masks/crops once per run and retain only band images in RAM."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "band_preprocessing_report.csv"
    caches: dict[str, list[np.ndarray]] = {}
    report_rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    batch_size = segmenter.config.batch_size

    for split, frame in frames.items():
        split_cache: list[np.ndarray | None] = [None] * len(frame)
        print(
            f"runtime band preprocessing: split={split} samples={len(frame)}",
            flush=True,
        )
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            batch = frame.iloc[start:stop]
            paths = batch["path"].astype(str).tolist()
            images = _load_images(paths)
            masks = segmenter.predict(images)
            for offset, (path, image, mask) in enumerate(zip(paths, images, masks)):
                index = start + offset
                row = {
                    "split": split,
                    "row_index": index,
                    "path": path,
                    "status": "ok",
                    "error": "",
                }
                try:
                    band_image, stats = create_band_only_crop(
                        image, mask, band_config
                    )
                    split_cache[index] = band_image
                    row.update(stats)
                except Exception as exc:  # keep a complete failure report
                    row["status"] = "failed"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    failures.append((path, row["error"]))
                report_rows.append(row)
            if stop == len(frame) or stop % max(batch_size * 10, 20) == 0:
                print(f"  {split}: {stop}/{len(frame)}", flush=True)
        caches[split] = [value for value in split_cache if value is not None]

    report = pd.DataFrame(report_rows)
    report.to_csv(report_path, index=False)
    summary = {
        "samples": len(report),
        "successful": int(report["status"].eq("ok").sum()),
        "failed": int(report["status"].eq("failed").sum()),
        "mask_area_ratio_mean": float(
            pd.to_numeric(report.get("mask_area_ratio"), errors="coerce").mean()
        ),
        "band_area_ratio_mean": float(
            pd.to_numeric(report.get("band_area_ratio"), errors="coerce").mean()
        ),
        "mask_files_written": 0,
        "cache": "memory_only",
    }
    (output / "band_preprocessing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if failures:
        examples = "; ".join(f"{path}: {error}" for path, error in failures[:5])
        raise RuntimeError(
            f"runtime band preprocessing failed for {len(failures)} image(s); "
            f"see {report_path}; examples: {examples}"
        )
    return {key: list(value) for key, value in caches.items()}, report_path


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


def create_runtime_data(
    config: DataConfig,
    prepared: PreparedManifests,
    caches: dict[str, list[np.ndarray]],
    train_transform,
    eval_transform,
    preprocessing_report: Path,
) -> RuntimeBandData:
    datasets = {
        split: CachedBandDataset(
            frame,
            caches[split],
            prepared.label_to_index,
            train_transform if split == "train" else eval_transform,
        )
        for split, frame in prepared.frames.items()
    }
    return RuntimeBandData(
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
        preprocessing_report=preprocessing_report,
    )


def prepare_runtime_manifests(config: DataConfig) -> PreparedManifests:
    prepared = prepare_manifests(config)
    for split, frame in prepared.frames.items():
        if "mask_path" in frame.columns:
            raise ValueError(
                "band_vit_3 generates SAM2 masks at runtime and must not receive "
                f"a precomputed mask manifest ({split} contains mask_path). "
                "Use the raw path,label manifest, for example train.csv."
            )
    return prepared
