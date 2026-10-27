"""In-memory runtime band preprocessing and classification data loaders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .core import DataConfig, PreparedManifests, prepare_manifests

from .band_preprocessing import BandCropConfig, INPUT_MODES, create_input_crop
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
    input_mode: str = "band",
    max_failure_rate: float = 0.0,
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]], Path]:
    """Generate masks/crops once per run; cache inputs and aligned band masks.

    input_mode selects what the classifier will see (band / original);
    the band mask cache always marks the boundary band in the same geometry
    so attention-on-band statistics can be computed later.
    """
    if input_mode not in INPUT_MODES:
        raise ValueError(f"input_mode must be one of {INPUT_MODES}")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "band_preprocessing_report.csv"
    caches: dict[str, list[np.ndarray]] = {}
    band_mask_caches: dict[str, list[np.ndarray]] = {}
    report_rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    failed_rows: dict[str, list[int]] = {}
    batch_size = segmenter.config.batch_size

    for split, frame in frames.items():
        split_cache: list[np.ndarray | None] = [None] * len(frame)
        split_band_cache: list[np.ndarray | None] = [None] * len(frame)
        print(
            f"runtime band preprocessing: split={split} mode={input_mode} "
            f"samples={len(frame)}",
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
                    "input_mode": input_mode,
                    "status": "ok",
                    "error": "",
                }
                try:
                    input_image, band_mask, stats = create_input_crop(
                        image, mask, band_config, input_mode=input_mode
                    )
                    split_cache[index] = input_image
                    split_band_cache[index] = band_mask
                    row.update(stats)
                except Exception as exc:  # keep a complete failure report
                    row["status"] = "failed"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    failures.append((path, row["error"]))
                    failed_rows.setdefault(split, []).append(index)
                report_rows.append(row)
            if stop == len(frame) or stop % max(batch_size * 10, 20) == 0:
                print(f"  {split}: {stop}/{len(frame)}", flush=True)
        caches[split] = [value for value in split_cache if value is not None]
        band_mask_caches[split] = [
            value for value in split_band_cache if value is not None
        ]

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
        "input_mode": input_mode,
    }
    (output / "band_preprocessing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if failures:
        total = sum(len(frame) for frame in frames.values())
        rate = len(failures) / max(total, 1)
        examples = "; ".join(f"{path}: {error}" for path, error in failures[:5])
        if rate > max_failure_rate:
            raise RuntimeError(
                f"runtime band preprocessing failed for {len(failures)}/{total} image(s) "
                f"({rate:.3%} > 허용치 {max_failure_rate:.3%}); "
                f"see {report_path}; examples: {examples}\n"
                f"소수의 실패를 넘기려면 04_masked_vit.yaml의 band.max_failure_rate 를 올리세요."
            )
        # 허용치 안이면 그 이미지를 빼고 계속한다. 캐시에서 이미 빠졌으므로
        # manifest에서도 같은 행을 지워야 길이가 맞는다.
        for split, rows in failed_rows.items():
            frames[split] = frames[split].drop(frames[split].index[rows]).reset_index(drop=True)
        print(f"WARNING band 전처리 실패 {len(failures)}/{total}장 ({rate:.3%}) 제외하고 진행 — "
              f"목록: {report_path}", flush=True)
    return (
        {key: list(value) for key, value in caches.items()},
        {key: list(value) for key, value in band_mask_caches.items()},
        report_path,
    )


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
    """masked-vit용 manifest. band는 SAM2로 런타임 생성하므로 mask_path는 쓰지 않는다.

    Stage 0-a가 만드는 표준 manifest에는 mask_path가 항상 들어 있다(band-vit용).
    예전에는 그 컬럼이 있으면 실행을 막았지만, 그러면 파이프라인 자기 산출물을
    받을 수 없다. 이제는 컬럼을 떨어뜨리고 로그로 알린다 — 미리 만든 마스크를
    쓰지 않는다는 사실만 분명히 하면 된다.
    """
    prepared = prepare_manifests(config)
    dropped = [split for split, frame in prepared.frames.items()
               if "mask_path" in frame.columns]
    for split in dropped:
        prepared.frames[split] = prepared.frames[split].drop(columns=["mask_path"])
    if dropped:
        print(f"[masked-vit] mask_path 컬럼 무시 ({', '.join(dropped)}) — "
              f"band는 SAM2로 런타임 생성합니다", flush=True)
    return prepared
