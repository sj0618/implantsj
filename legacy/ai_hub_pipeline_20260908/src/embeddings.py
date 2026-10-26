"""학습 종료 시 임베딩 저장 hook — 모든 스테이지 공통.

모든 학습(.pt 1개)이 끝날 때마다, 파이프라인이 아는 **전체 이미지**에 대해
백본 임베딩과 head 출력을 npz로 남긴다.

  <run_dir>/embeddings/<manifest_stem>_<split>_backbone.npz   백본 feature (pooled, [N, D])
  <run_dir>/embeddings/<manifest_stem>_<split>_head.npz       head 출력 (logits / projection, [N, K])
  <run_dir>/embeddings/index.json                             파일 목록 + 체크포인트 SHA-256

"전체 이미지"의 출처는 yaml의 data 블록이다:
  data.full_manifest 가 있으면 그것(42클래스 전부, 모든 split),
  없으면 data.closed_manifest + data.open_manifest 를 각각.
closed27_open15 프로토콜에서 open 15 이미지도 임베딩만 뽑는 것은 gradient에 닿지
않으므로 leakage가 아니다. partition 컬럼(closed/open)이 그 구분을 파일에 남긴다.

왜 필요한가: Stage 2의 prototype × distance 재평가, Stage 5 fuse, 사후 분석(t-SNE,
Mahalanobis, few-shot)은 전부 임베딩만 있으면 되므로 학습을 다시 돌리지 않는다.

npz 키:
  features            [N, D] float32 (head가 여러 개면 features__<head> 로 나뉜다)
  paths, labels       [N] str
  split, partition    [N] str — partition ∈ {closed, open, unknown}
  manifest, manifest_sha256, checkpoint, checkpoint_sha256, model, stage, source,
  head_kind(head 파일만: logits | projection), created, extra(JSON 문자열)

설정(base.yaml):
  embeddings:
    enabled: true
    batch_size: 64
    num_workers: null      # null이면 train.num_workers
    input_mode: original   # masked-vit 전용: original(원본 픽셀) | band(SAM2 band crop)

각 스테이지는 자기 eval transform 과 forward(images → (backbone, head)) 만 넘긴다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_MODES = ("original", "band")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# =====================================================================
# 설정 → 계획
# =====================================================================

@dataclass
class EmbedPlan:
    enabled: bool = True
    manifests: List[str] = field(default_factory=list)
    closed_labels: Optional[set] = None       # None이면 partition=unknown
    batch_size: int = 64
    num_workers: int = 4
    input_mode: str = "original"


def plan_from_config(cfg: dict) -> EmbedPlan:
    """yaml(base.yaml embeddings + 스테이지 data 블록)에서 계획을 만든다."""
    e = dict(cfg.get("embeddings") or {})
    if not bool(e.get("enabled", True)):
        return EmbedPlan(enabled=False)      # 꺼져 있으면 manifest 를 열지도 않는다
    d = dict(cfg.get("data") or {})
    if d.get("full_manifest"):
        manifests = [str(d["full_manifest"])]
    else:
        manifests = [str(d[k]) for k in ("closed_manifest", "open_manifest") if d.get(k)]

    closed = None
    closed_path = d.get("closed_manifest")
    if closed_path and Path(closed_path).expanduser().is_file():
        frame = pd.read_csv(closed_path, usecols=["label"])
        closed = set(frame["label"].astype(str).str.strip())

    workers = e.get("num_workers")
    if workers is None:
        workers = (cfg.get("train") or {}).get("num_workers", 4)
    input_mode = str(e.get("input_mode", "original"))
    if input_mode not in INPUT_MODES:
        raise ValueError(f"embeddings.input_mode must be one of {INPUT_MODES}: {input_mode!r}")
    return EmbedPlan(
        enabled=bool(e.get("enabled", True)),
        manifests=manifests,
        closed_labels=closed,
        batch_size=int(e.get("batch_size", 64)),
        num_workers=int(workers),
        input_mode=input_mode,
    )


# =====================================================================
# manifest / dataset / transform
# =====================================================================

def read_manifest(path: str | Path) -> pd.DataFrame:
    """path,label(,split). 상대경로는 manifest 폴더 기준, 없으면 cwd 기준으로 푼다."""
    manifest_path = Path(path).expanduser().resolve()
    frame = pd.read_csv(manifest_path)
    if not {"path", "label"}.issubset(frame.columns):
        raise ValueError(f"manifest {manifest_path} must contain path,label")
    frame = frame.copy()

    def resolve(value) -> str:
        raw = Path(str(value)).expanduser()
        if raw.is_absolute():
            return str(raw)
        by_manifest = manifest_path.parent / raw
        if by_manifest.is_file():
            return str(by_manifest.resolve())
        if raw.is_file():
            return str(raw.resolve())
        return str(by_manifest)

    frame["path"] = frame["path"].map(resolve)
    frame["label"] = frame["label"].astype(str).str.strip()
    if "split" not in frame.columns:
        frame["split"] = "all"
    return frame.reset_index(drop=True)


def pad_to_square(image: Image.Image, fill=0) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    canvas = Image.new(image.mode, (side, side), fill)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def square_pad_transform(image_size: int, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Stage 1 백본의 eval 전처리와 동일: 정사각 패딩 → resize → normalize."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as vision

    def transform(image: Image.Image):
        image = pad_to_square(image.convert("RGB"), fill=0)
        image = vision.resize(image, [image_size, image_size], interpolation=InterpolationMode.BILINEAR)
        return vision.normalize(vision.to_tensor(image), list(mean), list(std))

    return transform


class PathDataset:
    """이미지 경로 → (tensor, index). 라벨은 dump 쪽이 manifest에서 붙인다."""

    def __init__(self, paths: Sequence[str], transform: Callable) -> None:
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as source:
            tensor = self.transform(source.convert("RGB"))
        return tensor, index


class ArrayDataset:
    """메모리에 있는 RGB 배열(masked-vit band crop 캐시) → (tensor, index)."""

    def __init__(self, images: Sequence[np.ndarray], transform: Callable) -> None:
        self.images = list(images)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        return self.transform(Image.fromarray(self.images[index], mode="RGB")), index


# =====================================================================
# forward 도우미
# =====================================================================

def timm_classifier_forward(model) -> Callable:
    """timm 분류기(ViT/CNN) → (pooled feature, logits). Stage 1·3이 쓴다."""
    model.eval()

    def forward(images):
        feats = model.forward_features(images)
        pooled = model.forward_head(feats, pre_logits=True)
        logits = model.forward_head(feats)
        return pooled, logits

    return forward


# =====================================================================
# 저장
# =====================================================================

def _partition(labels: np.ndarray, closed: Optional[set]) -> np.ndarray:
    if closed is None:
        return np.asarray(["unknown"] * len(labels), dtype=str)
    return np.where(np.isin(labels, sorted(closed)), "closed", "open").astype(str)


def dump_embeddings(
    *,
    output_dir: str | Path,
    plan: EmbedPlan,
    forward: Callable,
    transform: Optional[Callable] = None,
    stage: str,
    model_name: str,
    checkpoint: str | Path | None = None,
    device="cpu",
    head_kind: str = "logits",
    extra: Optional[dict] = None,
    dataset_factory: Optional[Callable[[pd.DataFrame], Tuple[object, pd.DataFrame]]] = None,
) -> List[Path]:
    """plan.manifests 의 모든 split 에 대해 backbone/head npz 를 쓴다.

    forward(images) → (backbone [B,D], head [B,K] 또는 {이름: [B,K]})
    transform       PIL → tensor (eval 전처리). dataset_factory 를 주면 대신 쓴다:
                    dataset_factory(frame) → (dataset, 실제로 쓰인 frame)
    """
    if not plan.enabled:
        log("embeddings: 꺼져 있음 (embeddings.enabled=false)")
        return []
    if transform is None and dataset_factory is None:
        raise ValueError("transform 또는 dataset_factory 중 하나는 필요합니다")

    import torch
    from torch.utils.data import DataLoader

    torch_device = device if isinstance(device, torch.device) else torch.device(device)
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    checkpoint_str = str(checkpoint) if checkpoint else ""
    checkpoint_sha = sha256_of(checkpoint_str) if checkpoint_str and Path(checkpoint_str).is_file() else ""
    created = datetime.now().isoformat(timespec="seconds")

    # 같은 checkpoint 로 이미 다 뽑아 뒀으면(skip_existing 재개 등) 전체 추론을 반복하지 않는다.
    index_path = out / "index.json"
    if checkpoint_sha and index_path.is_file():
        try:
            previous = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
        files = [out / row[key] for row in previous.get("files", []) for key in ("backbone_file", "head_file")]
        if previous.get("checkpoint_sha256") == checkpoint_sha and files and all(f.is_file() for f in files):
            log(f"embeddings: 같은 checkpoint(sha256 {checkpoint_sha[:12]}…)의 npz {len(files)}개가 이미 있음 → 건너뜀")
            return files
    extra_json = json.dumps(extra or {}, ensure_ascii=False, default=str)
    use_amp = torch_device.type == "cuda"

    written: List[Path] = []
    index_rows: List[dict] = []
    for manifest in plan.manifests:
        manifest_path = Path(manifest).expanduser()
        if not manifest_path.is_file():
            log(f"embeddings: manifest 없음 → 건너뜀: {manifest_path}")
            continue
        frame = read_manifest(manifest_path)
        manifest_sha = sha256_of(manifest_path)
        stem = manifest_path.stem

        for split in sorted(frame["split"].astype(str).unique()):
            part = frame[frame["split"].astype(str) == split].reset_index(drop=True)
            if dataset_factory is not None:
                dataset, part = dataset_factory(part)
            else:
                dataset = PathDataset(part["path"].tolist(), transform)
            if len(dataset) == 0:
                continue
            if len(dataset) != len(part):
                raise RuntimeError(f"embeddings: dataset({len(dataset)}) != manifest rows({len(part)})")

            loader = DataLoader(dataset, batch_size=plan.batch_size, shuffle=False,
                                num_workers=plan.num_workers,
                                pin_memory=torch_device.type == "cuda")
            backbone_chunks: List[np.ndarray] = []
            head_chunks: Dict[str, List[np.ndarray]] = {}
            seen: List[int] = []
            with torch.inference_mode():
                for images, index in loader:
                    images = images.to(torch_device, non_blocking=True)
                    with torch.autocast("cuda", enabled=use_amp):
                        backbone_out, head_out = forward(images)
                    backbone_chunks.append(backbone_out.float().cpu().numpy())
                    heads = head_out if isinstance(head_out, dict) else {"": head_out}
                    for name, tensor in heads.items():
                        head_chunks.setdefault(str(name), []).append(tensor.float().cpu().numpy())
                    seen.extend(int(i) for i in index)
            if seen != list(range(len(part))):
                raise RuntimeError("embeddings: loader 순서가 manifest 순서와 다릅니다 (shuffle=False 필요)")

            paths = np.asarray(part["path"].astype(str).tolist(), dtype=str)
            labels = np.asarray(part["label"].astype(str).tolist(), dtype=str)
            common = dict(
                paths=paths, labels=labels,
                split=np.asarray([split] * len(paths), dtype=str),   # np.full(dtype=str) 은 <U1 로 잘린다
                partition=_partition(labels, plan.closed_labels),
                manifest=str(manifest_path.resolve()), manifest_sha256=manifest_sha,
                checkpoint=checkpoint_str, checkpoint_sha256=checkpoint_sha,
                model=str(model_name), stage=str(stage), created=created, extra=extra_json,
            )
            backbone_features = np.concatenate(backbone_chunks, axis=0).astype(np.float32)
            backbone_path = out / f"{stem}_{split}_backbone.npz"
            np.savez(backbone_path, features=backbone_features, source="backbone", **common)

            head_path = out / f"{stem}_{split}_head.npz"
            head_arrays = {name: np.concatenate(chunks, axis=0).astype(np.float32)
                           for name, chunks in head_chunks.items()}
            if list(head_arrays) == [""]:
                np.savez(head_path, features=head_arrays[""], source="head",
                         head_kind=head_kind, **common)
                head_dims = {"": int(head_arrays[""].shape[1])}
            else:
                np.savez(head_path, source="head", head_kind=head_kind,
                         heads=np.asarray(list(head_arrays), dtype=str),
                         **{f"features__{name}": arr for name, arr in head_arrays.items()},
                         **common)
                head_dims = {name: int(arr.shape[1]) for name, arr in head_arrays.items()}

            written += [backbone_path, head_path]
            index_rows.append({
                "manifest": stem, "split": split, "n": int(len(paths)),
                "backbone_file": backbone_path.name, "backbone_dim": int(backbone_features.shape[1]),
                "head_file": head_path.name, "head_dims": head_dims,
                "closed": int((common["partition"] == "closed").sum()),
                "open": int((common["partition"] == "open").sum()),
            })
            log(f"embeddings: {stem}/{split} {len(paths)}장 → "
                f"backbone[{backbone_features.shape[1]}] + head{list(head_dims.values())} 저장")

    index = {
        "stage": stage, "model": str(model_name), "checkpoint": checkpoint_str,
        "checkpoint_sha256": checkpoint_sha, "head_kind": head_kind, "created": created,
        "input_mode": plan.input_mode, "extra": extra or {}, "files": index_rows,
    }
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2, default=str) + "\n",
                                     encoding="utf-8")
    if not written:
        log("embeddings: 저장된 파일 없음 (manifest 경로를 확인하세요)")
    else:
        log(f"embeddings: {len(written)}개 파일 → {out}")
    return written


def try_dump_embeddings(**kwargs) -> List[Path]:
    """dump_embeddings 를 감싸 실패해도 학습 산출물을 잃지 않게 한다.

    임베딩은 분석용 부산물이다. 학습이 끝난 뒤 이것 때문에 평가·시각화까지 죽으면 안 되므로
    예외를 삼키고 원인을 로그로 남긴다. 스테이지 드라이버는 이 함수를 쓴다.
    """
    try:
        return dump_embeddings(**kwargs)
    except Exception:
        import traceback

        log("embeddings: 저장 실패 — 학습 산출물은 그대로이고 임베딩만 빠졌다. 원인:")
        traceback.print_exc()
        return []
