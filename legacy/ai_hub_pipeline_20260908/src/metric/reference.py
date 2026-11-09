"""대표 이미지 폴더 → 클래스별 임베딩 (prototype method: reference).

폴더 구조는 <root>/<클래스>/*.jpg — 데이터셋과 같은 방식이다.
제조사 카탈로그처럼 그 시스템을 가장 잘 보여주는 사진을 prototype으로 쓴다.

데이터셋 라벨과 폴더명 표기가 조금 달라도 맞춰주고(대소문자·공백·구분자·앞 일련번호),
대표 이미지가 없는 클래스는 build_prototypes가 갤러리 평균으로 대체한다.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .core import build_transforms, log

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _normalize(name: str) -> str:
    s = re.sub(r"^\d+[_\-\s]+", "", str(name).strip().lower())
    return re.sub(r"[^a-z0-9]+", "", s)


def scan_reference_dir(root: str | Path, classes: Sequence[str],
                       *, extensions: Sequence[str] | None = None) -> Dict[str, List[Path]]:
    """클래스 → 대표 이미지 경로 목록. 데이터셋 라벨 이름에 맞춰 돌려준다."""
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"대표 이미지 폴더가 없습니다: {root}")
    allowed = {e.lower() if e.startswith(".") else f".{e.lower()}"
               for e in (extensions or IMAGE_EXTENSIONS)}

    # 폴더명 → 이미지들
    found: Dict[str, List[Path]] = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(f for f in folder.iterdir()
                        if f.is_file() and f.suffix.lower() in allowed)
        if images:
            found[folder.name] = images

    # 데이터셋 라벨에 붙이기 (정확히 일치 → 표기 무시 일치)
    loose: Dict[str, List[str]] = {}
    for name in found:
        loose.setdefault(_normalize(name), []).append(name)

    mapped: Dict[str, List[Path]] = {}
    corrected, missing = [], []
    for label in classes:
        if label in found:
            mapped[label] = found[label]
            continue
        candidates = loose.get(_normalize(label), [])
        if len(candidates) == 1:
            mapped[label] = found[candidates[0]]
            corrected.append(f"{label} ← {candidates[0]}")
        elif len(candidates) > 1:
            raise ValueError(f"대표 이미지 폴더가 여러 개 겹칩니다: {label} ↔ {candidates}")
        else:
            missing.append(label)

    total = sum(len(v) for v in mapped.values())
    log(f"  대표 이미지: {len(mapped)}/{len(classes)}클래스, {total}장  ({root})")
    if corrected:
        log(f"    폴더명 보정 {len(corrected)}건: {corrected[:3]}"
            f"{' ...' if len(corrected) > 3 else ''}")
    if missing:
        log(f"    ⚠ 대표 이미지가 없는 클래스 {len(missing)}개: {missing}")
        log(f"      → 이 클래스는 갤러리 평균 prototype으로 대체됩니다")
    return mapped


def embed_reference(model, reference: Dict[str, List[Path]], *, image_size: int,
                    batch_size: int, num_workers: int, device,
                    source: str = "projection") -> Dict[str, np.ndarray]:
    """대표 이미지를 query와 같은 전처리·모델로 임베딩한다 (같은 공간이어야 비교된다)."""
    import torch
    from PIL import Image

    transform = build_transforms(image_size, train=False)

    class _Dataset:
        def __init__(self, paths):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, index):
            with Image.open(self.paths[index]) as source_image:
                return transform(source_image.convert("RGB")), index

    labels = [l for l, v in reference.items() if v]
    flat = [(label, path) for label in labels for path in reference[label]]
    if not flat:
        return {}

    loader = torch.utils.data.DataLoader(
        _Dataset([p for _, p in flat]), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    model.eval()
    chunks = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            z = model.forward_features(images) if source == "backbone" else model(images)
            chunks.append(z.float().cpu().numpy())
    features = np.concatenate(chunks, axis=0)

    out: Dict[str, np.ndarray] = {}
    position = 0
    for label in labels:
        count = len(reference[label])
        out[label] = features[position:position + count]
        position += count
    return out


def audit_reference_leakage(reference: Dict[str, List[Path]], manifests: Dict[str, Sequence[str]],
                            *, output_dir: str | Path | None = None,
                            strict: bool = True) -> dict:
    """대표 이미지가 학습·평가 이미지와 바이트 단위로 같은지 대조한다.

    GT 이미지가 train에도 들어가 있으면 prototype이 정답을 미리 본 셈이 되므로,
    같은 파일이 발견되면 기본적으로 실행을 중단한다.
    출처: implantsj pipelines/eval/evaluate_metric_diagnosis.py 의 content_audit
    """
    import hashlib

    import pandas as pd

    def digest(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    records = []
    for split, paths in manifests.items():
        for path in paths:
            file = Path(path)
            if file.is_file():
                records.append({"sha256": digest(file), "source": split, "path": str(file)})
    for label, paths in reference.items():
        for file in paths:
            records.append({"sha256": digest(Path(file)), "source": "reference",
                            "path": str(file), "label": label})

    by_hash: Dict[str, list] = {}
    for record in records:
        by_hash.setdefault(record["sha256"], []).append(record)

    collisions = []
    for checksum, items in by_hash.items():
        sources = {i["source"] for i in items}
        if "reference" in sources and len(sources) > 1:
            other = sorted(sources - {"reference"})
            collisions.append({
                "sha256": checksum, "shared_with": ",".join(other),
                "paths": " | ".join(i["path"] for i in items),
            })

    result = {"files": len(records), "unique_contents": len(by_hash),
              "collisions": len(collisions), "ok": not collisions}
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(out / "reference_audit.csv", index=False)
        if collisions:
            pd.DataFrame(collisions).to_csv(out / "reference_collisions.csv", index=False)

    if collisions:
        examples = "; ".join(c["paths"][:160] for c in collisions[:3])
        message = (
            f"대표 이미지 {len(collisions)}장이 학습/평가 데이터와 내용이 같습니다 "
            f"(prototype이 정답을 미리 본 셈이 됩니다).\n  예: {examples}"
        )
        if strict:
            raise RuntimeError(message)
        log(f"  ⚠ {message}")
    else:
        log(f"  누출 검사 통과: 대표 이미지와 데이터셋에 동일한 파일 없음 "
            f"({result['files']}개 파일 대조)")
    return result


def load_reference_prototypes(model, root: str | Path, classes: Sequence[str], *,
                              image_size: int, batch_size: int, num_workers: int,
                              device, source: str = "projection",
                              extensions: Sequence[str] | None = None) -> Dict[str, np.ndarray]:
    """폴더 스캔 + 임베딩을 한 번에."""
    mapped = scan_reference_dir(root, classes, extensions=extensions)
    return embed_reference(model, mapped, image_size=image_size, batch_size=batch_size,
                           num_workers=num_workers, device=device, source=source)
