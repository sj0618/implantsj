"""protocol에 맞는 manifest를 골라 split별 임시 CSV로 나눠 준다.

band_vit / masked_vit의 학습 코드는 train/valid/test manifest를 **따로** 받는다
(다른 스테이지는 split 컬럼이 있는 manifest 하나를 쓴다). 그 차이를 여기서 흡수해,
yaml에는 다른 스테이지와 똑같이 manifest 하나만 적으면 되게 한다.

protocol:
  closed27_open15  closed_manifest 사용 → classifier도 27클래스만 본다 (leakage 방지)
  full42           full_manifest 사용
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def split_by_protocol(cfg: dict, *, out_dir: str | Path | None = None) -> Dict[str, str]:
    """(train/valid/test → CSV 경로). protocol에 맞는 manifest를 split별로 저장."""
    protocol = cfg.get("protocol", "closed27_open15")
    data = cfg["data"]
    source = data["closed_manifest"] if protocol == "closed27_open15" else data["full_manifest"]
    source_path = Path(source).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"manifest가 없습니다: {source_path}\n"
            f"  먼저 데이터 준비를 돌리세요: bash scripts/00_data/prepare_data.sh"
        )

    frame = pd.read_csv(source_path)
    for column in ("path", "label", "split"):
        if column not in frame.columns:
            raise ValueError(f"{source_path}에 {column} 컬럼이 없습니다")

    out = Path(out_dir) if out_dir else Path(cfg["paths"]["output_dir"]) / "_manifest_splits"
    out.mkdir(parents=True, exist_ok=True)

    result: Dict[str, str] = {}
    counts = []
    for split in ("train", "valid", "test"):
        part = frame[frame["split"].astype(str) == split]
        if part.empty:
            raise ValueError(f"{source_path}에 {split} split이 없습니다")
        target = out / f"{protocol}_{split}.csv"
        part.to_csv(target, index=False)
        result[split] = str(target)
        counts.append(f"{split} {len(part)}장")

    print(f"manifest({protocol}): {source_path.name} → {', '.join(counts)} "
          f"/ {frame['label'].nunique()}클래스", flush=True)
    return result
