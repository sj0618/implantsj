"""
⑦ 예측 마스크에서 임플란트 경계의 band(고리) 영역 추출
   band = dilate(mask, outer) - erode(mask, inner)
   → <stem>_band.png 로 저장. band-* 백본과 band-vit의 입력이 된다.

manifest 연결은 Stage 0-a(run_prepare_data.py)가 맡는다 — 여기서는 그림만 만든다.
   (원본: implant_ny/make_seg_model/make_band_masks.py)

실행: python make_band_masks.py --config configs/02_segmentation.yaml
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_utils import load_config_from_cli


def main():
    cfg = load_config_from_cli(__doc__)
    bcfg = cfg["band"]

    size = int(bcfg["size"])
    outer = int(bcfg["outer"])
    inner = int(bcfg["inner"])
    source_suffix = bcfg.get("source_suffix", "_pred.png")

    mask_root = Path(bcfg["mask_root"])
    out = Path(bcfg["out_root"])
    out.mkdir(parents=True, exist_ok=True)

    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * outer + 1, 2 * outer + 1))
    ki = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inner + 1, 2 * inner + 1))

    sources = sorted(mask_root.rglob(f"*{source_suffix}"))
    if not sources:
        raise FileNotFoundError(
            f"{mask_root}에서 {source_suffix} 파일을 찾지 못했습니다. "
            f"05_predict_masks.sh를 먼저 실행하세요."
        )
    print(f"예측 마스크 {len(sources)}장 → band 추출 (outer={outer}, inner={inner}, size={size})")

    written, empty = 0, 0
    for index, source in enumerate(sources, 1):
        stem = source.name[: -len(source_suffix)]
        m = cv2.imread(str(source), 0)
        if m is None:
            continue
        m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
        m = (m > 127).astype(np.uint8) * 255

        # band = 바깥쪽 팽창 - 안쪽 침식 (고리 영역)
        band = cv2.subtract(cv2.dilate(m, ko), cv2.erode(m, ki))
        if not band.any():
            empty += 1          # 마스크가 비었거나 너무 얇아 고리가 안 남은 경우
        cv2.imwrite(str(out / f"{stem}_band.png"), band)
        written += 1
        if index % 1000 == 0:
            print(f"  {index}/{len(sources)}")

    print(f"완료: {written}장 저장 → {out}")
    if empty:
        print(f"  경고: band가 비어 있는 마스크 {empty}장 "
              f"(세그멘테이션이 실패했거나 outer/inner가 너무 큽니다)")
    print("다음: bash scripts/00_data/prepare_data.sh 를 (다시) 실행하면 "
          "manifest의 mask_path가 채워집니다")


if __name__ == "__main__":
    main()
