#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
repad_image4.py — Image4(tight crop)를 원본 Image3_최종에서 padding 넣어 다시 crop.

- crop이 원본의 정확한 부분영역이라 template matching(TM_CCOEFF_NORMED)으로
  원본 내 bbox를 복원한 뒤, 각 변을 (pad * 변길이)만큼 확장해 다시 자른다.
- 파일명은 그대로 유지, 버전별 폴더에 저장.
"""
import cv2, numpy as np, re, os, glob
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "3label.0721.fix" / "Image3_최종"   # 원본 640x640
CROPS = HERE / "3label.0721.fix" / "Image4"        # tight crop
PADS = {"pad10": 0.10, "pad25": 0.25, "pad50": 0.50}
MIN_SCORE = 0.9  # 매칭 신뢰 임계

def orig_stem(name):
    # Bicon-104_1 -> Bicon-104 ,  Bicon-100 -> Bicon-100
    m = re.match(r'(.+?)(_\d+)?$', name)
    return m.group(1)

def main():
    out_dirs = {}
    for v in PADS:
        d = HERE / "3label.0721.fix" / f"Image4_{v}"
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[v] = d

    orig_cache = {}
    n_ok, n_low, n_missing = 0, 0, 0
    for cp in sorted(glob.glob(str(CROPS / "*.jpg"))):
        name = Path(cp).stem
        crop = cv2.imread(cp)
        if crop is None:
            continue
        cg = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        op = BASE / f"{orig_stem(name)}.jpg"
        if not op.exists():
            print(f"!! 원본 없음: {op.name} (crop={name})")
            n_missing += 1
            continue
        key = str(op)
        if key not in orig_cache:
            orig_cache[key] = cv2.imread(str(op))
        orig = orig_cache[key]
        og = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        H, W = og.shape
        res = cv2.matchTemplate(og, cg, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        h, w = cg.shape
        x0, y0, x1, y1 = loc[0], loc[1], loc[0] + w, loc[1] + h
        if mx < MIN_SCORE:
            print(f"!! 매칭 낮음 score={mx:.3f} {name}")
            n_low += 1
        for v, pad in PADS.items():
            px, py = int(round(w * pad)), int(round(h * pad))
            a, b = max(0, x0 - px), max(0, y0 - py)
            c, d = min(W, x1 + px), min(H, y1 + py)
            cv2.imwrite(str(out_dirs[v] / f"{name}.jpg"),
                        orig[b:d, a:c], [cv2.IMWRITE_JPEG_QUALITY, 95])
        n_ok += 1

    print(f"[done] {n_ok}개 crop -> {list(out_dirs.values())}")
    if n_low:     print(f"  (매칭 낮은 것: {n_low}개 — 확인 권장)")
    if n_missing: print(f"  (원본 못 찾음: {n_missing}개)")

if __name__ == "__main__":
    main()
