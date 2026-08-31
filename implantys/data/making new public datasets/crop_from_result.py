#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crop_from_result.py — result.json(COCO GT)의 bbox 기준으로 임플란트 영역을 잘라 저장.

- 입력 라벨 : making new public datasets/result.json  (COCO, bbox=픽셀 xywh)
- 입력 이미지: making new public datasets/ls_images/<file_name basename>
- 출력      : making new public datasets/crops/  (기본)

사용 예:
    python crop_from_result.py                 # 기본값으로 전부 crop
    python crop_from_result.py --pad 0.05      # bbox를 5% 여유 두고 crop
    python crop_from_result.py --by-manufacturer   # 제조사별 하위폴더로 정리
    python crop_from_result.py --min-size 16   # 너무 작은 박스는 스킵

출력 파일명: <이미지stem>__<annId>.jpg  (한 이미지에 임플란트 여러 개면 annId로 구분)
"""
import argparse, json, os
from pathlib import Path
from PIL import Image

# 파일명 접두어 -> 제조사 폴더명 (--by-manufacturer 옵션용)
MFR_PREFIXES = [
    ("i-mplance", "IMPLANCE"), ("implance", "IMPLANCE"),
    ("nucleoss", "NUCLEOSS"), ("bilim", "BILIM"),
    ("adin", "ADIN"), ("noris", "NORIS"), ("osstem", "osstem"),
    ("dentium", "Dentium"), ("dionavi", "DIONAVI"), ("nobel", "nobel"),
    ("mis", "MIS"),
]


def guess_manufacturer(fname):
    low = fname.lower()
    for pref, name in MFR_PREFIXES:
        if low.startswith(pref):
            return name
    return "UNKNOWN"


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", default=str(here / "result.json"), help="COCO 라벨 json")
    ap.add_argument("--images", default=str(here / "ls_images"), help="이미지 폴더")
    ap.add_argument("--out", default=str(here / "crops"), help="crop 저장 폴더")
    ap.add_argument("--pad", type=float, default=0.0,
                    help="bbox 각 변을 (pad*변길이)만큼 확장. 0.05=5%%")
    ap.add_argument("--min-size", type=int, default=0,
                    help="crop 후 가로/세로가 이 값보다 작으면 스킵(px)")
    ap.add_argument("--by-manufacturer", action="store_true",
                    help="제조사별 하위폴더로 저장")
    ap.add_argument("--quality", type=int, default=95, help="JPEG 품질")
    args = ap.parse_args()

    coco = json.load(open(args.coco, encoding="utf-8"))
    id2img = {im["id"]: im for im in coco["images"]}
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    n_ok, n_skip_small, n_missing = 0, 0, 0
    cache = {}  # path -> PIL Image (같은 이미지 여러 박스일 때 재오픈 방지)

    for iid, anns in by_img.items():
        im_meta = id2img[iid]
        fname = os.path.basename(im_meta["file_name"])
        src = Path(args.images) / fname
        if not src.exists():
            print(f"!! 이미지 없음: {src}")
            n_missing += len(anns)
            continue

        if src not in cache:
            cache[src] = Image.open(src).convert("RGB")
        img = cache[src]
        W, H = img.size

        stem = Path(fname).stem
        if args.by_manufacturer:
            out_dir = out_root / guess_manufacturer(fname)
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = out_root

        for a in anns:
            x, y, w, h = a["bbox"]
            # 패딩 적용
            px, py = w * args.pad, h * args.pad
            x0 = int(round(max(0, x - px)))
            y0 = int(round(max(0, y - py)))
            x1 = int(round(min(W, x + w + px)))
            y1 = int(round(min(H, y + h + py)))
            if x1 <= x0 or y1 <= y0:
                continue
            if (x1 - x0) < args.min_size or (y1 - y0) < args.min_size:
                n_skip_small += 1
                continue
            crop = img.crop((x0, y0, x1, y1))
            out_path = out_dir / f"{stem}__{a['id']}.jpg"
            crop.save(out_path, quality=args.quality)
            n_ok += 1

    print(f"[done] crop 저장 {n_ok}개 -> {out_root}")
    if n_skip_small:
        print(f"  (너무 작아 스킵: {n_skip_small}개)")
    if n_missing:
        print(f"  (이미지 못 찾아 스킵: {n_missing}개)")


if __name__ == "__main__":
    main()
