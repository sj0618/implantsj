#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coco_to_yolo.py — result.json(COCO GT)을 YOLO 형식 라벨로 변환.

- 입력 라벨 : making new public datasets/result.json  (COCO, bbox=픽셀 xywh)
- 출력 라벨 : making new public datasets/yolo_labels/<이미지stem>.txt
              각 줄: <class> <cx> <cy> <w> <h>  (모두 0~1 정규화)
- data.yaml : 클래스 정보와 경로를 함께 생성

COCO category_id -> YOLO class 매핑:
  COCO는 category id가 0부터일 수도, 1부터일 수도 있어 categories 순서대로 0..N-1 재부여.
  (여기선 'implant' 단일 클래스라 전부 0)

사용 예:
    python coco_to_yolo.py                 # 전부 변환
    python coco_to_yolo.py --no-empty      # 박스 없는 이미지의 빈 txt는 안 만듦
"""
import argparse, json, os
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", default=str(here / "result.json"), help="COCO 라벨 json")
    ap.add_argument("--images", default=str(here / "ls_images"), help="이미지 폴더(경로만 기록)")
    ap.add_argument("--out", default=str(here / "yolo_labels"), help="YOLO txt 저장 폴더")
    ap.add_argument("--no-empty", action="store_true",
                    help="박스 없는 이미지의 빈 txt를 만들지 않음")
    args = ap.parse_args()

    coco = json.load(open(args.coco, encoding="utf-8"))

    # category_id -> 0..N-1 클래스 인덱스
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    catid2cls = {c["id"]: i for i, c in enumerate(cats)}
    names = [c["name"] for c in cats]

    id2img = {im["id"]: im for im in coco["images"]}
    by_img = {im["id"]: [] for im in coco["images"]}
    for a in coco["annotations"]:
        by_img[a["image_id"]].append(a)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_txt, n_box, n_bad = 0, 0, 0
    for iid, im in id2img.items():
        fname = os.path.basename(im["file_name"])
        W, H = im["width"], im["height"]
        anns = by_img[iid]

        if not anns and args.no_empty:
            continue

        lines = []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                n_bad += 1
                continue
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H
            # 0~1 범위로 clamp (경계 넘침 방지)
            cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
            nw, nh = min(max(nw, 0), 1), min(max(nh, 0), 1)
            cls = catid2cls[a["category_id"]]
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            n_box += 1

        stem = Path(fname).stem
        with open(out_dir / f"{stem}.txt", "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")
        n_txt += 1

    # data.yaml 생성
    yaml_path = here / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {here}\n")
        f.write(f"train: {os.path.relpath(args.images, here)}\n")
        f.write(f"val: {os.path.relpath(args.images, here)}\n")
        f.write(f"nc: {len(names)}\n")
        f.write(f"names: {names}\n")

    print(f"[done] txt {n_txt}개, box {n_box}개 -> {out_dir}")
    print(f"       data.yaml -> {yaml_path}  (nc={len(names)}, names={names})")
    if n_bad:
        print(f"  (w/h<=0 인 박스 스킵: {n_bad}개)")


if __name__ == "__main__":
    main()
