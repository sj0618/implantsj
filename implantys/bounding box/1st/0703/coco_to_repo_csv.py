#!/usr/bin/env python3
"""
dataset_coco (표준 COCO) -> Asad-Ismail/Grounding-Dino-FineTuning repo 형식으로 변환.

이 repo는 annotation을 CSV로 읽는다. 컬럼:
  image_name, bbox_x, bbox_y, bbox_width, bbox_height, label_name
box 형식은 COCO와 동일한 [x, y, w, h] (좌상단+너비/높이, 픽셀). 좌표 변환 불필요.

결과 구조 (repo의 train_config.yaml이 기대하는 형태):
  multimodal-data/implant_dataset/
    images/train/*.jpg   +  train_annotations.csv
    images/val/*.jpg     +  val_annotations.csv
    images/test/*.jpg    +  test_annotations.csv

사용법 (repo 루트에서):
  python coco_to_repo_csv.py \
      --coco-root /workspace/dataset_coco \
      --out-root  multimodal-data/implant_dataset
"""
import argparse, csv, json, shutil
from pathlib import Path


def convert_split(split, coco_root, out_root):
    src_dir = Path(coco_root) / split
    ann_json = src_dir / "annotations.json"
    img_src = src_dir / "images"

    if not ann_json.exists():
        print(f"  [skip] {split}: annotations.json 없음")
        return

    coco = json.loads(ann_json.read_text(encoding="utf-8"))
    id2name = {im["id"]: im["file_name"] for im in coco["images"]}

    # 이미지 복사: images/<split>/
    img_out = Path(out_root) / "images" / split
    img_out.mkdir(parents=True, exist_ok=True)
    for im in coco["images"]:
        src = img_src / im["file_name"]
        if src.exists():
            shutil.copy2(src, img_out / im["file_name"])
        else:
            print(f"    [warn] 이미지 없음: {src}")

    # CSV 작성: <split>_annotations.csv
    csv_path = Path(out_root) / f"{split}_annotations.csv"
    n_box = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_name", "bbox_x", "bbox_y",
                    "bbox_width", "bbox_height", "label_name"])
        for a in coco["annotations"]:
            x, y, bw, bh = a["bbox"]            # COCO [x,y,w,h] 그대로
            img_name = id2name[a["image_id"]]
            w.writerow([img_name, int(round(x)), int(round(y)),
                        int(round(bw)), int(round(bh)), "implant"])
            n_box += 1

    print(f"  [{split}] images={len(coco['images'])} boxes={n_box} -> {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-root", default="/workspace/dataset_coco")
    ap.add_argument("--out-root", default="multimodal-data/implant_dataset")
    args = ap.parse_args()

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        convert_split(split, args.coco_root, args.out_root)

    print(f"\n[done] {args.out_root}/ 아래 images/{{train,val,test}} + *_annotations.csv 생성")


if __name__ == "__main__":
    main()
