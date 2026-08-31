#!/usr/bin/env python3
"""
COCO export(result.json + images/)를 train/val/test = 7:2:1 로 분할.

핵심 원칙:
- 분할은 '한 번만' 정하고 파일명 목록으로 얼려둔다 (splits/*.txt).
  -> 나중에 pseudo-label로 이미지가 늘어나도 val/test는 절대 안 변함.
- 원본은 건드리지 않고 복사만 한다.
- 각 split의 json은 그 폴더만 보고 학습 가능하도록 file_name을 basename으로 정리.

사용법:
  python split_coco.py \
      --coco-dir "COCO export" \
      --out-dir  "dataset_coco"

옵션:
  --seed 42          재현용 시드 (기본 42)
  --ratios 0.7 0.2 0.1
"""
import argparse, json, random, shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-dir", required=True,
                    help="result.json 과 images/ 가 들어있는 폴더")
    ap.add_argument("--out-dir", default="dataset_coco")
    ap.add_argument("--json-name", default="result.json")
    ap.add_argument("--images-subdir", default="images")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.2, 0.1],
                    metavar=("TRAIN", "VAL", "TEST"))
    args = ap.parse_args()

    assert abs(sum(args.ratios) - 1.0) < 1e-6, "ratios 합이 1이어야 합니다"

    coco_dir = Path(args.coco_dir)
    img_src_dir = coco_dir / args.images_subdir
    out_dir = Path(args.out_dir)

    coco = json.loads((coco_dir / args.json_name).read_text(encoding="utf-8"))
    images = coco["images"]
    anns = coco["annotations"]
    cats = coco["categories"]

    print(f"[load] images={len(images)}  annotations={len(anns)}  "
          f"categories={[c['name'] for c in cats]}")

    # image_id -> annotation 리스트
    ann_by_img = {}
    for a in anns:
        ann_by_img.setdefault(a["image_id"], []).append(a)

    # 라벨(박스)이 하나도 없는 이미지는 분할에서 제외할지 결정.
    # detection에서는 '박스 없는 배경 이미지'도 학습에 도움되지만,
    # 여기선 seed 전량 라벨된 상황이므로 전부 포함.
    random.seed(args.seed)
    ids = [im["id"] for im in images]
    random.shuffle(ids)

    n = len(ids)
    n_train = int(n * args.ratios[0])
    n_val = int(n * args.ratios[1])
    # 나머지 전부 test (반올림 오차 흡수)
    split_ids = {
        "train": set(ids[:n_train]),
        "val":   set(ids[n_train:n_train + n_val]),
        "test":  set(ids[n_train + n_val:]),
    }
    print(f"[split] train={len(split_ids['train'])} "
          f"val={len(split_ids['val'])} test={len(split_ids['test'])}")

    im_by_id = {im["id"]: im for im in images}
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    for name in ("train", "val", "test"):
        sub_ids = split_ids[name]
        sub_images = [im_by_id[i] for i in sub_ids]

        img_out_dir = out_dir / name / "images"
        img_out_dir.mkdir(parents=True, exist_ok=True)

        # 이미지 복사 + file_name을 basename으로 정규화
        frozen_names = []
        new_images = []
        for im in sub_images:
            base = Path(im["file_name"]).name          # 경로/접두어 제거
            src = img_src_dir / base
            if not src.exists():
                # Label Studio가 file_name에 경로를 넣는 경우 대비 폴백
                cand = list(img_src_dir.glob(base))
                if cand:
                    src = cand[0]
                else:
                    print(f"  [warn] 이미지 못 찾음: {base} (건너뜀)")
                    continue
            shutil.copy2(src, img_out_dir / base)
            frozen_names.append(base)
            im2 = dict(im)
            im2["file_name"] = base
            new_images.append(im2)

        kept = {im["id"] for im in new_images}
        new_anns = [a for a in anns if a["image_id"] in kept]

        out_json = out_dir / name / "annotations.json"
        out_json.write_text(json.dumps(
            {"images": new_images, "annotations": new_anns, "categories": cats},
            ensure_ascii=False), encoding="utf-8")

        # 얼려둔 목록 (다음 분할 때 재사용/검증용)
        (splits_dir / f"{name}.txt").write_text(
            "\n".join(sorted(frozen_names)), encoding="utf-8")

        print(f"  [{name}] images={len(new_images)} anns={len(new_anns)} "
              f"-> {out_json}")

    print(f"\n[done] {out_dir}/ 아래 train|val|test 폴더 + splits/*.txt 생성 완료")
    print("splits/val.txt, splits/test.txt 는 앞으로 '고정'. "
          "pseudo-label로 늘어나는 이미지는 train 에만 추가하세요.")


if __name__ == "__main__":
    main()
