"""
dataset2 폴더 구조 생성 스크립트

목표 구조:
dataset2/
├── test/          ← 기존 test 13장 (고정)
├── 01/
│   ├── train/     ← 기존 train 83장
│   └── val/       ← 기존 val 23장
└── 02/
    ├── train/     ← 신규 100장 중 80장 (랜덤)
    └── val/       ← 신규 100장 중 20장 (랜덤)
"""

import json
import random
import shutil
from pathlib import Path

random.seed(42)

BASE = Path("/Users/gim-yesong/Desktop/implant")
SRC_COCO = BASE / "dataset_coco"
SRC_100_IMAGES = BASE / "ls_data" / "pseudo_batch_01_images"
SRC_100_LABELS = BASE / "ls_data" / "pseudo_batch_01_coco" / "result.json"
OUT = BASE / "dataset2"


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def copy_split(src_split: Path, dst_split: Path):
    """dataset_coco/{split}/ → dataset2/{dst}/ 복사 (images + annotations.json)"""
    dst_split.mkdir(parents=True, exist_ok=True)
    dst_img = dst_split / "images"
    dst_img.mkdir(exist_ok=True)

    # 이미지 복사
    src_img = src_split / "images"
    for f in sorted(src_img.iterdir()):
        shutil.copy2(f, dst_img / f.name)

    # annotations.json 복사
    shutil.copy2(src_split / "annotations.json", dst_split / "annotations.json")

    imgs = list(sorted(src_img.iterdir()))
    print(f"  {dst_split.relative_to(BASE)} : images={len(imgs)}")
    return [f.name for f in imgs]


def make_coco_subset(all_coco: dict, image_names: list[str], dst_dir: Path):
    """
    all_coco에서 image_names에 해당하는 images+annotations만 추출해
    dst_dir/annotations.json 저장 + images 복사.
    이미지는 SRC_100_IMAGES에서 가져옴.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_img_dir = dst_dir / "images"
    dst_img_dir.mkdir(exist_ok=True)

    name_set = set(image_names)

    # 이미지 필터 (file_name이 절대경로일 수 있어 basename 사용)
    filtered_imgs = []
    id_map = {}  # old_id → new_id
    for i, img in enumerate(all_coco["images"]):
        basename = Path(img["file_name"]).name
        if basename in name_set:
            new_id = len(filtered_imgs)
            id_map[img["id"]] = new_id
            filtered_imgs.append({**img, "id": new_id, "file_name": basename})

    # 어노테이션 필터
    old_img_ids = set(id_map.keys())
    filtered_anns = []
    for ann in all_coco["annotations"]:
        if ann["image_id"] in old_img_ids:
            new_ann = {**ann, "id": len(filtered_anns), "image_id": id_map[ann["image_id"]]}
            filtered_anns.append(new_ann)

    # COCO JSON 저장
    subset = {
        "images": filtered_imgs,
        "annotations": filtered_anns,
        "categories": all_coco["categories"],
        "info": all_coco.get("info", {}),
    }
    with open(dst_dir / "annotations.json", "w") as f:
        json.dump(subset, f, indent=2)

    # 이미지 복사
    for img in filtered_imgs:
        src = SRC_100_IMAGES / img["file_name"]
        shutil.copy2(src, dst_img_dir / img["file_name"])

    print(f"  {dst_dir.relative_to(BASE)} : images={len(filtered_imgs)}, annotations={len(filtered_anns)}")
    return [img["file_name"] for img in filtered_imgs]


# ── 메인 ──────────────────────────────────────────────────────────────────────

print("=" * 60)
print("dataset2 생성 시작")
print("=" * 60)

# ── 1. test (13장) ────────────────────────────────────────────────────────────
print("\n[1] test 세트 복사 (13장)")
test_names = copy_split(SRC_COCO / "test", OUT / "test")

# ── 2. 01/train (83장), 01/val (23장) ────────────────────────────────────────
print("\n[2] 01 세트 복사 (83 train + 23 val)")
train01_names = copy_split(SRC_COCO / "train", OUT / "01" / "train")
val01_names   = copy_split(SRC_COCO / "val",   OUT / "01" / "val")

# ── 3. 02 세트: 100장 → 80 train / 20 val ────────────────────────────────────
print("\n[3] 02 세트 구성 (100장 → 80 train / 20 val)")

with open(SRC_100_LABELS) as f:
    coco_100 = json.load(f)

# 전체 파일명 목록 (basename)
all_100_names = sorted({Path(img["file_name"]).name for img in coco_100["images"]})
assert len(all_100_names) == 100, f"예상 100장인데 {len(all_100_names)}장"

random.shuffle(all_100_names)
train02_names_target = all_100_names[:80]
val02_names_target   = all_100_names[80:]

train02_names = make_coco_subset(coco_100, train02_names_target, OUT / "02" / "train")
val02_names   = make_coco_subset(coco_100, val02_names_target,   OUT / "02" / "val")

# ── 4. Leakage 검증 ───────────────────────────────────────────────────────────
print("\n[4] Leakage 검증 (test ∩ train/val = ∅)")

test_set    = set(test_names)
train01_set = set(train01_names)
val01_set   = set(val01_names)
train02_set = set(train02_names)
val02_set   = set(val02_names)
all_train   = train01_set | val01_set | train02_set | val02_set

overlap = test_set & all_train
assert len(overlap) == 0, f"LEAKAGE 발견! test ∩ train/val = {overlap}"
print("  test ∩ (01+02) = ∅  ✓")

# 01과 02 간 중복도 확인
overlap_0102 = (train01_set | val01_set) & (train02_set | val02_set)
assert len(overlap_0102) == 0, f"01과 02 간 중복! {overlap_0102}"
print("  01 ∩ 02 = ∅  ✓")

# ── 5. 최종 요약 ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("완료! 최종 구조:")
print(f"  dataset2/test/         : {len(test_names)}장")
print(f"  dataset2/01/train/     : {len(train01_names)}장")
print(f"  dataset2/01/val/       : {len(val01_names)}장")
print(f"  dataset2/02/train/     : {len(train02_names)}장")
print(f"  dataset2/02/val/       : {len(val02_names)}장")
print(f"  전체 고유 이미지       : {len(test_set | train01_set | val01_set | train02_set | val02_set)}장")
print("=" * 60)
