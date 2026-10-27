"""
② 순정 SAM(vit_h) automatic mask generator로 샘플 이미지들에 초기 마스크 자동 생성
   - 임플란트 형태 필터(면적비/종횡비)로 후보 마스크 걸러냄
   - 이후 사람이 잘 생성된 마스크를 직접 선별해 image:mask 1:1 폴더 구성
   (원본: implant_ny/make_seg_model/SAM1_new.py)

실행: python generate_sam1_masks.py --config configs/02_segmentation.yaml
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_utils import load_config_from_cli


def is_implant_like(mask, image_shape, fcfg):
    """임플란트 형태 필터: 면적비/종횡비/높이 조건"""
    h, w = image_shape[:2]
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return False

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    box_w = x2 - x1 + 1
    box_h = y2 - y1 + 1
    area = mask.sum()

    aspect_ratio = box_h / (box_w + 1e-6)
    area_ratio = area / (h * w)

    if area_ratio < fcfg["min_area_ratio"]:
        return False
    if area_ratio > fcfg["max_area_ratio"]:
        return False
    if aspect_ratio < fcfg["min_aspect_ratio"]:
        return False
    if box_h < h * fcfg["min_box_h_ratio"]:
        return False
    return True


def load_sam(scfg):
    from segmentation.train_sam2_adapter import choose_device
    device = str(choose_device(cfg.get("device", "auto")))
    print("device:", device)

    sam = sam_model_registry[scfg["model_type"]](checkpoint=str(scfg["checkpoint"]))
    sam.to(device=device)

    return SamAutomaticMaskGenerator(model=sam, **scfg["generator"])


def generate_masks(mask_generator, scfg):
    image_dir = Path(scfg["image_dir"])
    output_dir = Path(scfg["output_mask_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])
    print("총 이미지 수:", len(image_paths))

    failed_images = []
    for img_idx, img_path in enumerate(image_paths):
        print(f"\n[{img_idx + 1}/{len(image_paths)}]")
        print("처리 중:", img_path)
        try:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                raise ValueError("읽기실패")
            image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print("이미지 읽기 실패:", e)
            failed_images.append(str(img_path))
            continue

        masks = mask_generator.generate(image_rgb)
        print("전체 마스크 수:", len(masks))

        relative_dir = img_path.parent.relative_to(image_dir)
        save_dir = output_dir / relative_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0
        for m in masks:
            mask = m["segmentation"].astype(np.uint8)
            if not is_implant_like(mask, image_rgb.shape, scfg["filter"]):
                continue

            save_path = save_dir / f"{img_path.stem}_mask_{saved_count:02d}.png"
            cv2.imwrite(str(save_path), mask * 255)
            saved_count += 1

        print("저장된 마스크 수:", saved_count)

    if failed_images:
        print("\n읽기 실패한 이미지:", len(failed_images))


def main():
    cfg = load_config_from_cli(__doc__)
    scfg = cfg["sam1"]

    if not Path(scfg["image_dir"]).is_dir():
        raise FileNotFoundError(f"이미지 폴더가 없습니다: {scfg['image_dir']}")
    if not Path(scfg["checkpoint"]).is_file():
        raise FileNotFoundError(f"SAM 체크포인트가 없습니다: {scfg['checkpoint']}")

    mask_generator = load_sam(scfg)
    generate_masks(mask_generator, scfg)
    print("\n모든 작업 완료")


if __name__ == "__main__":
    main()
