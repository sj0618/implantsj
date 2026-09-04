import os
import cv2
import numpy as np
import torch
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator



# =========================
# 설정
# =========================
IMAGE_DIR = Path("/home/ubuntu/workspace/seg_dataset/images")         
OUTPUT_MASK_DIR =Path( "/home/ubuntu/workspace/seg_dataset/masks")   
SAM_CHECKPOINT = Path("/home/ubuntu/checkpoints/sam/sam_vit_h_4b8939.pth")
MODEL_TYPE = "vit_h"

IMAGE_EXTS={'.jpg'}


# =========================
# 임플란트 형태 필터
# =========================
def is_implant_like(mask, image_shape):

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

    # ===== 필터 조건 =====

    if area_ratio < 0.10:
        return False

    if area_ratio > 0.97:
        return False

    if aspect_ratio < 1.5:
        return False

    if box_h < h * 0.3:
        return False

    return True


def check_paths():
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f'이미지 폴더가 없습니다:{IMAGE_DIR}')
    
    if not SAM_CHECKPOINT.is_file():
        raise FileNotFoundError(f'SAM 체크포인트가 없습니다: {SAM_CHECKPOINT}')

    OUTPUT_MASK_DIR.mkdir(parents=True,exist_ok=True)

# =========================
# SAM 모델 로드
# =========================
def load_sam():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("device:", device)

    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))

    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=64,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.80,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100
    )

    return mask_generator


# =========================
# 마스크 생성
# =========================
def generate_masks(mask_generator):

    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

    image_paths = sorted([
        p for p in Path(IMAGE_DIR).rglob("*")
        if p.is_file()
        and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    print("총 이미지 수:", len(image_paths))
    failed_images=[]
    for img_idx, img_path in enumerate(image_paths):

        print(f"\n[{img_idx+1}/{len(image_paths)}]")
        print("처리 중:", img_path)
        try:
            img_bgr=cv2.imread(str(img_path))
            if img_bgr is None:
                raise ValueError("읽기실패")
            image_rgb=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)

        except Exception as e:
            print('이미지 읽기 실패:',e)
            failed_images.append(str(img_path))
            continue

  


        masks = mask_generator.generate(image_rgb)

        print("전체 마스크 수:", len(masks))

        relative_dir=img_path.parent.relative_to(IMAGE_DIR)

        save_dir=Path(OUTPUT_MASK_DIR)/relative_dir
        os.makedirs(save_dir,exist_ok=True)

        saved_count = 0

        for i, m in enumerate(masks):

            mask = m["segmentation"].astype(np.uint8)

            if not is_implant_like(mask, image_rgb.shape):
                continue

            mask_png = mask * 255

            save_name = (
                f"{img_path.stem}_mask_{saved_count:02d}.png"
            )

            save_path = save_dir / save_name

            cv2.imwrite(str(save_path),mask_png)

            saved_count += 1

        print("저장된 마스크 수:", saved_count)


# =========================
# 메인 함수
# =========================
def main():
    check_paths()
    mask_generator = load_sam()

    generate_masks(mask_generator)

    print("\n모든 작업 완료")


# =========================
# 실행
# =========================
if __name__ == "__main__":
    main()