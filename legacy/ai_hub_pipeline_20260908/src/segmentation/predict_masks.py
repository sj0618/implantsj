"""
⑥ 파인튜닝된 SAM2+Adapter+U-Net 모델로 전체 데이터(42클래스) 마스크 예측
   + 후처리: closing으로 끊긴 부분 연결 → 내부 구멍 채우기 → 최대 덩어리만 남기기
   → 결과: save_dir/<stem>_pred.png
   (원본: implant_ny/make_seg_model/SAM_mask_produce_util.py)

실행: python predict_masks.py --config configs/02_segmentation.yaml
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_utils import load_config_from_cli
from segmentation.train_sam2_adapter import build_model, choose_device


def preprocess_for_sam(image_pil, device):
    """SAM 입력 전처리: 긴 변 1024로 리사이즈 + 패딩 + normalize"""
    image = np.array(image_pil.convert("RGB"))

    h, w = image.shape[:2]
    scale = 1024 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.zeros((1024, 1024, 3), dtype=np.uint8)
    padded[:new_h, :new_w, :] = image

    x = torch.as_tensor(padded, device=device)
    x = x.permute(2, 0, 1).contiguous().float()

    pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(3, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375], device=device).view(3, 1, 1)
    x = (x - pixel_mean) / pixel_std

    return x.unsqueeze(0), (new_w, new_h)


def postprocess(mask, pcfg, img_name):
    """closing → 구멍 채우기 → 최대 덩어리만 남기기"""
    k = pcfg["close_kernel"]
    if k and k > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if pcfg["fill_holes"]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    if pcfg["largest_only"]:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = np.where(labels == largest, 255, 0).astype(np.uint8)
        else:
            print("EMPTY mask:", img_name)

    return mask


def main():
    cfg = load_config_from_cli(__doc__)
    tcfg = cfg["sam2_train"]
    pcfg = cfg["predict"]

    device = choose_device(cfg.get("device", "auto"))
    print("Device:", device)

    save_dir = Path(pcfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # 학습 때와 같은 구조로 모델 조립 후 파인튜닝 체크포인트 로드
    model = build_model(tcfg, device)
    model.load_state_dict(torch.load(str(tcfg["save_path"]), map_location=device))
    model.eval()
    # 1024 입력을 그대로 쓰도록 출력 크기 조정
    model.image_size = 1024

    threshold = pcfg["threshold"]

    with torch.no_grad():
        count = 0
        for img_path in Path(pcfg["image_dir"]).rglob("*.jpg"):
            image_pil = Image.open(img_path).convert("RGB")
            original_w, original_h = image_pil.size

            x, (resized_w, resized_h) = preprocess_for_sam(image_pil, device)

            pred = torch.sigmoid(model(x))
            mask = pred.squeeze().cpu().numpy()

            # 출력 -> 1024x1024 -> 패딩 제거 -> 원본 크기 복원
            mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            mask = mask[:resized_h, :resized_w]
            mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

            mask = (mask > threshold).astype(np.uint8) * 255
            mask = postprocess(mask, pcfg["postprocess"], img_path.name)

            cv2.imwrite(str(save_dir / f"{img_path.stem}_pred.png"), mask)

            count += 1
            if count % 1000 == 0:
                print(count, "done")

    print("예측 마스크 저장 완료:", save_dir)


if __name__ == "__main__":
    main()
