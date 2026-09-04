
#best_sam 모델로 이미지의 마스크 생성하기 (파인튜닝된 샘 모델로 잘 나오는지 보기 )
#이미지 전처리 부분 해당 코드에 있음 

'''
원본 이미지
 → SAM image_encoder
 → feature
 → ImplantMaskHead   <- 이 부분이 파인튜닝 된거임 (best_sam_implant_head1.pth)
 → mask
 '''

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from sam2.build_sam import build_sam2
from sam2_adapter_decoder_train import Adapter, UNetDecoder,SAM2AdapterUNet

# =====================
# 경로 설정
# =====================

#IMAGE_DIR = "/home/ubuntu/workspace/seg_test_sample_2"
IMAGE_DIR = "/home/ubuntu/workspace/data/datas"
MODEL_PATH = "/home/ubuntu/workspace/implantny/best_sam2.1_tiny_adapter_unet.pth"
#SAVE_DIR = "/home/ubuntu/workspace/implantny/seg_test_res_AIhub1_preprocessing_ver2"
SAVE_DIR = "/home/ubuntu/workspace/seg_masks_train"

SAM2_CHECKPOINT = "/home/ubuntu/checkpoints/sam2/sam2.1_hiera_tiny.pt"
SAM2_CONFIG="configs/sam2.1/sam2.1_hiera_t.yaml"

os.makedirs(SAVE_DIR, exist_ok=True)

# =====================
# 학습 코드에 있던 head import
# =====================


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =====================
# SAM 불러오기
# =====================
sam2_model=build_sam2(SAM2_CONFIG,SAM2_CHECKPOINT,device=device)
adapter=Adapter(in_channels=256, out_channels=256)
decoder=UNetDecoder(in_channels=256)
model=SAM2AdapterUNet(sam2_model,adapter,decoder).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


# =====================
# SAM 입력 전처리 함수
# =====================
def preprocess_for_sam(image_pil):
    image = np.array(image_pil.convert("RGB"))

    # SAM은 긴 변을 1024로 맞춰서 입력
    h, w = image.shape[:2]
    scale = 1024 / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # padding해서 1024 x 1024 만들기
    padded = np.zeros((1024, 1024, 3), dtype=np.uint8)
    padded[:new_h, :new_w, :] = image

    x = torch.as_tensor(padded, device=device)
    x = x.permute(2, 0, 1).contiguous().float()

    # SAM normalize
    pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(3, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375], device=device).view(3, 1, 1)

    x = (x - pixel_mean) / pixel_std
    x = x.unsqueeze(0)

    return x, (new_w, new_h)


# =====================
# 예측
# =====================
with torch.no_grad():
    count=0
    for img_path in Path(IMAGE_DIR).rglob("*.jpg"):
        img_name=img_path.name

        image_pil = Image.open(img_path).convert("RGB")
        original_w, original_h = image_pil.size

        x, resized_size = preprocess_for_sam(image_pil)

        pred=model(x)
        pred = torch.sigmoid(pred)

        mask = pred.squeeze().cpu().numpy()

        # head 출력 크기 -> 1024 x 1024로 확대
        mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_LINEAR)

        resized_w, resized_h = resized_size

        # padding 제거
        mask = mask[:resized_h, :resized_w]

        # 원본 크기로 복원
        mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        # threshold 적용
        mask = (mask > 0.5).astype(np.uint8) * 255

        #0. 팽창으로 끊긴 부분 연결 후 구멍 채우기 
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)

        #1. 내부 구멍 채우기 
        contours,_ =cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        mask= np.zeros_like(mask)
        cv2.drawContours(mask,contours,-1,255,thickness=cv2.FILLED)

        #2. 작은 노이즈 제거 (가장 큰 덩어리만 남기고 나머지 제거)
        num_labels,labels,stats, _ =cv2.connectedComponentsWithStats(mask)
        if num_labels>1:
            largest= 1+ np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask=np.where(labels==largest,255,0).astype(np.uint8)
        else:
            print("EMPTY mask:",img_name)
        

     
        

        save_name = Path(img_name).stem + "_pred.png"
        save_path = os.path.join(SAVE_DIR, save_name)

        cv2.imwrite(save_path, mask)

        count+=1
        if count%1000==0:
            print(count,"done")

print("예측 마스크 저장 완료:", SAVE_DIR)