import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# =========================
# 경로 설정
# =========================
TRAIN_IMG_DIR = "/home/ubuntu/workspace/SAM_finetuning_dataset_1_final/train/images"
TRAIN_MASK_DIR = "/home/ubuntu/workspace/SAM_finetuning_dataset_1_final/train/masks"
VALID_IMG_DIR = "/home/ubuntu/workspace/SAM_finetuning_dataset_1_final/val/images"
VALID_MASK_DIR = "/home/ubuntu/workspace/SAM_finetuning_dataset_1_final/val/masks"

SAVE_PATH = "/home/ubuntu/workspace/implantny/best_sam2.1_tiny_adapter_unet2.pth"

IMAGE_SIZE = 512
BATCH_SIZE = 2
EPOCHS = 60
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


# =========================
# Dataset
# =========================
class SegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=512):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size

        self.images = sorted([
            p for p in self.image_dir.rglob("*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        # 이미지와 같은 이름의 mask 찾기
        mask_candidates=list(self.mask_dir.glob(f"{img_path.stem}*"))
        mask_path=mask_candidates[0] 

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.astype(np.float32) / 255.0

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask = (mask > 127).astype(np.float32)

        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


# =========================
# Loss
# =========================
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        smooth = 1e-6

        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

        dice_loss = 1 - ((2 * intersection + smooth) / (union + smooth))
        dice_loss = dice_loss.mean()

        return bce_loss + dice_loss


# =========================
# 작은 Adapter
# =========================
class Adapter(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()

        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.adapter(x)


# =========================
# 간단한 U-Net Decoder
# =========================
class UNetDecoder(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()

        self.up1 = self.block(in_channels, 128)
        self.up2 = self.block(128, 64)
        self.up3 = self.block(64, 32)
        self.up4 = self.block(32, 16)

        self.final = nn.Conv2d(16, 1, kernel_size=1)

    def block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final(x)
        return x


# =========================
# SAM2 Encoder + Adapter + Decoder
# =========================
class SAM2AdapterUNet(nn.Module):
    def __init__(self, sam2_model, adapter, decoder):
        super().__init__()

        self.sam2_encoder = sam2_model.image_encoder
        self.adapter = adapter
        self.decoder = decoder

        # SAM2 encoder freeze
        for p in self.sam2_encoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            features = self.sam2_encoder(x)

        # SAM2 encoder output 형태가 dict/list/tensor일 수 있어서 처리
        if isinstance(features, dict):
            if "vision_features" in features:
                features = features["vision_features"]
            elif "backbone_fpn" in features:
                features = features["backbone_fpn"][-1]
            else:
                features = list(features.values())[-1]

        if isinstance(features, (list, tuple)):
            features = features[-1]

        x = self.adapter(features)
        x = self.decoder(x)

        # 최종 mask 크기를 원본 입력 크기로 맞춤
        x = F.interpolate(
            x,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        return x


# =========================
# SAM2 로드
# =========================
# 먼저 sam2 repo 설치 필요:
# git clone https://github.com/facebookresearch/sam2.git
# cd sam2
# pip install -e .

from sam2.build_sam import build_sam2

SAM2_CONFIG = "sam2.1_hiera_t.yaml"
SAM2_CHECKPOINT ="/workspace/implant_seg/checkpoints/sam2.1_hiera_tiny.pt"

sam2_model = build_sam2(
    "configs/sam2.1/sam2.1_hiera_t.yaml",
    SAM2_CHECKPOINT,
    device=device
)

sam2_model.eval()


# =========================
# Adapter input channel 자동 확인
# =========================
with torch.no_grad():
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    dummy_features = sam2_model.image_encoder(dummy)

    if isinstance(dummy_features, dict):
        if "vision_features" in dummy_features:
            dummy_features = dummy_features["vision_features"]
        elif "backbone_fpn" in dummy_features:
            dummy_features = dummy_features["backbone_fpn"][-1]
        else:
            dummy_features = list(dummy_features.values())[-1]

    if isinstance(dummy_features, (list, tuple)):
        dummy_features = dummy_features[-1]

    in_channels = dummy_features.shape[1]

print("SAM2 feature channels:", in_channels)


adapter = Adapter(in_channels=in_channels, out_channels=256)
decoder = UNetDecoder(in_channels=256)

model = SAM2AdapterUNet(
    sam2_model=sam2_model,
    adapter=adapter,
    decoder=decoder
).to(device)


# =========================
# DataLoader
# =========================
train_dataset = SegDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, IMAGE_SIZE)
valid_dataset = SegDataset(VALID_IMG_DIR, VALID_MASK_DIR, IMAGE_SIZE)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =========================
# Optimizer: adapter + decoder만 학습
# =========================
optimizer = torch.optim.AdamW(
    list(model.adapter.parameters()) + list(model.decoder.parameters()),
    lr=LR,
    weight_decay=1e-4
)

criterion = DiceBCELoss()


# =========================
# IoU 계산
# =========================
def compute_iou(logits, masks, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = ((preds + masks) > 0).float().sum(dim=(1, 2, 3))

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


# =========================
# Train
# =========================
best_valid_iou = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()

    train_loss = 0.0
    train_iou = 0.0

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_iou += compute_iou(logits.detach(), masks)

    train_loss /= len(train_loader)
    train_iou /= len(train_loader)

    model.eval()

    valid_loss = 0.0
    valid_iou = 0.0

    with torch.no_grad():
        for images, masks in tqdm(valid_loader, desc=f"Epoch {epoch} Valid"):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            valid_loss += loss.item()
            valid_iou += compute_iou(logits, masks)

    valid_loss /= len(valid_loader)
    valid_iou /= len(valid_loader)

    print(
        f"[Epoch {epoch:03d}] "
        f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} || "
        f"Valid Loss: {valid_loss:.4f} | Valid IoU: {valid_iou:.4f}"
    )

    if valid_iou > best_valid_iou:
        best_valid_iou = valid_iou

        torch.save(model.state_dict(), SAVE_PATH)

        print(f"best model saved: {SAVE_PATH}")



    test_loader=DataLoader(SegDataset("/home/ubuntu/workspace/SAM_finetuning_dataset_1_final/test/images","/home/ubuntu/workspace/SAM_finetuning_dataset_1_final/test/masks",IMAGE_SIZE),batch_size=BATCH_SIZE,shuffle=False,num_workers=0)

    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()
    test_loss,test_iou=0.0,0.0

    with torch.no_grad():
        for images,masks in test_loader:
    
            images,masks= images.to(device), masks.to(device)
            logits=model(images)
            test_loss+= criterion(logits,masks).item()
            test_iou+=compute_iou(logits,masks)

    print(f"[Test] Loss: {test_loss/len(test_loader):.4f} | IoU: {test_iou/len(test_loader):.4f}")

