"""
⑤ SAM2(hiera-tiny) encoder frozen + Adapter + U-Net decoder 파인튜닝
   → 임플란트 맞춤형 segmentation model 제작
   (원본: implant_ny/make_seg_model/sam2_adapter_decoder_train.py)

사전 준비: sam2 repo 설치 필요
    git clone https://github.com/facebookresearch/sam2.git && cd sam2 && pip install -e .

실행: python train_sam2_adapter.py --config configs/02_segmentation.yaml

참고: Adapter/UNetDecoder/SAM2AdapterUNet 클래스는 predict_masks.py에서도 import해서 사용
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_utils import load_config_from_cli


def choose_device(requested: str = "auto"):
    """yaml의 device 설정을 따른다 (auto | cuda | mps | cpu).

    다른 스테이지(metric/attributes 등)와 동일한 규칙. segmentation만 cuda/cpu로
    고정돼 있어 Apple Silicon에서 MPS를 못 쓰고 yaml 설정도 무시됐다.
    """
    import torch

    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



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
        mask_candidates = list(self.mask_dir.rglob(f"{img_path.stem}_mask_*.png"))
        mask_path = mask_candidates[0]

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
        return bce_loss + dice_loss.mean()


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
        return self.final(x)


# =========================
# SAM2 Encoder + Adapter + Decoder
# =========================
def _extract_features(features):
    """SAM2 encoder output이 dict/list/tensor일 수 있어서 통일 처리"""
    if isinstance(features, dict):
        if "vision_features" in features:
            features = features["vision_features"]
        elif "backbone_fpn" in features:
            features = features["backbone_fpn"][-1]
        else:
            features = list(features.values())[-1]
    if isinstance(features, (list, tuple)):
        features = features[-1]
    return features


class SAM2AdapterUNet(nn.Module):
    def __init__(self, sam2_model, adapter, decoder, image_size=512):
        super().__init__()
        self.sam2_encoder = sam2_model.image_encoder
        self.adapter = adapter
        self.decoder = decoder
        self.image_size = image_size

        # SAM2 encoder freeze
        for p in self.sam2_encoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            features = self.sam2_encoder(x)
        features = _extract_features(features)

        x = self.adapter(features)
        x = self.decoder(x)

        # 최종 mask 크기를 입력 크기로 맞춤
        return F.interpolate(
            x, size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        )


def build_model(tcfg, device):
    """SAM2 로드 + adapter 입력 채널 자동 확인 + 전체 모델 조립"""
    from sam2.build_sam import build_sam2

    sam2_model = build_sam2(tcfg["sam2_config"], str(tcfg["sam2_checkpoint"]), device=device)
    sam2_model.eval()

    image_size = tcfg["image_size"]
    with torch.no_grad():
        dummy = torch.randn(1, 3, image_size, image_size).to(device)
        in_channels = _extract_features(sam2_model.image_encoder(dummy)).shape[1]
    print("SAM2 feature channels:", in_channels)

    adapter = Adapter(in_channels=in_channels, out_channels=tcfg["adapter_channels"])
    decoder = UNetDecoder(in_channels=tcfg["adapter_channels"])
    return SAM2AdapterUNet(sam2_model, adapter, decoder, image_size).to(device)


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


def main():
    cfg = load_config_from_cli(__doc__)
    tcfg = cfg["sam2_train"]

    device = choose_device(cfg.get("device", "auto"))
    print("device:", device)

    ds_dir = Path(tcfg["dataset_dir"])
    image_size = tcfg["image_size"]
    save_path = str(tcfg["save_path"])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    model = build_model(tcfg, device)

    train_loader = DataLoader(
        SegDataset(ds_dir / "train/images", ds_dir / "train/masks", image_size),
        batch_size=tcfg["batch_size"], shuffle=True, num_workers=0,
    )
    valid_loader = DataLoader(
        SegDataset(ds_dir / "val/images", ds_dir / "val/masks", image_size),
        batch_size=tcfg["batch_size"], shuffle=False, num_workers=0,
    )

    # adapter + decoder만 학습 (encoder는 frozen)
    optimizer = torch.optim.AdamW(
        list(model.adapter.parameters()) + list(model.decoder.parameters()),
        lr=tcfg["lr"], weight_decay=tcfg["weight_decay"],
    )
    criterion = DiceBCELoss()

    best_valid_iou = 0.0
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        train_loss, train_iou = 0.0, 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            images, masks = images.to(device), masks.to(device)

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
        valid_loss, valid_iou = 0.0, 0.0
        with torch.no_grad():
            for images, masks in tqdm(valid_loader, desc=f"Epoch {epoch} Valid"):
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                valid_loss += criterion(logits, masks).item()
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
            torch.save(model.state_dict(), save_path)
            print(f"best model saved: {save_path}")

    # ---- 학습 종료 후 best 체크포인트로 test 평가 1회 ----
    # (원본 코드는 epoch 루프 안에서 매번 test를 돌렸음 — 종료 후 1회로 수정)
    test_loader = DataLoader(
        SegDataset(ds_dir / "test/images", ds_dir / "test/masks", image_size),
        batch_size=tcfg["batch_size"], shuffle=False, num_workers=0,
    )
    model.load_state_dict(torch.load(save_path))
    model.eval()

    test_loss, test_iou = 0.0, 0.0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            test_loss += criterion(logits, masks).item()
            test_iou += compute_iou(logits, masks)

    print(f"[Test] Loss: {test_loss / len(test_loader):.4f} | IoU: {test_iou / len(test_loader):.4f}")


if __name__ == "__main__":
    main()
