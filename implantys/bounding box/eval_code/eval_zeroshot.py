#!/usr/bin/env python3
"""
Zero-shot baseline 평가: fine-tuning(LoRA) 없이 사전학습 Grounding DINO 그대로
test 셋에 'implant' 프롬프트로 돌려서 mAP 측정.

목적: fine-tuning의 순수 기여도를 보여주기 위한 baseline.
  - 이 results(zero-shot)와 fine-tuned(epoch_20) results를 비교하면
    "fine-tuning이 실제로 얼마나 성능을 올렸는지"가 정량으로 드러남.

실행 (repo 루트에서):
  export LD_LIBRARY_PATH=$(python -c "import torch,os;print(os.path.dirname(torch.__file__))")/lib:$LD_LIBRARY_PATH
  python eval_zeroshot.py
"""
import os, json, copy
import torch
from torchvision.ops import box_convert

from groundingdino.util.inference import load_model, load_image, predict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TEST_IMG_DIR = "multimodal-data/implant_dataset/images/test"
TEST_COCO_GT = "/workspace/dataset_coco/test/annotations.json"
BASE_WEIGHTS = "weights/groundingdino_swint_ogc.pth"
MODEL_CFG    = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
TEXT_PROMPT  = "implant"
BOX_TH       = 0.25
TXT_TH       = 0.20


class MC:
    """load_model이 기대하는 model_config. LoRA 없이 base만 로드."""
    def __init__(self):
        self.config_path = MODEL_CFG
        self.weights_path = BASE_WEIGHTS
        # lora 경로는 use_lora=False라 사용 안 됨 (형식상 둠)
        self.lora_weights = None
        self.lora_weigths = None


def build_gt_index(coco_gt):
    name2id, id2wh = {}, {}
    for im in coco_gt.dataset["images"]:
        name2id[im["file_name"]] = im["id"]
        id2wh[im["id"]] = (im["width"], im["height"])
    return name2id, id2wh


def main():
    coco_gt = COCO(TEST_COCO_GT)
    cat_id = coco_gt.dataset["categories"][0]["id"]  # implant의 실제 id (0)
    print(f"GT category_id = {cat_id}")
    name2id, id2wh = build_gt_index(coco_gt)

    # ---- fine-tuning 없이 base 모델만 로드 (zero-shot) ----
    print("\n[load] 사전학습 Grounding DINO (fine-tuning 없음)")
    model = load_model(MC(), use_lora=False)

    results = []
    files = [f for f in os.listdir(TEST_IMG_DIR)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for fn in files:
        if fn not in name2id:
            continue
        img_id = name2id[fn]
        W, H = id2wh[img_id]
        image_source, image = load_image(os.path.join(TEST_IMG_DIR, fn))
        boxes, logits, phrases = predict(
            model=model, image=image, caption=TEXT_PROMPT,
            box_threshold=BOX_TH, text_threshold=TXT_TH)
        if boxes.shape[0] == 0:
            continue
        scaled = boxes * torch.tensor([W, H, W, H], dtype=boxes.dtype)
        xyxy = box_convert(scaled, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        for (x1, y1, x2, y2), sc in zip(xyxy, logits.numpy()):
            results.append({
                "image_id": int(img_id),
                "category_id": cat_id,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(sc),
            })

    print(f"예측 박스 수: {len(results)}")
    if not results:
        print("예측 박스 0개 - zero-shot이 아무것도 못 잡음 (이것도 유의미한 results)")
        print("=> fine-tuning 없이는 X-ray 임플란트를 탐지 못함을 의미")
        return

    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(copy.deepcopy(coco_gt), coco_dt, iouType="bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()

    print("\n" + "="*60)
    print("Zero-shot (fine-tuning 없음) results")
    print(f"  AP50 (mAP@.5)   = {ev.stats[1]:.3f}")
    print(f"  mAP@.5:.95      = {ev.stats[0]:.3f}")
    print(f"  AR@100          = {ev.stats[8]:.3f}")
    print("="*60)
    print("\n비교: fine-tuned(epoch_20)은 AP50=1.000, mAP@.5:.95=0.877, AR=0.887")
    print("     -> 두 값의 차이가 fine-tuning의 기여")


if __name__ == "__main__":
    main()
