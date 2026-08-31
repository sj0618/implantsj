#!/usr/bin/env python3
"""
학습된 GroundingDINO(LoRA) 체크포인트를 test 셋에서 COCO mAP로 정량 평가.
여러 체크포인트를 한 번에 비교(epoch_20 vs epoch_25 등).

정답: dataset_coco/test/annotations.json (COCO 형식, 사람 라벨)
예측: repo의 predict()로 test 이미지에 박스 -> COCO 형식으로 변환 -> pycocotools 비교

실행 (repo 루트에서):
  export LD_LIBRARY_PATH=$(python -c "import torch,os;print(os.path.dirname(torch.__file__))")/lib:$LD_LIBRARY_PATH
  python eval_map.py

출력: 각 체크포인트별 mAP@0.5, mAP@0.5:0.95, AR(recall)
"""
import os, json, copy, argparse
import numpy as np
import torch
from torchvision.ops import box_convert

from groundingdino.util.inference import load_model, load_image, predict
from config import ConfigurationManager
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---- 설정 ----
TEST_IMG_DIR = "multimodal-data/implant_dataset/images/test"
TEST_COCO_GT = "/workspace/dataset_coco/test/annotations.json"
BASE_WEIGHTS = "weights/groundingdino_swint_ogc.pth"
MODEL_CFG    = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
TEXT_PROMPT  = "implant"
BOX_TH       = 0.25   # 평가용은 낮게 잡아 recall 확보 (mAP는 여러 threshold를 내부 처리)
TXT_TH       = 0.20

# 비교할 체크포인트들
CKPTS = [
    "weights/20260703_0156/checkpoint_epoch_20.pth",
    "weights/20260703_0156/checkpoint_epoch_25.pth",
]


class MC:
    """load_model이 기대하는 model_config 흉내 (config.py 없이 간단히)"""
    def __init__(self, lora_weights):
        self.config_path = MODEL_CFG
        self.weights_path = BASE_WEIGHTS
        self.lora_weights = lora_weights
        self.lora_weigths = lora_weights  # repo 오타 대비 (양쪽 다 세팅)


def build_gt_index(coco_gt):
    """GT의 file_name -> image_id, (w,h) 매핑"""
    name2id, id2wh = {}, {}
    for im in coco_gt.dataset["images"]:
        name2id[im["file_name"]] = im["id"]
        id2wh[im["id"]] = (im["width"], im["height"])
    return name2id, id2wh


def predict_to_coco(model, name2id, id2wh):
    """test 이미지 전부 예측 -> COCO detection 결과 리스트"""
    results = []
    files = [f for f in os.listdir(TEST_IMG_DIR)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for fn in files:
        if fn not in name2id:
            print(f"  [warn] GT에 없는 이미지: {fn}")
            continue
        img_id = name2id[fn]
        W, H = id2wh[img_id]

        image_source, image = load_image(os.path.join(TEST_IMG_DIR, fn))
        boxes, logits, phrases = predict(
            model=model, image=image, caption=TEXT_PROMPT,
            box_threshold=BOX_TH, text_threshold=TXT_TH)

        if boxes.shape[0] == 0:
            continue
        # 정규화 cxcywh -> 픽셀 xyxy -> COCO xywh
        scaled = boxes * torch.tensor([W, H, W, H], dtype=boxes.dtype)
        xyxy = box_convert(scaled, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        for (x1, y1, x2, y2), sc in zip(xyxy, logits.numpy()):
            results.append({
                "image_id": int(img_id),
                "category_id": 1,   # implant (GT의 category_id와 일치해야 함)
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(sc),
            })
    return results


def evaluate(coco_gt, results):
    if not results:
        print("  예측 박스 0개 - 평가 불가")
        return
    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev.stats  # [mAP@.5:.95, mAP@.5, mAP@.75, ... AR ...]


def main():
    coco_gt = COCO(TEST_COCO_GT)
    # GT category_id 확인 (implant가 1이 아닐 수 있음)
    cats = coco_gt.dataset["categories"]
    print(f"GT categories: {cats}")
    name2id, id2wh = build_gt_index(coco_gt)

    summary = {}
    for ckpt in CKPTS:
        if not os.path.exists(ckpt):
            print(f"[skip] 체크포인트 없음: {ckpt}")
            continue
        print(f"\n{'='*60}\n[eval] {ckpt}\n{'='*60}")
        model = load_model(MC(ckpt), use_lora=True)
        results = predict_to_coco(model, name2id, id2wh)
        print(f"  예측 박스 수: {len(results)}")
        stats = evaluate(copy.deepcopy(coco_gt), results)
        if stats is not None:
            summary[ckpt] = {"mAP@.5:.95": stats[0], "mAP@.5": stats[1],
                             "AR@100": stats[8]}
        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*60}\n요약 비교\n{'='*60}")
    for ckpt, s in summary.items():
        name = os.path.basename(ckpt)
        print(f"{name:30s}  mAP@.5={s['mAP@.5']:.3f}  "
              f"mAP@.5:.95={s['mAP@.5:.95']:.3f}  AR@100={s['AR@100']:.3f}")


if __name__ == "__main__":
    main()
