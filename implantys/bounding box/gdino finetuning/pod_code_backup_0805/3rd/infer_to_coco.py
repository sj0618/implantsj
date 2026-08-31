#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_to_coco.py
FT-1000(LoRA) 으로 제조사별 폴더의 이미지에 implant 박스를 예측하고,
각 폴더에 COCO annotation.json 을 만든다. (모델 1회 로드 -> 전체 추론 -> 폴더별 저장)

infer_to_labelstudio.py 의 모델로딩/추론/NMS 로직 그대로, 출력만 COCO(픽셀 xywh).

사용:
  python infer_to_coco.py --lora-weights 3rd/results/FT-1000/weights/epoch_09.pth \
      --root "3rd/dataset/7+3 label_0721" \
      --box-threshold 0.30 --text-threshold 0.20 --nms-iou 0.5
"""
import argparse, json, os, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--config", default="groundingdino/config/GroundingDINO_SwinT_OGC.py")
    ap.add_argument("--base-weights", default="weights/groundingdino_swint_ogc.pth")
    ap.add_argument("--lora-weights", required=True)
    ap.add_argument("--root", required=True, help="제조사 폴더들이 든 상위 폴더")
    ap.add_argument("--prompt", default="implant")
    ap.add_argument("--label", default="implant")
    ap.add_argument("--box-threshold", type=float, default=0.30)
    ap.add_argument("--text-threshold", type=float, default=0.20)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--out-name", default="annotation.json")
    ap.add_argument("--only", default=None, help="특정 폴더만. 콤마로 여러 개: BILIM,IMPLANCE")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo_root)
    root = os.path.abspath(args.root)
    sys.path.insert(0, repo)
    os.chdir(repo)

    import torch
    from torchvision.ops import box_convert
    import torchvision.ops as ops
    from groundingdino.util.inference import load_model, load_image, predict
    from config import ModelConfig

    mc = ModelConfig(config_path=args.config, weights_path=args.base_weights,
                     lora_weigths=args.lora_weights)  # repo 오타 필드명 그대로
    print("[로드] base + LoRA 병합 중...", flush=True)
    model = load_model(mc, use_lora=True); model.eval()

    def nms(boxes, logits, phrases, w, h, thr):
        if boxes.shape[0] == 0:
            return boxes, logits, phrases
        scaled = box_convert(boxes * torch.tensor([w, h, w, h], dtype=boxes.dtype),
                             in_fmt="cxcywh", out_fmt="xyxy")
        kb, kl, kp = [], [], []
        for ph in set(phrases):
            ids = [i for i, p in enumerate(phrases) if p == ph]
            for k in ops.nms(scaled[ids], logits[ids], thr):
                kb.append(boxes[ids][k]); kl.append(logits[ids][k]); kp.append(ph)
        return torch.stack(kb), torch.stack(kl), kp

    EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    subdirs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        subdirs = [d for d in subdirs if d in want]
    g_img = g_box = 0
    for sd in subdirs:
        sdir = os.path.join(root, sd)
        # 폴더 직속 또는 하위(images/) 어디에 있든 재귀 탐색. (basename, 전체경로)
        found = []
        for r, _, fs in os.walk(sdir):
            for f in fs:
                if f.lower().endswith(EXT):
                    found.append((f, os.path.join(r, f)))
        found.sort(key=lambda t: t[0])
        if not found:
            print(f"[skip] {sd}: 이미지 0", flush=True); continue
        images, anns, aid = [], [], 1
        for iid, (fn, fpath) in enumerate(found, 1):
            src, image = load_image(fpath)
            h, w = src.shape[:2]
            boxes, logits, phrases = predict(model=model, image=image, caption=args.prompt,
                                             box_threshold=args.box_threshold,
                                             text_threshold=args.text_threshold)
            if boxes.shape[0] > 0:
                boxes, logits, phrases = nms(boxes, logits, phrases, w, h, args.nms_iou)
            images.append({"width": int(w), "height": int(h), "id": iid, "file_name": fn})
            for box, score in zip(boxes.tolist(), logits.tolist()):
                cx, cy, bw, bh = box
                x = max(0.0, (cx - bw / 2) * w); y = max(0.0, (cy - bh / 2) * h)
                ww, hh = bw * w, bh * h
                anns.append({"id": aid, "image_id": iid, "category_id": 0, "segmentation": [],
                             "bbox": [x, y, ww, hh], "ignore": 0, "iscrowd": 0,
                             "area": ww * hh, "score": round(float(score), 4)})
                aid += 1
        coco = {"images": images, "annotations": anns,
                "categories": [{"id": 0, "name": args.label}]}
        json.dump(coco, open(os.path.join(sdir, args.out_name), "w", encoding="utf-8"),
                  ensure_ascii=False)
        print(f"[{sd}] img={len(images)} box={len(anns)} -> {args.out_name}", flush=True)
        g_img += len(images); g_box += len(anns)
    print(f"[done] 폴더 {len(subdirs)} / 총 img={g_img} box={g_box}", flush=True)


if __name__ == "__main__":
    main()
