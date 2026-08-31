#!/usr/bin/env python3
"""
best epoch 체크포인트로 val을 추론해 박스를 그린다. (RULES 4의 viz/)

RULES 2: "val 예측 시각화도 best epoch일 때만 남긴다." — 그래서 이 스크립트는
best_epoch.json이 있어야만 돈다. 학습이 끝나기 전에는 못 돌린다.

  python train_code/viz_val.py --run results/FT-106_lr1e-4 --threshold 0.30

run/data/val_annotations.csv + images/val을 읽는다. 학습에 쓴 바로 그 파일이라
어긋날 수 없다.

색은 eval_code/viz_pred.py와 동일:
  초록 GT / 파랑 TP / 빨강 FP / 노랑 점선 FN
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path("/workspace/Grounding-Dino-FineTuning")
sys.path.insert(0, str(REPO / "2nd" / "eval_code"))

from viz_pred import GT_C, TP_C, FP_C, FN_C, iou_xywh, rect, banner, load_font  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

TEXT_PROMPT = "implant"
LOW_THR = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--threshold", type=float, required=True,
                    help="이 점수 미만은 안 그린다. 필수.")
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args()

    import torch
    from torchvision.ops import box_convert
    from groundingdino.util.train import load_model
    from groundingdino.util.inference import load_image, preprocess_caption
    from peft import set_peft_model_state_dict

    run = Path(a.run)
    bp = run / "best_epoch.json"
    if not bp.exists():
        sys.exit(f"[ERROR] 없음: {bp}\n"
                 "  RULES 2: val 시각화는 best epoch일 때만 남긴다.\n"
                 "  30 epoch가 끝나야 best_epoch.json이 생긴다.")
    best = json.loads(bp.read_text(encoding="utf-8"))
    ckpt = Path(best["checkpoint"])
    if not ckpt.exists():
        sys.exit(f"[ERROR] 체크포인트 없음: {ckpt}")

    va_csv = run / "data" / "val_annotations.csv"
    va_img = run / "data" / "images" / "val"
    rows = list(csv.DictReader(open(va_csv, encoding="utf-8")))
    gts = {}
    for r in rows:
        gts.setdefault(r["image_name"], []).append(
            [float(r["bbox_x"]), float(r["bbox_y"]),
             float(r["bbox_width"]), float(r["bbox_height"])])
    names = sorted({r["image_name"] for r in rows})

    print(f"[viz-val] {run.name} | best epoch {best['best_epoch']} "
          f"(val mAP {best['selected']['val_map']:.4f}) | thr {a.threshold:.2f}")
    print(f"  ckpt {ckpt.name} | val {len(names)}장 / {len(rows)}박스")

    model = load_model(str(REPO / "groundingdino/config/GroundingDINO_SwinT_OGC.py"),
                       str(REPO / "weights/groundingdino_swint_ogc.pth"),
                       use_lora=True)
    sd = torch.load(ckpt, map_location="cpu")
    set_peft_model_state_dict(model, sd["model"])
    model.eval().cuda()

    out = run / "viz"
    out.mkdir(exist_ok=True)
    font = load_font(18)
    caption = preprocess_caption(TEXT_PROMPT)

    TP = FP = FN = 0
    fn_files = []
    with torch.no_grad():
        for name in names:
            src = va_img / name
            _, image = load_image(str(src))
            o = model(image[None].cuda(), captions=[caption])
            logits = o["pred_logits"].sigmoid()[0]
            boxes = o["pred_boxes"][0]
            scores = logits.max(dim=1)[0]
            keep = scores > LOW_THR

            im = Image.open(src).convert("RGB")
            W, H = im.size
            d = ImageDraw.Draw(im)

            preds = []
            if keep.sum() > 0:
                b = boxes[keep].cpu()
                s = scores[keep].cpu().numpy()
                xyxy = box_convert(b * torch.tensor([W, H, W, H], dtype=b.dtype),
                                   in_fmt="cxcywh", out_fmt="xyxy").numpy()
                for (x1, y1, x2, y2), sc in zip(xyxy, s):
                    if sc >= a.threshold:
                        preds.append({"bbox": [float(x1), float(y1),
                                               float(x2 - x1), float(y2 - y1)],
                                      "score": float(sc)})
            preds.sort(key=lambda r: -r["score"])

            g = gts[name]
            for gb in g:
                rect(d, gb, GT_C, 3)

            matched = set()
            n_tp = 0
            for pr in preds:
                bi, bv = -1, 0.0
                for i, gb in enumerate(g):
                    if i in matched:
                        continue
                    v = iou_xywh(pr["bbox"], gb)
                    if v > bv:
                        bv, bi = v, i
                ok = bv >= a.iou
                if ok:
                    matched.add(bi)
                    n_tp += 1
                rect(d, pr["bbox"], TP_C if ok else FP_C, 2)
                d.text((pr["bbox"][0] + 2, max(0, pr["bbox"][1] - 20)),
                       f"{pr['score']:.2f}", fill=TP_C if ok else FP_C, font=font)

            fns = [gb for i, gb in enumerate(g) if i not in matched]
            for gb in fns:
                rect(d, gb, FN_C, 3, dash=True)

            TP += n_tp
            FP += len(preds) - n_tp
            FN += len(fns)

            banner(d, [
                (f"ep{best['best_epoch']}  GT {len(g)}  pred {len(preds)}  "
                 f"TP {n_tp}  FP {len(preds)-n_tp}  FN {len(fns)}", "#ffffff"),
                ("GT", GT_C), ("TP", TP_C), ("FP", FP_C), ("FN(dash)", FN_C),
            ], font)

            suffix = "_FN" if fns else ""
            im.save(out / f"{Path(name).stem}{suffix}.jpg", quality=90)
            if fns:
                fn_files.append(f"{Path(name).stem}{suffix}.jpg")

    print(f"  TP {TP} | FP {FP} | FN {FN}")
    print(f"  {len(names)}장 저장 -> {out}")
    if fn_files:
        print(f"  FN 있는 이미지 {len(fn_files)}장 — attention 분석 대상 (RULES 7)")
        for f_ in fn_files:
            print(f"    {f_}")


if __name__ == "__main__":
    main()
