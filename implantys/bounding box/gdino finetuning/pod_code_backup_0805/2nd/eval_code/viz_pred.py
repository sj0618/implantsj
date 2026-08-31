#!/usr/bin/env python3
"""
예측 박스 + GT 박스를 이미지에 그린다.

predictions.json + annotations.json + 이미지만 읽는다. 모델을 안 부르므로 GPU도
repo도 필요 없다.

  python eval_code/viz_pred.py --run results/00_zeroshot --threshold 0.30

threshold는 필수다. 예측을 몇 점에서 자르고 볼지에 따라 그림이 완전히 달라진다
(zero-shot 기준 0.05면 836개, 0.50이면 51개). 고르지 않고 정할 수 없다.

색:
  초록 실선   GT
  파랑        TP — GT와 IoU>=0.5로 매칭된 예측
  빨강        FP — 매칭 안 된 예측
  노랑 점선   FN — 놓친 GT

FN이 있는 이미지는 파일명에 _FN이 붙는다.
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path("/workspace/Grounding-Dino-FineTuning/2nd")

GT_C, TP_C, FP_C, FN_C = "#2ca02c", "#1f77b4", "#d62728", "#ffcc00"


def load_font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


def rect(d, box, color, width=3, dash=False):
    x, y, w, h = box
    if not dash:
        d.rectangle([x, y, x + w, y + h], outline=color, width=width)
        return
    step = 9
    for i in range(int(x), int(x + w), step * 2):
        d.line([i, y, min(i + step, x + w), y], fill=color, width=width)
        d.line([i, y + h, min(i + step, x + w), y + h], fill=color, width=width)
    for i in range(int(y), int(y + h), step * 2):
        d.line([x, i, x, min(i + step, y + h)], fill=color, width=width)
        d.line([x + w, i, x + w, min(i + step, y + h)], fill=color, width=width)


def banner(d, lines, font, pad=6):
    """좌상단에 검은 배경 텍스트 박스."""
    h = sum(font.getbbox(t)[3] - font.getbbox(t)[1] + 6 for t, _ in lines)
    w = max(font.getbbox(t)[2] for t, _ in lines)
    d.rectangle([0, 0, w + pad * 2, h + pad * 2], fill="#000000")
    y = pad
    for t, c in lines:
        d.text((pad, y), t, fill=c, font=font)
        y += font.getbbox(t)[3] - font.getbbox(t)[1] + 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="결과 폴더. 예: results/00_zeroshot")
    ap.add_argument("--threshold", type=float, required=True,
                    help="이 점수 미만 예측은 안 그린다. 필수.")
    ap.add_argument("--data-root", default=str(BASE / "dataset2"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="기본: <run>/viz/thr_<t>")
    a = ap.parse_args()

    run = Path(a.run)
    out = Path(a.out) if a.out else run / "viz" / f"thr_{a.threshold:.2f}"
    out.mkdir(parents=True, exist_ok=True)

    preds_p = run / "predictions.json"
    if not preds_p.exists():
        raise SystemExit(f"[ERROR] 없음: {preds_p} — evaluate.py를 먼저 돌릴 것")
    preds = json.loads(preds_p.read_text(encoding="utf-8"))

    split_dir = Path(a.data_root) / a.split
    coco = json.loads((split_dir / "annotations.json").read_text(encoding="utf-8"))
    id2name = {i["id"]: i["file_name"] for i in coco["images"]}
    gts = {i["id"]: [] for i in coco["images"]}
    for an in coco["annotations"]:
        gts[an["image_id"]].append(an["bbox"])

    by_img = {i: [] for i in id2name}
    kept = 0
    for p in preds:
        if p["score"] >= a.threshold:
            by_img[p["image_id"]].append(p)
            kept += 1

    print(f"[viz] {run.name} | thr {a.threshold:.2f} | "
          f"예측 {kept}/{len(preds)}개 | {len(id2name)}장")

    font = load_font(18)
    TP = FP = FN = 0
    fn_files = []

    for img_id, fname in sorted(id2name.items(), key=lambda x: x[1]):
        src = split_dir / "images" / fname
        if not src.exists():
            raise SystemExit(f"[ERROR] 이미지 없음: {src}")
        im = Image.open(src).convert("RGB")
        d = ImageDraw.Draw(im)

        g = gts[img_id]
        p = sorted(by_img[img_id], key=lambda r: -r["score"])

        for b in g:
            rect(d, b, GT_C, 3)

        matched = set()
        n_tp = 0
        for pr in p:
            best, bi = 0.0, -1
            for i, gb in enumerate(g):
                if i in matched:
                    continue
                v = iou_xywh(pr["bbox"], gb)
                if v > best:
                    best, bi = v, i
            ok = best >= a.iou
            if ok:
                matched.add(bi)
                n_tp += 1
            rect(d, pr["bbox"], TP_C if ok else FP_C, 2)
            x, y = pr["bbox"][0], pr["bbox"][1]
            d.text((x + 2, max(0, y - 20)), f"{pr['score']:.2f}",
                   fill=TP_C if ok else FP_C, font=font)

        fns = [gb for i, gb in enumerate(g) if i not in matched]
        for gb in fns:
            rect(d, gb, FN_C, 3, dash=True)

        TP += n_tp
        FP += len(p) - n_tp
        FN += len(fns)

        banner(d, [
            (f"GT {len(g)}  pred {len(p)}  TP {n_tp}  FP {len(p)-n_tp}  FN {len(fns)}",
             "#ffffff"),
            ("GT", GT_C), ("TP", TP_C), ("FP", FP_C), ("FN(dash)", FN_C),
        ], font)

        suffix = "_FN" if fns else ""
        im.save(out / f"{Path(fname).stem}{suffix}.jpg", quality=90)
        if fns:
            fn_files.append(f"{Path(fname).stem}{suffix}.jpg")

    print(f"  TP {TP} | FP {FP} | FN {FN}")
    print(f"  {len(id2name)}장 저장 -> {out}")
    if fn_files:
        print(f"  FN 있는 이미지 {len(fn_files)}장 (_FN 접미사):")
        for f in fn_files:
            print(f"    {f}")


if __name__ == "__main__":
    main()
