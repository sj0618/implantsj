#!/usr/bin/env python3
"""
Occlusion sensitivity — "어디가 판단을 결정하는가". (4th_plan ⑤)

attention은 모델이 **어디를 참조했는가**(기술)를 보여줄 뿐, 그것이 판단을 **결정했다**는
보장은 없다. 여기서는 실제로 가려보고 score가 무너지는지를 잰다(개입).

방법: 회색 패치를 격자로 옮기며 매 위치마다 재추론 → 대상 박스의 score 하락량을 기록.
      부위를 사람이 미리 정의하지 않으므로 자의성이 없고, 결과가 attention과 같은
      히트맵 형태라 나란히 비교할 수 있다.

출력: 케이스당 3분할 1장 — [원본+박스 | attention | occlusion 히트맵]

  python occlusion_4th.py --cases <...>/attention/cases.json \
      --images-dir <...>/test200/images --gt <...>/test200/annotations.json \
      --attention-dir <...>/attention/FT-1000 --weights <...>/epoch_09.pth \
      --out <...>/results/occlusion --n-fn 6 --n-tp 4
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO = Path("/workspace/Grounding-Dino-FineTuning")
TEXT_PROMPT = "implant"
TEXT_THR = 0.20


def iou_xywh(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--attention-dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-fn", type=int, default=6)
    ap.add_argument("--n-tp", type=int, default=4)
    ap.add_argument("--grid", type=int, default=14, help="한 변당 패치 위치 수")
    ap.add_argument("--patch-frac", type=float, default=0.12,
                    help="패치 크기 = 짧은 변 × 이 비율")
    a = ap.parse_args()

    from groundingdino.util.train import load_model
    from groundingdino.util.inference import load_image, preprocess_caption
    from groundingdino.util.misc import clean_state_dict  # noqa: F401  (있으면 사용)
    import groundingdino.datasets.transforms as T
    from torchvision.ops import box_convert
    from peft import set_peft_model_state_dict

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    img_dir = Path(a.images_dir)
    att_dir = Path(a.attention_dir)

    cases = json.loads(Path(a.cases).read_text())
    targets = [(n, "fn") for n in cases["fn"][:a.n_fn]] + \
              [(n, "tp") for n in cases["tp"][:a.n_tp]]

    gt = json.loads(Path(a.gt).read_text())
    name2id = {im["file_name"]: im["id"] for im in gt["images"]}
    gts = {}
    for an in gt["annotations"]:
        gts.setdefault(an["image_id"], []).append(an["bbox"])

    model = load_model(str(REPO / "groundingdino/config/GroundingDINO_SwinT_OGC.py"),
                       str(REPO / "weights/groundingdino_swint_ogc.pth"), use_lora=True)
    sd = torch.load(a.weights, map_location="cpu")
    set_peft_model_state_dict(model, sd["model"])
    model.eval().cuda()
    caption = preprocess_caption(TEXT_PROMPT)

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    @torch.no_grad()
    def score_for(pil, target_xywh):
        """target 박스와 IoU>=0.3으로 겹치는 예측 중 최고 score. 없으면 0."""
        img, _ = transform(pil, None)
        o = model(img[None].cuda(), captions=[caption])
        logits = o["pred_logits"].sigmoid()[0]          # (nq, ntxt)
        boxes = o["pred_boxes"][0]                      # (nq, 4) cxcywh 정규화
        sc = logits.max(dim=1)[0]
        keep = sc > 0.03
        if keep.sum() == 0:
            return 0.0
        W, H = pil.size
        b = boxes[keep].cpu() * torch.tensor([W, H, W, H], dtype=boxes.dtype)
        xyxy = box_convert(b, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        s = sc[keep].cpu().numpy()
        best = 0.0
        for (x1, y1, x2, y2), v in zip(xyxy, s):
            if iou_xywh([x1, y1, x2 - x1, y2 - y1], target_xywh) >= 0.3:
                best = max(best, float(v))
        return best

    meta = {}
    for name, kind in targets:
        src = img_dir / name
        if not src.exists():
            print(f"[skip] 이미지 없음 {name}"); continue
        pil = Image.open(src).convert("RGB")
        W, H = pil.size
        boxes = gts.get(name2id[name], [])
        if not boxes:
            print(f"[skip] GT 없음 {name}"); continue
        # 대상: 가장 큰 GT 박스 (FN이면 그 이미지에서 문제되는 임플란트일 가능성 높음)
        target = max(boxes, key=lambda b: b[2] * b[3])

        base = score_for(pil, target)
        p = max(8, int(min(W, H) * a.patch_frac))
        xs = np.linspace(0, max(0, W - p), a.grid).astype(int)
        ys = np.linspace(0, max(0, H - p), a.grid).astype(int)
        heat = np.zeros((len(ys), len(xs)), dtype=float)

        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                occ = pil.copy()
                occ.paste((128, 128, 128), (int(x), int(y), int(x + p), int(y + p)))
                heat[iy, ix] = base - score_for(occ, target)
        print(f"  {name} [{kind}] base score {base:.3f} | "
              f"최대 하락 {heat.max():.3f} @ 격자 {a.grid}x{a.grid}, 패치 {p}px")

        # ── 3분할 그림 ──────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
        axes[0].imshow(pil)
        for b in boxes:
            axes[0].add_patch(mpatches.Rectangle((b[0], b[1]), b[2], b[3], fill=False,
                                                 edgecolor="#2ca02c", lw=2))
        axes[0].add_patch(mpatches.Rectangle((target[0], target[1]), target[2], target[3],
                                             fill=False, edgecolor="#ff7f0e", lw=3))
        axes[0].set_title(f"{name[:28]}… [{kind.upper()}]\nGT(green) / target(orange) | "
                          f"base score {base:.3f}", fontsize=9)

        ap_img = att_dir / f"{Path(name).stem}.jpg"
        if ap_img.exists():
            axes[1].imshow(Image.open(ap_img))
            axes[1].set_title("attention (where it looked)", fontsize=10)
        else:
            axes[1].text(0.5, 0.5, "attention 없음", ha="center")

        axes[2].imshow(pil)
        hm = np.array(Image.fromarray(heat).resize((W, H), Image.BILINEAR))
        im2 = axes[2].imshow(hm, cmap="hot", alpha=0.55)
        axes[2].add_patch(mpatches.Rectangle((target[0], target[1]), target[2], target[3],
                                             fill=False, edgecolor="#00BFFF", lw=2))
        axes[2].set_title("occlusion sensitivity (what decided it)\n"
                          "brighter = occluding here drops the score more", fontsize=10)
        fig.colorbar(im2, ax=axes[2], fraction=0.035)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / f"{kind}_{Path(name).stem}.jpg", dpi=120, bbox_inches="tight")
        plt.close(fig)

        meta[name] = {"kind": kind, "base_score": base, "max_drop": float(heat.max()),
                      "patch_px": p, "grid": a.grid,
                      "target_bbox": [float(v) for v in target]}

    (out / "occlusion_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(meta)}케이스 -> {out}")


if __name__ == "__main__":
    main()
