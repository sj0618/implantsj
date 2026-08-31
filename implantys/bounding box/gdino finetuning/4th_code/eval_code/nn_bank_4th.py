#!/usr/bin/env python3
"""
사례 기반 설명 — query embedding 최근접 이웃. (4th_plan ⑥)

attention/occlusion이 "이미지의 어디를 봤나"를 답한다면, 이것은
**"학습 데이터의 무엇과 닮아서 그렇게 판단했나"**를 답한다 (this looks like that).

원리: Grounding DINO decoder의 마지막 layer hidden state `hs[-1]`(query당 256d)가
      곧 "이 박스를 임플란트라고 본" 내부 표현이다. score 자체가 이 벡터와 텍스트
      토큰의 내적이므로, 판단의 근거 표현으로 삼기에 적절하다.

2단계:
  build : 학습 이미지 추론 -> GT와 매칭된 예측의 embedding을 뱅크에 저장
  query : test 예측의 embedding으로 뱅크 k-NN -> [쿼리 crop | 이웃 top-k crop] 패널

  python nn_bank_4th.py build --images-dir <train images> --csv <train_annotations.csv> \
      --weights <...>/epoch_09.pth --out <...>/results/nn
  python nn_bank_4th.py query --bank <...>/nn/bank.npz --images-dir <test200 images> \
      --predictions <...>/test200/FT-1000/predictions.json --gt <...>/annotations.json \
      --weights <...>/epoch_09.pth --out <...>/results/nn --n-panels 50
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO = Path("/workspace/Grounding-Dino-FineTuning")
TEXT_PROMPT = "implant"
MATCH_IOU = 0.5
PAD = 0.15                # crop 여백 비율


def iou_xywh(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


def crop(pil, box, pad=PAD):
    x, y, w, h = box
    dx, dy = w * pad, h * pad
    return pil.crop((max(0, int(x - dx)), max(0, int(y - dy)),
                     min(pil.width, int(x + w + dx)), min(pil.height, int(y + h + dy))))


class Runner:
    """모델 1회 로드 + hs(decoder hidden state) 후킹."""

    def __init__(self, weights):
        from groundingdino.util.train import load_model
        from groundingdino.util.inference import preprocess_caption
        import groundingdino.datasets.transforms as T
        from peft import set_peft_model_state_dict

        self.model = load_model(
            str(REPO / "groundingdino/config/GroundingDINO_SwinT_OGC.py"),
            str(REPO / "weights/groundingdino_swint_ogc.pth"), use_lora=True)
        sd = torch.load(weights, map_location="cpu")
        set_peft_model_state_dict(self.model, sd["model"])
        self.model.eval().cuda()
        self.caption = preprocess_caption(TEXT_PROMPT)
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        # transformer의 출력 튜플에서 hs를 잡는다 (hs, reference, hs_enc, ref_enc, init_box)
        mods = dict(self.model.named_modules())
        cand = [n for n in mods if n.endswith("transformer")]
        if not cand:
            raise SystemExit("[ERROR] transformer 모듈을 못 찾음")
        self.path = min(cand, key=len)
        self.store = {}

        def hook(_m, _i, o):
            self.store["hs"] = o[0]          # (n_dec_layer, bs, nq, d)
        self.h = mods[self.path].register_forward_hook(hook)
        print(f"[nn] hs 후킹 지점: {self.path}")

    @torch.no_grad()
    def run(self, pil):
        from torchvision.ops import box_convert
        img, _ = self.transform(pil, None)
        o = self.model(img[None].cuda(), captions=[self.caption])
        hs = self.store.get("hs")
        if hs is None:
            raise SystemExit("[ERROR] hs를 못 잡았다")
        emb = hs[-1][0]                                    # (nq, d) 마지막 decoder layer
        sc = o["pred_logits"].sigmoid()[0].max(dim=1)[0]   # (nq,)
        W, H = pil.size
        b = o["pred_boxes"][0].cpu() * torch.tensor([W, H, W, H])
        xyxy = box_convert(b, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        xywh = np.stack([xyxy[:, 0], xyxy[:, 1],
                         xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], 1)
        return emb.cpu().numpy(), sc.cpu().numpy(), xywh


def cmd_build(a):
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    gts = defaultdict(list)
    for r in rows:
        gts[r["image_name"]].append([float(r["bbox_x"]), float(r["bbox_y"]),
                                     float(r["bbox_width"]), float(r["bbox_height"])])
    names = sorted(gts)
    print(f"[build] 학습 이미지 {len(names)}장 / GT {len(rows)}박스")

    R = Runner(a.weights)
    E, meta = [], []
    for i, name in enumerate(names, 1):
        p = Path(a.images_dir) / name
        if not p.exists():
            continue
        pil = Image.open(p).convert("RGB")
        emb, sc, xywh = R.run(pil)
        used = set()
        for q in np.argsort(-sc):
            if sc[q] < a.min_score:
                break
            best, bi = 0.0, -1
            for gi, g in enumerate(gts[name]):
                if gi in used:
                    continue
                v = iou_xywh(xywh[q], g)
                if v > best:
                    best, bi = v, gi
            if best >= MATCH_IOU:
                used.add(bi)
                E.append(emb[q])
                meta.append({"image": name, "bbox": [float(v) for v in xywh[q]],
                             "score": float(sc[q]), "iou": float(best)})
        if i % 100 == 0:
            print(f"  {i}/{len(names)} · 뱅크 {len(E)}개", flush=True)

    E = np.stack(E).astype(np.float32)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "bank.npz", emb=E,
                        meta=np.array(json.dumps(meta, ensure_ascii=False)))
    print(f"[build] 뱅크 {E.shape} -> {out / 'bank.npz'}")
    print(f"  (GT와 IoU>={MATCH_IOU}로 매칭된 예측만 저장 = '맞게 잡은' 사례)")


def cmd_query(a):
    z = np.load(a.bank, allow_pickle=True)
    bank = z["emb"].astype(np.float32)
    bmeta = json.loads(str(z["meta"]))
    bn = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-8)
    print(f"[query] 뱅크 {bank.shape}")

    preds = json.loads(Path(a.predictions).read_text())
    gt = json.loads(Path(a.gt).read_text())
    id2name = {im["id"]: im["file_name"] for im in gt["images"]}
    gts = defaultdict(list)
    for an in gt["annotations"]:
        gts[an["image_id"]].append(an["bbox"])

    by_img = defaultdict(list)
    for p in preds:
        if p["score"] >= a.thr:
            by_img[p["image_id"]].append(p)

    R = Runner(a.weights)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "panels").mkdir(exist_ok=True)
    rows, made = [], 0

    for img_id in sorted(by_img):
        if made >= a.n_panels:
            break
        name = id2name[img_id]
        p = Path(a.images_dir) / name
        if not p.exists():
            continue
        pil = Image.open(p).convert("RGB")
        emb, sc, xywh = R.run(pil)

        for pr in sorted(by_img[img_id], key=lambda r: -r["score"])[:a.per_image]:
            if made >= a.n_panels:
                break
            q = int(np.argmax([iou_xywh(xywh[i], pr["bbox"]) for i in range(len(xywh))]))
            if iou_xywh(xywh[q], pr["bbox"]) < 0.8:
                continue                       # 예측-쿼리 대응이 불확실하면 건너뜀
            v = emb[q] / (np.linalg.norm(emb[q]) + 1e-8)
            sim = bn @ v
            top = np.argsort(-sim)[:a.k]

            hit = max((iou_xywh(pr["bbox"], g) for g in gts[img_id]), default=0.0)
            tag = "TP" if hit >= MATCH_IOU else "FP"

            qc = crop(pil, pr["bbox"]).resize((160, 160))
            tiles = [qc]
            labels = [f"QUERY {tag} {pr['score']:.2f}"]
            for t in top:
                m = bmeta[int(t)]
                tp = Path(a.train_images_dir) / m["image"]
                if not tp.exists():
                    continue
                tiles.append(crop(Image.open(tp).convert("RGB"), m["bbox"]).resize((160, 160)))
                labels.append(f"{m['image'][:10]} {sim[t]:.3f}")
                rows.append({"query_image": name, "query_score": round(pr["score"], 3),
                             "query_tag": tag, "neighbor_image": m["image"],
                             "cos_sim": round(float(sim[t]), 4)})

            Wp = 160 * len(tiles) + 10 * (len(tiles) - 1)
            canvas = Image.new("RGB", (Wp, 190), "white")
            d = ImageDraw.Draw(canvas)
            x = 0
            for til, lab in zip(tiles, labels):
                canvas.paste(til, (x, 24))
                d.text((x + 3, 8), lab, fill="black")
                x += 170
            d.line([(165, 20), (165, 188)], fill="#C00000", width=2)
            canvas.save(out / "panels" / f"{tag.lower()}_{Path(name).stem[:24]}_{made}.jpg",
                        quality=92)
            made += 1

    with open(out / "neighbors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"[query] 패널 {made}장 -> {out / 'panels'}")
    print(f"        이웃 메타 {len(rows)}행 -> {out / 'neighbors.csv'}")
    print("  왼쪽이 test 예측(빨간 선 오른쪽이 학습 사례). "
          "'이 학습 사례들과 닮아서 임플란트라고 봤다'로 읽는다.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--images-dir", required=True)
    b.add_argument("--csv", required=True)
    b.add_argument("--weights", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--min-score", type=float, default=0.30)
    b.set_defaults(func=cmd_build)

    q = sub.add_parser("query")
    q.add_argument("--bank", required=True)
    q.add_argument("--images-dir", required=True)
    q.add_argument("--train-images-dir", required=True)
    q.add_argument("--predictions", required=True)
    q.add_argument("--gt", required=True)
    q.add_argument("--weights", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--thr", type=float, default=0.30)
    q.add_argument("--k", type=int, default=5)
    q.add_argument("--n-panels", type=int, default=50)
    q.add_argument("--per-image", type=int, default=1)
    q.set_defaults(func=cmd_query)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
