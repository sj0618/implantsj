#!/usr/bin/env python3
"""
텍스트 토큰 "implant" ↔ 이미지 feature 사이의 cross-attention. (RULES 7)

어디서 뽑는가
  encoder의 fusion_layers[i].attn (BiMultiHeadAttention). fuse_modules.py:174가
      attn_weights = bmm(query_states, key_states.T)   # bs*nhead, nimg, ntxt
  를 만든다. 이게 이미지 feature × 텍스트 토큰이라 규칙 7의 정의와 정확히 일치한다.
  decoder의 ca_text는 object query ↔ 텍스트라 다른 것이다. 쓰지 않는다.

어떻게 뽑는가
  BiAttentionBlock.forward가 layer_norm_v/l를 먼저 걸고 attn을 부르므로,
  attn에 forward_pre_hook을 걸면 forward가 실제로 쓰는 v, l을 그대로 잡는다.
  잡은 값으로 module 자신의 projection을 써서 attn_weights를 재계산한다.
  (forward 반환값은 delta_v/delta_l이라 attention weight가 안 나온다)
  clamp·stable softmax까지 fuse_modules.py의 순서를 그대로 따라간다.

"implant" 토큰 위치
  bert-base-uncased로 "implant." -> ['[CLS]','implant','.','[SEP]'] 이므로 index 1.
  단일 토큰이라 subword 평균이 필요 없다. 실행 시 실제로 검증한다.

  # zero-shot
  python eval_code/attention_map.py --out results/00_zeroshot/attention \
      --images-dir 2nd/dataset2/test/images --images a.jpg b.jpg

  # fine-tuned (동일 샘플로 비교. RULES 7)
  python eval_code/attention_map.py --out results/FT-106_lr1e-4/attention \
      --weights results/FT-106_lr1e-4/weights/epoch_18.pth \
      --images-dir 2nd/dataset2/test/images --images a.jpg b.jpg

RULES 7의 판정표 (사람이 본다):
  attention은 갔는데 점수가 낮음  -> 점수 보정 문제 -> threshold·score head
  attention이 아예 안 감          -> 표현 학습 문제 -> 학습 데이터 확대
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = Path("/workspace/Grounding-Dino-FineTuning")

TEXT_PROMPT = "implant"
LOW_THR = 0.05


def find_modules(model, class_name):
    return [m for _, m in model.named_modules()
            if type(m).__name__ == class_name]


class Grabber:
    """fusion layer의 attn_weights_l과 encoder의 spatial_shapes를 잡는다."""

    def __init__(self, model):
        self.v_l = []          # 레이어별 (v, l)
        self.spatial_shapes = None
        self.handles = []

        fusions = find_modules(model, "BiMultiHeadAttention")
        if not fusions:
            sys.exit("[ERROR] BiMultiHeadAttention을 못 찾았다 — repo 구조가 바뀌었다")
        self.fusions = fusions

        for m in fusions:
            self.handles.append(
                m.register_forward_pre_hook(self._grab_vl, with_kwargs=True))

        encs = find_modules(model, "TransformerEncoder")
        if not encs:
            sys.exit("[ERROR] TransformerEncoder를 못 찾았다")
        self.handles.append(
            encs[0].register_forward_pre_hook(self._grab_shapes, with_kwargs=True))

    def _grab_vl(self, module, args, kwargs):
        v = kwargs.get("v", args[0] if len(args) > 0 else None)
        l = kwargs.get("l", args[1] if len(args) > 1 else None)
        mask_v = kwargs.get("attention_mask_v")
        self.v_l.append((module, v.detach(), l.detach(),
                         None if mask_v is None else mask_v.detach()))
        return None

    def _grab_shapes(self, module, args, kwargs):
        ss = kwargs.get("spatial_shapes")
        if ss is None:
            for x in args:
                if torch.is_tensor(x) and x.dtype == torch.long and x.ndim == 2 \
                        and x.shape[1] == 2:
                    ss = x
                    break
        if ss is not None:
            self.spatial_shapes = ss.detach().cpu()
        return None

    def clear(self):
        self.v_l = []

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def attn_weights_l(module, v, l, mask_v):
    """fuse_modules.py의 forward를 그대로 재현. 반환: (nhead, ntxt, nimg) 확률분포."""
    bsz, tgt_len, _ = v.size()
    q = module.v_proj(v) * module.scale
    k = module._shape(module.l_proj(l), -1, bsz)
    proj_shape = (bsz * module.num_heads, -1, module.head_dim)
    q = module._shape(q, tgt_len, bsz).view(*proj_shape)
    k = k.view(*proj_shape)

    a = torch.bmm(q, k.transpose(1, 2))            # bs*nhead, nimg, ntxt
    if module.stable_softmax_2d:
        a = a - a.max()
    if module.clamp_min_for_underflow:
        a = torch.clamp(a, min=-50000)
    if module.clamp_max_for_overflow:
        a = torch.clamp(a, max=50000)

    aT = a.transpose(1, 2)                          # bs*nhead, ntxt, nimg
    al = aT - torch.max(aT, dim=-1, keepdim=True)[0]
    if module.clamp_min_for_underflow:
        al = torch.clamp(al, min=-50000)
    if module.clamp_max_for_overflow:
        al = torch.clamp(al, max=50000)
    if mask_v is not None:
        al = al.masked_fill(mask_v[:, None, :].bool(), float("-inf"))
    return al.softmax(dim=-1)[0:module.num_heads]   # bs=1 가정


def to_spatial(dist, spatial_shapes, out_hw):
    """flatten된 멀티스케일 분포를 2D로. 각 레벨을 올려서 더한다.

    softmax가 전 레벨을 통째로 걸쳐 한 번 걸리므로, 레벨별 슬라이스는 그 분포의
    일부다. 따라서 평균이 아니라 합이 '위치별 총 attention 질량'이 된다.
    """
    acc = torch.zeros(out_hw, dtype=torch.float32)
    off = 0
    for h, w in spatial_shapes.tolist():
        n = h * w
        seg = dist[off:off + n].reshape(1, 1, h, w).float()
        acc += F.interpolate(seg, size=out_hw, mode="bilinear",
                             align_corners=False)[0, 0]
        off += n
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--images", nargs="+", required=True,
                    help="파일명. FN 케이스를 반드시 넣을 것 (RULES 7)")
    ap.add_argument("--weights", default=None, help="LoRA 체크포인트. 없으면 zero-shot")
    ap.add_argument("--per-layer", action="store_true",
                    help="fusion layer 6개를 따로도 저장")
    a = ap.parse_args()

    from groundingdino.util.train import load_model
    from groundingdino.util.inference import load_image, preprocess_caption

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    img_dir = Path(a.images_dir)

    kind = "zero-shot" if not a.weights else f"FT({Path(a.weights).stem})"
    model = load_model(str(REPO / "groundingdino/config/GroundingDINO_SwinT_OGC.py"),
                       str(REPO / "weights/groundingdino_swint_ogc.pth"),
                       use_lora=bool(a.weights))
    if a.weights:
        from peft import set_peft_model_state_dict
        sd = torch.load(a.weights, map_location="cpu")
        set_peft_model_state_dict(model, sd["model"])
    model.eval().cuda()

    # "implant" 토큰 위치를 실제로 확인한다. 가정하지 않는다.
    tok = model.tokenizer
    caption = preprocess_caption(TEXT_PROMPT)
    ids = tok(caption)["input_ids"]
    toks = tok.convert_ids_to_tokens(ids)
    idxs = [i for i, t in enumerate(toks) if TEXT_PROMPT.startswith(t.replace("##", ""))
            and t not in ("[CLS]", "[SEP]", ".")]
    if not idxs:
        sys.exit(f"[ERROR] '{TEXT_PROMPT}' 토큰을 못 찾음: {toks}")
    print(f"[attn] {kind} | caption '{caption}' -> {toks} | implant 토큰 index {idxs}")

    grab = Grabber(model)
    meta = {"kind": kind, "weights": a.weights, "caption": caption,
            "tokens": toks, "implant_token_index": idxs, "images": {}}

    for name in a.images:
        src = img_dir / name
        if not src.exists():
            sys.exit(f"[ERROR] 이미지 없음: {src}")
        pil = Image.open(src).convert("RGB")
        W, H = pil.size
        _, image = load_image(str(src))

        grab.clear()
        o = model(image[None].cuda(), captions=[caption])
        if grab.spatial_shapes is None:
            sys.exit("[ERROR] spatial_shapes를 못 잡았다")

        scores = o["pred_logits"].sigmoid()[0].max(dim=1)[0]
        top = float(scores.max())
        n_over = int((scores > 0.3).sum())

        maps = []
        for (module, v, l, mask_v) in grab.v_l:
            d = attn_weights_l(module, v, l, mask_v)        # (nhead, ntxt, nimg)
            d = d[:, idxs, :].mean(dim=(0, 1))              # head·토큰 평균 -> (nimg,)
            maps.append(to_spatial(d.cpu(), grab.spatial_shapes, (H, W)))

        avg = torch.stack(maps).mean(0).numpy()

        def save(arr, path, title):
            fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
            ax.imshow(pil)
            n = arr - arr.min()
            n = n / n.max() if n.max() > 0 else n
            ax.imshow(n, cmap="jet", alpha=0.45)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            fig.tight_layout(pad=0.2)
            fig.savefig(path, dpi=110, bbox_inches="tight")
            plt.close(fig)

        stem = Path(name).stem
        save(avg, out / f"{stem}.jpg",
             f"{kind} | '{TEXT_PROMPT}' text-to-image attention "
             f"(mean of 6 fusion layers) | max score {top:.3f}, "
             f"boxes>0.3: {n_over}")
        if a.per_layer:
            for i, m in enumerate(maps):
                save(m.numpy(), out / f"{stem}_L{i}.jpg", f"{kind} | fusion layer {i}")

        meta["images"][name] = {"max_score": top, "n_pred_over_0.3": n_over}
        print(f"  {stem}: max score {top:.3f}, >0.3 박스 {n_over}개")

    grab.remove()
    (out / "attention_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(a.images)}장 -> {out}")
    print("판정은 사람이 한다: attention은 갔는데 점수 낮음 -> 점수 보정 / "
          "attention이 안 감 -> 데이터 확대 (RULES 7)")


if __name__ == "__main__":
    main()
