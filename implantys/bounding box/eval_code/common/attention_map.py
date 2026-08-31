#!/usr/bin/env python3
"""
텍스트 토큰 "implant" ↔ 이미지 feature 사이의 cross-attention 추출. (RULES.md 7번)

⚠️ TODO — repo 내부 구조를 봐야 채울 수 있는 부분이 있다. (RULES.md 8번: 코드는 로컬,
   실행은 pod. 현재 로컬에 repo가 없어 hook 지점을 확정하지 못했다)

FN 케이스에서 이 구분이 다음 라운드 전략을 결정한다:
  - attention은 갔는데 점수가 낮음  → 점수 보정 문제 → threshold·score head 조정
  - attention이 아예 안 감          → 표현 학습 문제 → 학습 데이터 확대

  python attention_map.py --out results/00_zeroshot --data-root /workspace/dataset2 \
      --images adin302_....jpg noris773_....jpg

────────────────────────────────────────────────────────────────────────────
pod에서 채울 것 (순서대로)

1. hook 지점 찾기
     python -c "
     from groundingdino.util.inference import load_model
     m = load_model(MC(), use_lora=False)
     for n, _ in m.named_modules():
         if 'attn' in n.lower() or 'cross' in n.lower(): print(n)
     "
   GroundingDINO는 보통 transformer.decoder.layers[i].ca_text (text cross-attention)
   또는 .cross_attn 계열에 붙는다. 실제 이름을 확인해 ATTN_MODULE_PATH에 넣을 것.

2. "implant" 토큰 인덱스 확인
   BERT tokenizer가 "implant"를 몇 개 토큰으로 쪼개는지 먼저 볼 것.
   subword로 쪼개지면(im ##plant 등) 해당 토큰들의 attention을 평균낸다.
   [CLS]/[SEP]를 잘못 집으면 그림은 그럴듯한데 완전히 틀린 results가 나온다.

3. attention weight -> 이미지 좌표 복원
   Grounding DINO는 multi-scale deformable attention을 쓴다. attention weight가
   (query, head, level, point) 형태라 단순 reshape로 HxW가 나오지 않는다.
   sampling_locations와 attention_weights를 같이 받아 이미지 좌표에 흩뿌려야 한다.
   → 이 부분이 제일 까다롭다. 먼저 zero-shot으로 검증할 것 (RULES.md 7번이
     "학습 전에 구현"인 이유).

4. 검증 방법
   임플란트가 크고 뚜렷한 이미지(모델이 확실히 맞히는 것)로 먼저 돌려서
   attention이 임플란트 위에 오는지 확인. 거기서 안 맞으면 hook이 틀린 것이다.
   FN 케이스부터 보면 "attention이 안 갔다"가 버그인지 진짜인지 구분 못 한다.
────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

TEXT_PROMPT = "implant"

# TODO(pod): 위 1번에서 확인한 실제 모듈 경로로 교체
ATTN_MODULE_PATH = None      # 예: "transformer.decoder.layers.5.ca_text"


class MC:
    def __init__(self, cfg, weights, lora=None):
        self.config_path = cfg
        self.weights_path = weights
        self.lora_weights = lora
        self.lora_weigths = lora


def capture_attention(model, image, prompt=TEXT_PROMPT):
    """
    TODO(pod): forward hook으로 cross-attention을 받아 (H, W) heatmap을 반환한다.

    골격:
        store = {}
        def hook(module, inp, out):
            # out 형태를 print해서 먼저 확인할 것. 모듈마다 다르다.
            store["attn"] = out
        mod = dict(model.named_modules())[ATTN_MODULE_PATH]
        h = mod.register_forward_hook(hook)
        with torch.no_grad():
            model(image[None], captions=[prompt])
        h.remove()
        # store["attn"] -> "implant" 토큰 행만 골라 이미지 좌표로 복원 -> (H, W)
        return heatmap
    """
    raise NotImplementedError(
        "attention hook이 아직 구현되지 않았다. 이 파일 상단의 pod 체크리스트를 따를 것.\n"
        "RULES.md 7번: 학습을 시작하기 전에 zero-shot으로 먼저 검증해야 한다."
    )


def overlay(image_path, heatmap, out_path, boxes=None):
    """heatmap을 원본 위에 겹쳐 저장. repo와 무관하므로 미리 구현해둔다."""
    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    im = Image.open(image_path).convert("RGB")
    W, H = im.size
    hm = np.array(heatmap, dtype=float)
    if hm.shape != (H, W):
        hm = np.array(Image.fromarray(hm).resize((W, H), Image.BILINEAR))
    if hm.max() > hm.min():
        hm = (hm - hm.min()) / (hm.max() - hm.min())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(im)
    ax.imshow(hm, cmap="jet", alpha=0.45)
    if boxes:
        for x, y, w, h in boxes:
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False,
                                       edgecolor="#2ca02c", lw=2))
    ax.axis("off")
    ax.set_title(f'cross-attention: "{TEXT_PROMPT}"', fontsize=10)
    fig.tight_layout(); fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default="/workspace/dataset2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--gdino-root", default=os.environ.get("GDINO_ROOT", ""))
    ap.add_argument("--lora", default=None)
    ap.add_argument("--images", nargs="*", default=None,
                    help="대상 파일명. 미지정 시 viz/의 _FN 파일에서 자동 추출")
    a = ap.parse_args()

    if ATTN_MODULE_PATH is None:
        sys.exit(
            "[TODO] ATTN_MODULE_PATH가 비어있다.\n"
            "  이 파일 상단 체크리스트 1번을 pod에서 먼저 수행할 것.\n"
            "  RULES.md 7번: attention은 학습 전에 zero-shot으로 검증한다."
        )

    out = Path(a.out) / "attention"
    out.mkdir(parents=True, exist_ok=True)

    targets = a.images
    if not targets:
        # FN 케이스 우선 (RULES.md 7번)
        viz = Path(a.out) / "viz"
        targets = [p.name.replace("_FN", "") for p in viz.glob("*_FN.jpg")]
        print(f"FN 케이스 {len(targets)}장 자동 선택")

    from groundingdino.util.inference import load_model, load_image
    g = Path(a.gdino_root)
    model = load_model(MC(str(g / "groundingdino/config/GroundingDINO_SwinT_OGC.py"),
                          str(g / "weights/groundingdino_swint_ogc.pth"), a.lora),
                       use_lora=bool(a.lora))

    split_dir = Path(a.data_root) / a.split
    coco = json.loads((split_dir / "annotations.json").read_text())
    gts: dict[str, list] = {}
    id2name = {i["id"]: i["file_name"] for i in coco["images"]}
    for an in coco["annotations"]:
        gts.setdefault(id2name[an["image_id"]], []).append(an["bbox"])

    for fn in targets:
        p = split_dir / "images" / fn
        _, image = load_image(str(p))
        hm = capture_attention(model, image)
        overlay(p, hm, out / f"att_{Path(fn).stem}.jpg", gts.get(fn))
        print(f"  {fn}")

    print(f"\n{len(targets)}장 -> {out}")


if __name__ == "__main__":
    main()
