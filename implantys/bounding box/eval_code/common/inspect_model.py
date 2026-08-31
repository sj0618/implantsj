#!/usr/bin/env python3
"""
attention hook을 걸기 위해 repo/모델 내부 구조를 조사한다. (RULES.md 7번 준비)

읽기만 한다. 모델을 바꾸거나 저장하지 않는다.
출력을 그대로 복사해서 주면 attention_map.py의 hook을 정확히 채울 수 있다.

  export PROJ_ROOT=/workspace/Grounding-Dino-FineTuning/2nd
  source /workspace/implant-env/bin/activate
  cd /workspace/Grounding-Dino-FineTuning
  python "$PROJ_ROOT/eval_code/common/inspect_model.py" > /tmp/inspect.txt 2>&1
  cat /tmp/inspect.txt
"""

import inspect
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

GDINO = Path(os.environ.get("GDINO_ROOT", "/workspace/Grounding-Dino-FineTuning"))


def sec(t):
    print("\n" + "=" * 70)
    print(f"## {t}")
    print("=" * 70)


class MC:
    config_path = str(GDINO / "groundingdino/config/GroundingDINO_SwinT_OGC.py")
    weights_path = str(GDINO / "weights/groundingdino_swint_ogc.pth")
    lora_weights = None
    lora_weigths = None


def main():
    sec("0. 환경")
    import torch
    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | "
          f"available={torch.cuda.is_available()}")
    print(f"GDINO_ROOT = {GDINO}")

    import groundingdino
    print(f"groundingdino: {Path(groundingdino.__file__).parent}")

    sec("1. inference.predict() 소스 — 내부 호출 방식 확인")
    from groundingdino.util import inference
    try:
        print(inspect.getsource(inference.predict))
    except Exception as e:
        print(f"(소스 못 읽음: {e})")

    sec("2. load_model() 소스 — LoRA 적용 지점 확인")
    try:
        print(inspect.getsource(inference.load_model))
    except Exception as e:
        print(f"(소스 못 읽음: {e})")

    sec("3. 모델 로드")
    model = inference.load_model(MC(), use_lora=False)
    model.eval()
    print(f"class: {type(model).__module__}.{type(model).__name__}")
    print(f"파라미터: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    sec("4. attention 관련 모듈 (hook 후보)")
    hits = [(n, type(m).__name__) for n, m in model.named_modules()
            if any(k in n.lower() for k in ("attn", "attention", "cross", "text"))]
    print(f"총 {len(hits)}개\n")
    for n, t in hits:
        print(f"  {n:62} {t}")

    sec("5. 최상위 구조 (depth 2)")
    for n, m in model.named_modules():
        if n and n.count(".") <= 1:
            print(f"  {n:44} {type(m).__name__}")

    sec("6. transformer.decoder 구조")
    dec = getattr(getattr(model, "transformer", None), "decoder", None)
    if dec is None:
        print("  transformer.decoder 없음 — 5번 출력에서 경로를 찾을 것")
    else:
        print(f"  class: {type(dec).__name__}")
        layers = getattr(dec, "layers", None)
        if layers is not None:
            print(f"  layers: {len(layers)}개")
            print(f"  layer[0] 하위 모듈:")
            for n, m in layers[0].named_modules():
                if n:
                    print(f"    {n:40} {type(m).__name__}")

    sec('7. "implant" 토크나이즈 — 토큰 인덱스 확인')
    # subword로 쪼개지면 해당 토큰들의 attention을 평균내야 한다.
    # [CLS]/[SEP]를 잘못 집으면 그림은 그럴듯한데 완전히 틀린다.
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        for cap in ["implant", "implant .", "implant ."]:
            enc = tok(cap, return_tensors="pt")
            ids = enc["input_ids"][0].tolist()
            print(f'  caption={cap!r}')
            print(f'    ids    = {ids}')
            print(f'    tokens = {tok.convert_ids_to_tokens(ids)}')
    except Exception as e:
        print(f"  실패: {e}")

    sec("8. tokenizer 위치 (모델이 내부에 들고 있는지)")
    for attr in ["tokenizer", "bert", "text_encoder", "text_backbone"]:
        v = getattr(model, attr, None)
        print(f"  model.{attr:14} = {type(v).__name__ if v is not None else '없음'}")

    sec("9. forward 시그니처")
    try:
        print(f"  forward{inspect.signature(model.forward)}")
    except Exception as e:
        print(f"  실패: {e}")

    sec("10. deformable attention 모듈 (좌표 복원에 필요)")
    for n, m in model.named_modules():
        t = type(m).__name__
        if "MSDeform" in t or "Deform" in t:
            print(f"  {n:62} {t}")
            for a in ["n_levels", "n_heads", "n_points", "num_heads", "num_levels", "num_points"]:
                if hasattr(m, a):
                    print(f"      {a} = {getattr(m, a)}")
            break

    print("\n" + "=" * 70)
    print("끝. 이 출력을 그대로 복사해서 주면 hook을 채울 수 있다.")
    print("=" * 70)


if __name__ == "__main__":
    main()
