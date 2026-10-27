"""
백본 레지스트리 점검 도구 (학습 전에 한 번 돌려보는 용도)

base.yaml의 backbones 항목마다:
  - timm 모델명이 실제로 존재하는지
  - 지정한 image_size로 만들어지는지
  - 파라미터 수 / depth / embed_dim / patch grid
  - band-* 백본이면 attention 캡처가 실제로 붙는지
를 확인한다. 큰 모델로 바꾸기 전에 이름과 메모리 감을 잡는 데 쓴다.

실행:
  python src/check_backbones.py --config configs/01_backbone.yaml            # grid에 있는 것만
  python src/check_backbones.py --config configs/01_backbone.yaml --all      # 레지스트리 전체
  python src/check_backbones.py --config configs/01_backbone.yaml --search vit_large   # 후보 검색
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_utils import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--all", action="store_true", help="grid 대신 레지스트리 전체 점검")
    parser.add_argument("--search", help="timm 모델 이름 검색만 하고 종료 (예: vit_large)")
    args = parser.parse_args()

    import timm
    import torch

    if args.search:
        matches = timm.list_models(f"*{args.search}*", pretrained=True)
        print(f"'{args.search}' 검색 결과 ({len(matches)}개, pretrained 있는 것만):")
        for name in matches:
            print("  ", name)
        return

    cfg = load_config(args.config)
    registry = cfg["backbones"]
    names = sorted(registry) if args.all else cfg.get("grid", {}).get("backbone", sorted(registry))

    from backbone.core import BackboneSpec, attach_attention_capture, create_model, patch_grid_size

    print(f"{'backbone':22s} {'params':>10s} {'depth':>6s} {'dim':>5s} {'grid':>8s}  band  status")
    print("-" * 78)
    failures = []
    for name in names:
        try:
            spec = BackboneSpec.from_registry(name, registry)
            # pretrained=False로 구조만 확인 (가중치 다운로드 없이 빠르게)
            model = create_model(spec, num_classes=2, pretrained=False)
            params = sum(p.numel() for p in model.parameters())
            blocks = getattr(model, "blocks", None)
            depth = len(blocks) if blocks is not None else 0
            dim = int(getattr(model, "num_features", 0))
            grid = status = ""
            if spec.use_band:
                attach_attention_capture(model, min(6, depth))
                gh, gw = patch_grid_size(model)
                grid = f"{gh}x{gw}"
                with torch.no_grad():
                    model.eval()
                    model(torch.zeros(1, 3, spec.image_size, spec.image_size))
                from backbone.core import get_cls_patch_attention

                attn = get_cls_patch_attention(model)
                if attn.shape[1] != gh * gw:
                    raise RuntimeError(f"attention 패치 수 {attn.shape[1]} != grid {gh*gw}")
                status = "attention OK"
            else:
                try:
                    gh, gw = patch_grid_size(model)
                    grid = f"{gh}x{gw}"
                except TypeError:
                    grid = "-"
                status = "OK"
            print(f"{name:22s} {params:>10,} {depth:>6d} {dim:>5d} {grid:>8s}  "
                  f"{'Y' if spec.use_band else 'N':^4s}  {status}")
            del model
        except Exception as exc:
            print(f"{name:22s} {'-':>10s} {'-':>6s} {'-':>5s} {'-':>8s}   -    "
                  f"FAIL {type(exc).__name__}: {exc}")
            failures.append(name)

    print()
    if failures:
        print(f"실패 {len(failures)}개: {failures}")
        print("timm 이름이 틀렸을 수 있습니다. --search 로 후보를 찾아보세요.")
        raise SystemExit(1)
    print(f"{len(names)}개 백본 전부 정상")


if __name__ == "__main__":
    main()
