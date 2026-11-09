"""
Stage 3-<1>. band-vit classifier — α/λ/block sweep 드라이버

yaml(configs/04_band_vit.yaml)의 sweep 조합을 순회하며:
  1) band_vit.training  — band-attention loss로 ViT 학습
     (test 평가 + attention map n개 저장은 training이 내부적으로 수행)
  2) band_vit.rollout_vit2 — test 이미지에 대해 gradient rollout 시각화 n개 저장

실행: python src/run_band_vit.py --config configs/04_band_vit.yaml
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_utils import as_list, load_config_from_cli
from embeddings import plan_from_config, timm_classifier_forward, try_dump_embeddings


def _find_best_checkpoint(output_dir: Path) -> Path:
    candidates = sorted((output_dir / "checkpoints").glob("best.*"))
    ckpts = [p for p in candidates if p.suffix in (".pt", ".ckpt", ".safetensors")]
    if not ckpts:
        raise FileNotFoundError(f"best checkpoint가 없습니다: {output_dir}/checkpoints")
    return ckpts[0]


def resolve_model(cfg: dict) -> tuple[str, str | None, int]:
    """(timm 모델명, 초기 체크포인트, image_size).

    backbone에 base.yaml 레지스트리 이름을 적으면 Stage 1에서 만든 백본을 끌어다 쓴다.
    timm 이름을 직접 적어도 되고, 그때는 pretrained로 시작한다.
    """
    name = cfg.get("backbone") or cfg.get("model")
    if not name:
        raise ValueError("yaml에 backbone(또는 model)이 없습니다")
    registry = cfg.get("backbones", {})
    if name in registry:
        entry = registry[name]
        return entry["timm"], entry.get("checkpoint"), int(entry.get("image_size", 224))
    return name, cfg.get("checkpoint"), int(cfg.get("input_size", 224))


def run_one(cfg: dict, manifests: dict, alpha: float, lam: float, block: int,
            embed_plan) -> None:
    from band_vit import rollout_vit2, training

    t = cfg["train"]
    viz = cfg["viz"]
    model_name, checkpoint, image_size = resolve_model(cfg)
    out_root = Path(cfg["paths"]["output_dir"]) / cfg["output_subdir"]
    out_dir = out_root / f"{cfg['protocol']}_a{alpha}_l{lam}_b{block}"

    print(f"\n===== band-vit: alpha={alpha} lambda={lam} block={block} "
          f"({cfg['protocol']}) =====", flush=True)

    args = [
        "--model", model_name,
        "--train-data", manifests["train"],
        "--valid-data", manifests["valid"],
        "--test-data", manifests["test"],
        "--output-dir", str(out_dir),
        "--input-size", str(image_size),
        "--epochs", str(t["epochs"]),
        "--batch-size", str(t["batch_size"]),
        "--learning-rate", str(t["lr"]),
        "--num-workers", str(t["num_workers"]),
        "--seed", str(t["seed"]),
        "--device", cfg.get("device", "auto"),
        "--band-alpha", str(alpha),
        "--band-loss-weight", str(lam),
        "--band-attention-blocks", str(block),
        "--band-mask-column", cfg["data"]["band_mask_column"],
        "--test-attention-samples", str(viz["attention_samples"]),
    ]
    if checkpoint:
        # Stage 1에서 만든 백본을 초기 weight로 (없으면 timm pretrained)
        args += ["--checkpoint", str(checkpoint)]
        print(f"  초기 백본: {checkpoint}", flush=True)
    training.main(args)

    best = _find_best_checkpoint(out_dir)

    # gradient rollout 시각화 (test 이미지 n개)
    rollout_vit2.main([
        "--checkpoint", str(best),
        "--manifest", manifests["test"],
        "--output-dir", str(out_dir / "rollout_test"),
        "--rollout-samples", str(viz["rollout_samples"]),
        "--device", cfg.get("device", "auto"),
    ])

    # ---- 학습 종료 hook: 전체 이미지 backbone/logits 임베딩 (best checkpoint 기준) ----
    # 평가·rollout 이 다 끝난 뒤 마지막에 돌고, 실패해도 그 산출물은 남는다.
    if embed_plan.enabled:
        import torch

        from band_vit.models import choose_device, create_transforms, load_classifier_checkpoint

        resolved = choose_device(cfg.get("device", "auto"))
        model, meta = load_classifier_checkpoint(best, resolved)
        _, eval_transform, _, _ = create_transforms(model, image_size)
        try_dump_embeddings(
            output_dir=out_dir / "embeddings", plan=embed_plan,
            forward=timm_classifier_forward(model), transform=eval_transform,
            stage="classifier/band_vit", model_name=meta["architecture"], checkpoint=best,
            device=resolved, head_kind="logits",
            extra={"alpha": alpha, "lambda": lam, "block": block,
                   "protocol": cfg["protocol"], "labels": meta["labels"]},
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    cfg = load_config_from_cli(__doc__)
    from split_manifest import split_by_protocol

    manifests = split_by_protocol(cfg)
    embed_plan = plan_from_config(cfg)   # 학습 종료 시 임베딩 저장 계획
    # 리스트가 아니어도 된다 — `alpha: 0.5` 처럼 하나만 적으면 조합 1개로 돈다
    sweep = cfg["sweep"]
    combos = list(itertools.product(as_list(sweep["alpha"]), as_list(sweep["lambda"]),
                                    as_list(sweep["block"])))
    print(f"band-vit sweep ({cfg['protocol']}): {len(combos)}개 조합")

    for alpha, lam, block in combos:
        run_one(cfg, manifests, alpha, lam, block, embed_plan)

    print(f"\nband-vit sweep 완료: {len(combos)}개")


if __name__ == "__main__":
    main()
