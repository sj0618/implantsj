"""
Stage 3-<2>. masked-vit classifier 드라이버

  1) masked_vit.training — SAM2 seg 모델로 런타임에 band만 남기고
     중앙/배경을 가린 crop을 만들어 ViT 학습
     (test 평가 + attention map n개 저장은 training이 내부적으로 수행)
  2) masked_vit.evaluation / gradient_rollout — 저장된 체크포인트로 test 재평가:
     - input-mode "original": 원본 픽셀 입력 (핵심 실험 —
       가리고 학습한 모델이 원본에서도 band에 집중하는지 focus_gain으로 정량화)
     - input-mode "band"    : 학습 분포와 동일한 입력 (비교 기준)
     각 모드마다 attention map + gradient rollout 시각화를 n개 저장

실행: python src/run_masked_vit.py --config configs/04_masked_vit.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_utils import as_list, load_config_from_cli
from embeddings import ArrayDataset, plan_from_config, timm_classifier_forward, try_dump_embeddings


def resolve_model(cfg: dict) -> tuple[str, str | None, int]:
    """(timm 모델명, 초기 체크포인트, image_size). 레지스트리 이름이면 Stage 1 백본을 쓴다."""
    name = cfg.get("backbone") or cfg.get("model")
    if not name:
        raise ValueError("yaml에 backbone(또는 model)이 없습니다")
    registry = cfg.get("backbones", {})
    if name in registry:
        entry = registry[name]
        return entry["timm"], entry.get("checkpoint"), int(entry.get("image_size", 224))
    return name, cfg.get("checkpoint"), int(cfg.get("input_size", 224))


def _find_best_checkpoint(output_dir: Path) -> Path:
    candidates = sorted((output_dir / "checkpoints").glob("best.*"))
    ckpts = [p for p in candidates if p.suffix in (".pt", ".ckpt", ".safetensors")]
    if not ckpts:
        raise FileNotFoundError(f"best checkpoint가 없습니다: {output_dir}/checkpoints")
    return ckpts[0]


def main():
    cfg = load_config_from_cli(__doc__)
    t = cfg["train"]
    d = cfg["data"]
    s = cfg["sam"]
    b = cfg["band"]
    viz = cfg["viz"]
    device = cfg.get("device", "auto")
    out_dir = Path(cfg["paths"]["output_dir"]) / cfg["output_subdir"] / cfg["protocol"]

    from masked_vit import evaluation as eval_original
    from masked_vit import gradient_rollout, training
    from split_manifest import split_by_protocol

    manifests = split_by_protocol(cfg)
    model_name, checkpoint, image_size = resolve_model(cfg)

    # ---- 1) 학습 (band-only 입력) ----
    print("===== masked-vit: 학습 (band-only crop) =====", flush=True)
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
        "--device", device,
        "--sam2-config", s["sam2_config"],
        "--sam2-checkpoint", s["sam2_checkpoint"],
        "--sam-model-checkpoint", s["model_checkpoint"],
        "--sam-input-size", str(s["input_size"]),
        "--sam-threshold", str(s["threshold"]),
        "--band-outer-radius", str(b["outer_radius"]),
        "--band-inner-radius", str(b["inner_radius"]),
        "--band-max-failure-rate", str(b.get("max_failure_rate", 0.0)),
        "--test-attention-samples", str(viz["attention_samples"]),
    ]
    if checkpoint:
        args += ["--checkpoint", str(checkpoint)]
        print(f"  초기 백본: {checkpoint}", flush=True)
    training.main(args)

    best = _find_best_checkpoint(out_dir)

    # ---- 2) valid 재평가 (Stage 5 fuse가 가중치/임계값을 학습할 때 쓴다) ----
    # test로 가중치를 학습하면 그 test 성능을 믿을 수 없게 되므로 valid가 반드시 필요하다.
    # 배포 조건과 같은 original(원본 픽셀) 모드로만 평가한다.
    if cfg.get("eval_valid", True):
        print("\n===== masked-vit eval: split=valid input-mode=original =====", flush=True)
        eval_original.main([
            "--checkpoint", str(best),
            "--manifest", manifests["valid"],
            "--input-mode", "original",
            "--sam-model-checkpoint", s["model_checkpoint"],
            "--sam2-config", s["sam2_config"],
            "--sam2-checkpoint", s["sam2_checkpoint"],
            "--device", device,
            "--output-dir", str(out_dir / "eval_valid"),
            "--attention-samples", "0",     # valid는 점수만 필요 — 시각화 생략
        ])

    # ---- 3) test 변형별 재평가: attention 평가 + gradient rollout ----
    for mode in as_list(cfg["test_variants"]):          # original(핵심) / band(비교)
        print(f"\n===== masked-vit eval: input-mode={mode} =====", flush=True)
        common = [
            "--checkpoint", str(best),
            "--manifest", manifests["test"],
            "--input-mode", mode,
            "--sam-model-checkpoint", s["model_checkpoint"],
            "--sam2-config", s["sam2_config"],
            "--sam2-checkpoint", s["sam2_checkpoint"],
            "--device", device,
        ]
        eval_original.main(common + [
            "--output-dir", str(out_dir / f"eval_{mode}"),
            "--attention-samples", str(viz["attention_samples"]),
        ])
        gradient_rollout.main(common + [
            "--output-dir", str(out_dir / f"rollout_{mode}"),
            "--rollout-samples", str(viz["rollout_samples"]),
        ])

    # ---- 4) 학습 종료 hook: 전체 이미지 backbone/logits 임베딩 (best checkpoint 기준) ----
    # 평가·rollout 이 다 끝난 뒤 마지막에 돌고, 실패해도 그 산출물은 남는다.
    # embeddings.input_mode=original 이면 원본 픽셀(배포 조건, SAM2 불필요),
    # band 면 학습 분포와 같은 SAM2 band crop 으로 임베딩한다 (전체 이미지에 seg 를 돌리므로 느리다).
    embed_plan = plan_from_config(cfg)
    if embed_plan.enabled:
        from masked_vit.core import choose_device, create_transforms, load_classifier_checkpoint

        resolved = choose_device(device)
        model, meta = load_classifier_checkpoint(best, resolved)
        _, eval_transform, _, _ = create_transforms(model, image_size)
        segmenter = None
        dataset_factory = None
        if embed_plan.input_mode == "band":
            from masked_vit.band_preprocessing import BandCropConfig
            from masked_vit.data import precompute_band_cache
            from masked_vit.segmentation import RuntimeSegmenter, SegmenterConfig

            segmenter = RuntimeSegmenter(SegmenterConfig(
                sam2_config=s["sam2_config"], sam2_checkpoint=s["sam2_checkpoint"],
                model_checkpoint=s["model_checkpoint"], input_size=int(s["input_size"]),
                threshold=float(s["threshold"]), morph_kernel_size=3, batch_size=2,
            ), resolved)
            band_config = BandCropConfig(
                working_size=224, outer_radius=int(b["outer_radius"]),
                inner_radius=int(b["inner_radius"]), crop_margin=4, output_size=image_size,
            )

            def dataset_factory(frame):
                # split 마다 따로 부르므로 max_failure_rate 는 split 단위로 적용된다 (학습은 전체 기준)
                split_name = str(frame["split"].iloc[0]) if len(frame) else "all"
                frames = {"embed": frame.reset_index(drop=True)}
                caches, _, _ = precompute_band_cache(
                    frames, segmenter, band_config,
                    out_dir / "embeddings" / "preprocessing" / split_name,
                    input_mode="band", max_failure_rate=float(b.get("max_failure_rate", 0.0)),
                )
                # 실패 허용치 안에서 빠진 이미지는 frames 에서도 빠져 있다 → 길이가 맞는다
                return ArrayDataset(caches["embed"], eval_transform), frames["embed"]

        try:
            try_dump_embeddings(
                output_dir=out_dir / "embeddings", plan=embed_plan,
                forward=timm_classifier_forward(model), transform=eval_transform,
                dataset_factory=dataset_factory, stage="classifier/masked_vit",
                model_name=meta["architecture"], checkpoint=best, device=resolved,
                head_kind="logits",
                extra={"input_mode": embed_plan.input_mode, "protocol": cfg["protocol"],
                       "labels": meta["labels"]},
            )
        finally:
            if segmenter is not None:
                segmenter.close()
        del model


    print("\nmasked-vit 완료:", out_dir)


if __name__ == "__main__":
    main()
