"""
Stage 1. Backbone 학습 드라이버

yaml(configs/01_backbone.yaml)의 grid(backbone × finetune_mode)를 순회하며
재사용 가능한 백본을 만든다.

  backbone      vit / band-vit / dinov2 / band-dinov2
  finetune_mode full(전체) / partial(마지막 n block만)
  protocol      closed27_open15(27클래스로만 학습) / full42(42클래스 전부)

산출물: checkpoints/backbones/<backbone>_<mode>_<protocol>.pt
→ base.yaml의 backbones.<name>.checkpoint 에 경로를 넣으면
  Stage 2(metric) / Stage 3(classifier)이 그대로 끌어다 쓴다.

실행: python src/run_backbone.py --config configs/01_backbone.yaml
      PROTOCOL=full42 로 프로토콜만 바꿔 재실행 가능
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_utils import as_list, load_config_from_cli
from embeddings import plan_from_config


def main():
    cfg = load_config_from_cli(__doc__)

    from backbone.core import BackboneSpec, log, write_json
    from backbone.train import train_backbone

    protocol = cfg["protocol"]
    registry = cfg["backbones"]
    d = cfg["data"]
    t = cfg["train"]
    band_cfg = cfg.get("band", {}) or {}
    embed_plan = plan_from_config(cfg)   # 학습 종료 시 임베딩 저장 계획

    # 프로토콜에 따라 학습 manifest 선택 —
    # closed27_open15면 백본도 27클래스만 봐야 한다 (open 15 leakage 방지)
    manifest = d["closed_manifest"] if protocol == "closed27_open15" else d["full_manifest"]

    out_root = Path(cfg["paths"]["output_dir"]) / cfg["output_subdir"]
    ckpt_root = Path(cfg["paths"]["checkpoints_dir"]) / "backbones"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # 리스트가 아니어도 된다 — `backbone: vit` 처럼 하나만 적으면 조합 1개로 돈다
    grid = cfg["grid"]
    combos = list(itertools.product(as_list(grid["backbone"]),
                                    as_list(grid["finetune_mode"])))
    log(f"backbone grid ({protocol}): {len(combos)}개 조합 — manifest={manifest}")

    produced = {}
    for name, mode in combos:
        spec = BackboneSpec.from_registry(name, registry)
        tag = f"{name}_{mode}_{protocol}"
        checkpoint = ckpt_root / f"{tag}.pt"
        log(f"\n===== {tag} =====")
        train_backbone(
            spec=spec,
            manifest=manifest,
            protocol=protocol,
            output_dir=out_root / tag,
            checkpoint_path=checkpoint,
            finetune_mode=mode,
            trainable_blocks=int(cfg.get("partial", {}).get("trainable_blocks", 4)),
            band_alpha=float(band_cfg.get("alpha", 4.0)),
            band_loss_weight=float(band_cfg.get("loss_weight", 0.5)),
            attention_blocks=int(band_cfg.get("attention_blocks", 6)),
            mask_column=d.get("band_mask_column", "mask_path"),
            epochs=int(t["epochs"]),
            batch_size=int(t["batch_size"]),
            lr=float(t["lr"]),
            weight_decay=float(t.get("weight_decay", 0.05)),
            label_smoothing=float(t.get("label_smoothing", 0.1)),
            use_class_weights=bool(t.get("use_class_weights", False)),
            grad_checkpointing=bool(t.get("grad_checkpointing", False)),
            num_workers=int(t.get("num_workers", 4)),
            seed=int(t.get("seed", 42)),
            patience=int(t.get("patience", 12)),
            device=cfg.get("device", "auto"),
            embed_plan=embed_plan,
        )
        produced[tag] = str(checkpoint)

    # 다른 스테이지가 쓸 수 있도록 산출 경로를 한 곳에 모아 기록
    index_path = ckpt_root / f"index_{protocol}.json"
    write_json(index_path, {"protocol": protocol, "checkpoints": produced})
    log(f"\nbackbone 학습 완료 → {index_path}")
    log("base.yaml의 backbones.<name>.checkpoint 에 아래 경로를 넣으면 끌어다 쓸 수 있습니다:")
    for tag, path in produced.items():
        log(f"  {tag}: {path}")


if __name__ == "__main__":
    main()
