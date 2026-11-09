"""
Stage 2. Metric learning 그리드 드라이버

yaml(configs/03_metric_grid.yaml 또는 03_clustering_full42.yaml)의
grid(backbone × embedding_loss)를 순회하며:
  1) loss ∈ {supcon, triplet, arcface} → metric.train으로 임베딩 학습
     loss = none                       → 학습 생략 (백본 임베딩 그대로)
  2) metric.evaluate로 prototype 방법 × 거리 전체 평가
     - closed27_open15: 42-way 갤러리 (closed 27 train + open 15 support)
     - full42         : 42클래스 prototype 분류 (clustering 방식)

학습은 조합당 1회, 거리/prototype은 재학습 없이 평가만 반복 (비용 절감).

실행: python src/run_metric.py --config configs/03_metric_grid.yaml
"""
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_utils import as_list, load_config_from_cli
from embeddings import plan_from_config, try_dump_embeddings


def main():
    cfg = load_config_from_cli(__doc__)

    import torch

    from metric.core import choose_device, log
    from metric.evaluate import evaluate_gallery
    from metric.models import build_metric_model
    from metric.train import load_trained_model, train_metric

    protocol = cfg["protocol"]
    registry = cfg["backbones"]
    t = cfg["train"]
    e = cfg["eval"]
    d = cfg["data"]
    device = torch.device(choose_device(cfg.get("device", "auto")))
    out_root = Path(cfg["paths"]["output_dir"]) / cfg["output_subdir"]
    embed_plan = plan_from_config(cfg)   # 학습 종료 시 임베딩 저장 계획

    # 임베딩 학습용 manifest (closed27_open15은 closed만으로 학습 — leakage 방지)
    train_manifest = d["closed_manifest"] if protocol == "closed27_open15" else d["full_manifest"]

    # 리스트가 아니어도 된다 — `backbone: vit` 처럼 하나만 적으면 조합 1개로 돈다
    grid = cfg["grid"]
    combos = list(itertools.product(as_list(grid["backbone"]),
                                    as_list(grid["embedding_loss"])))
    log(f"metric grid ({protocol}): {len(combos)}개 조합 "
        f"(x {len(as_list(e.get('prototype', {}).get('methods', ['mean'])))} prototype "
        f"x {len(as_list(e.get('distances')))} distance 평가)")

    all_rows = []
    for backbone, loss in combos:
        tag = f"{protocol}_{backbone}_{loss}"
        out_dir = out_root / tag
        log(f"\n===== {tag} =====")

        if loss == "none":
            # 학습 없음: 백본 임베딩을 그대로 사용
            model, image_size = build_metric_model(backbone, registry, int(t["embedding_dim"]))
            model = model.to(device).eval()
            embedding_source = "backbone"
        else:
            run_dir = train_metric(
                manifest=train_manifest,
                output_dir=out_dir / "train",
                backbone_name=backbone,
                backbone_registry=registry,
                loss_name=loss,
                epochs=int(t["epochs"]),
                batch_size=int(t["batch_size"]),
                samples_per_class=int(t.get("samples_per_class", 2)),
                lr=float(t["lr"]),
                weight_decay=float(t.get("weight_decay", 0.05)),
                embedding_dim=int(t["embedding_dim"]),
                temperature=float(t.get("temperature", 0.07)),
                triplet_margin=float(t.get("triplet_margin", 0.2)),
                arcface_s=float(t.get("arcface", {}).get("s", 30.0)),
                arcface_m=float(t.get("arcface", {}).get("m", 0.5)),
                covaware=t.get("covaware"),
                grad_checkpointing=bool(t.get("grad_checkpointing", False)),
                num_workers=int(t.get("num_workers", 4)),
                seed=int(t.get("seed", 42)),
                patience=int(t.get("patience", 12)),
                device=cfg.get("device", "auto"),
            )
            model, image_size = load_trained_model(
                run_dir / "checkpoints" / "best.pt", backbone, registry,
                int(t["embedding_dim"]), device,
            )
            embedding_source = e.get("embedding_source", "projection")

        summary = evaluate_gallery(
            model=model,
            protocol=protocol,
            data_cfg=d,
            eval_cfg=e,
            image_size=image_size,
            batch_size=int(t["batch_size"]),
            num_workers=int(t.get("num_workers", 4)),
            device=device,
            embedding_source=embedding_source,
            output_dir=out_dir / "eval",
            seed=int(t.get("seed", 42)),
        )
        summary.insert(0, "backbone", backbone)
        summary.insert(1, "loss", loss)
        all_rows.append(summary)

        # ---- 학습 종료 hook: 전체 이미지 backbone/projection 임베딩 (평가가 끝난 뒤, 실패해도 평가는 남는다) ----
        if embed_plan.enabled:
            import torch.nn.functional as F

            from metric.core import build_transforms

            model.eval()
            metric_model = model            # 아래 del model 과 무관하게 클로저가 잡을 이름
            trained_head = loss != "none"   # none 은 projector 가 학습되지 않았다 → 백본 L2 정규화를 head 로

            def _forward(images):
                features = metric_model.forward_features(images)
                head = metric_model.projector(features) if trained_head else features
                return features, F.normalize(head, dim=1)

            try_dump_embeddings(
                output_dir=out_dir / "embeddings", plan=embed_plan, forward=_forward,
                transform=build_transforms(image_size, train=False), stage="metric",
                model_name=registry[backbone]["timm"],
                checkpoint=(out_dir / "train" / "checkpoints" / "best.pt" if trained_head
                            else registry[backbone].get("checkpoint")),
                device=device, head_kind="projection" if trained_head else "l2norm_backbone",
                extra={"backbone": backbone, "loss": loss, "protocol": protocol,
                       "embedding_source": embedding_source},
            )

        # GPU 메모리 정리 후 다음 조합
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    grand = pd.concat(all_rows, ignore_index=True)
    out_root.mkdir(parents=True, exist_ok=True)
    grand.to_csv(out_root / f"grid_summary_{protocol}.csv", index=False)
    log(f"\nmetric grid 완료 → {out_root / f'grid_summary_{protocol}.csv'}")
    with pd.option_context("display.width", 200):
        print(grand.to_string(index=False))


if __name__ == "__main__":
    main()
