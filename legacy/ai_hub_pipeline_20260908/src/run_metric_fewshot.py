"""
Stage 2c. Few-shot target adaptation — 별개 실험

"새 임플란트 시스템을 몇 장만 등록해도 인식되는가"를 본다.
클래스마다 support를 1/5/10장만 뽑아 prototype을 만들고 나머지를 query로 평가,
support 뽑기를 여러 seed로 반복해 평균±표준편차를 보고한다.

학습된 metric 체크포인트를 재사용하며 재학습하지 않는다
(checkpoint를 비워두면 백본 임베딩 그대로 평가).

실행: python src/run_metric_fewshot.py --config configs/03_metric_fewshot.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_utils import load_config_from_cli


def main():
    cfg = load_config_from_cli(__doc__)

    import torch

    from metric.core import choose_device, log
    from metric.fewshot import run_fewshot
    from metric.models import build_metric_model
    from metric.train import load_trained_model

    registry = cfg["backbones"]
    t = cfg["train"]
    e = cfg["eval"]
    d = cfg["data"]
    f = cfg["fewshot"]
    device = torch.device(choose_device(cfg.get("device", "auto")))
    out_root = Path(cfg["paths"]["output_dir"]) / cfg["output_subdir"]

    all_summaries = []
    for entry in cfg["models"]:
        name = entry["name"]
        backbone = entry["backbone"]
        checkpoint = entry.get("checkpoint")
        log(f"\n===== few-shot: {name} (backbone={backbone}) =====")

        if checkpoint:
            model, image_size = load_trained_model(
                checkpoint, backbone, registry, int(t["embedding_dim"]), device
            )
            embedding_source = e.get("embedding_source", "projection")
        else:
            model, image_size = build_metric_model(backbone, registry, int(t["embedding_dim"]))
            model = model.to(device).eval()
            embedding_source = "backbone"   # 학습 안 된 projection은 의미 없음

        summary = run_fewshot(
            model=model,
            data_cfg=d,
            eval_cfg=e,
            shots_list=f["shots"],
            seeds=f["seeds"],
            image_size=image_size,
            batch_size=int(t["batch_size"]),
            num_workers=int(t.get("num_workers", 4)),
            device=device,
            embedding_source=embedding_source,
            output_dir=out_root / name,
        )
        summary.insert(0, "model", name)
        all_summaries.append(summary)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    import pandas as pd

    grand = pd.concat(all_summaries, ignore_index=True)
    out_root.mkdir(parents=True, exist_ok=True)
    grand.to_csv(out_root / "fewshot_grid_summary.csv", index=False)
    log(f"\nfew-shot 완료 → {out_root / 'fewshot_grid_summary.csv'}")
    with pd.option_context("display.width", 200):
        print(grand.to_string(index=False))


if __name__ == "__main__":
    main()
