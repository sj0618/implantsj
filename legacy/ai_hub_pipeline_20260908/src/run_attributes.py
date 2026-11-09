"""
Stage 4. Attributes 드라이버 (노션 5번)

각 feature마다 모델을 만들어 학습하고, 예측을 spec table과 비교해 최종 분류한다.

  1) spec table 로딩 — 데이터셋 클래스를 표에 붙이고 feature vocab 구성
  2) 학습 — train.mode = shared(동시) | separate(feature마다 따로)
  3) fuse  — fuse.mode = learned(가중치 자동 학습) | min_match(n개 이상 일치)
  4) 저장 — 최종 점수([N,C] 행렬 + predictions.csv) → Stage 5에서 다른 브랜치와 결합

실행: python src/run_attributes.py --config configs/05_attributes.yaml
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_utils import as_list, load_config_from_cli
from embeddings import plan_from_config


def _selected_features(features_cfg: dict, key: str) -> list[str]:
    """13개 feature 중 해당 플래그가 켜진 것만."""
    return [name for name, cfg in features_cfg.items() if (cfg or {}).get(key, False)]


def main():
    cfg = load_config_from_cli(__doc__)

    import torch

    from attributes.core import choose_device, log, read_manifest, seed_everything, write_json
    from attributes.fuse import (
        fuse_scores,
        learn_feature_weights,
        match_probability_tensor,
        save_score_matrix,
        scores_to_frame,
    )
    from attributes.spec_table import build_spec_table
    from attributes.train import train_attributes
    from attributes.visualize import visualize_run

    protocol = cfg["protocol"]
    d = cfg["data"]
    t = cfg["train"]
    f = cfg["fuse"]
    spec_cfg = cfg["spec"]
    features_cfg = cfg["features"]
    seed = int(t.get("seed", 42))
    seed_everything(seed)
    device = choose_device(cfg.get("device", "auto"))
    mode = str(t.get("mode", "separate"))          # separate | shared (05_attributes.yaml train.mode)
    if mode not in ("separate", "shared"):
        raise ValueError(f"train.mode must be separate or shared: {mode!r}")
    embed_plan = plan_from_config(cfg)             # 학습 종료 시 임베딩 저장 계획

    manifest = d["closed_manifest"] if protocol == "closed27_open15" else d["full_manifest"]
    out_root = Path(cfg["paths"]["output_dir"]) / cfg["output_subdir"]

    # ---- 1) spec table ----
    frames = {s: read_manifest(manifest, split=s) for s in ("train", "valid", "test")}
    labels = sorted(pd.concat(frames.values())["label"].unique().tolist())
    train_features = _selected_features(features_cfg, "train")
    table = build_spec_table(
        labels, spec_cfg["csv"],
        attributes=train_features,
        label_aliases=spec_cfg.get("label_aliases"),
        unknown_values=spec_cfg.get("unknown_values"),
        value_aliases=spec_cfg.get("value_aliases"),
        drop_constant=bool(spec_cfg.get("drop_constant", True)),
    )
    log(f"spec table: 데이터셋 {len(labels)}클래스 중 {len(table.classes)}개 매칭, "
        f"학습 feature {len(table.attributes)}개 {table.attributes}")
    manual = [l for l, m in table.match_method.items() if m == "dataset_label"]
    if manual:
        log(f"  dataset_label 열로 수동 매칭된 클래스 {len(manual)}개")
    if table.unmatched:
        log(f"  ⚠ 표에 못 붙은 데이터셋 클래스 {len(table.unmatched)}개: {table.unmatched}")
        log(f"    → spec_table.csv의 dataset_label 열에 폴더명을 적어 등록하세요")
    if table.unused_spec_rows:
        log(f"  표에만 있고 데이터에 없는 행 {len(table.unused_spec_rows)}개 "
            f"(사용 안 함): {table.unused_spec_rows}")
    if table.dropped_constant:
        log(f"  값이 1종뿐이라 제외: {table.dropped_constant}")
    unknown = {k: v for k, v in table.unknown_report().items() if v}
    if unknown:
        log(f"  unknown(정답 없음) 칸: {unknown}  ← 학습/평가에서 제외됨")

    # ---- fuse에 반영할 feature (학습된 것 중에서) ----
    fuse_wanted = [a for a in _selected_features(features_cfg, "use_in_fuse")
                   if a in table.attributes]
    if not fuse_wanted:
        raise ValueError("fuse에 반영할 feature가 없습니다 (features.*.use_in_fuse 확인)")

    summary_rows = []
    # 리스트가 아니어도 된다 — `backbone: cnn` 처럼 하나만 적으면 1회만 돈다
    for backbone_name in as_list(cfg["grid"]["backbone"]):
        entry = dict(cfg["backbones"][backbone_name])
        image_size = int(entry.get("image_size", 224))
        tag = f"{protocol}_{backbone_name}_{mode}"   # mode 가 붙어 separate/shared 가 서로 덮어쓰지 않는다
        run_dir = out_root / tag
        log(f"\n===== {tag} =====")

        # ---- 2) 학습: separate(feature마다 독립 모델) | shared(백본 공유 multi-head) ----
        run_cfg = cfg.get("run", {}) or {}
        requested = run_cfg.get("features", "all")
        run_features = (table.attributes if requested in ("all", None)
                        else as_list(requested))
        log(f"  이번 실행에서 학습할 feature {len(run_features)}개 ({mode}): {run_features}")
        trained = train_attributes(
            mode=mode,
            embed_plan=embed_plan,
            backbone_entry=entry,
            table=table,
            run_features=run_features,
            frames=frames,
            image_size=image_size,
            run_dir=run_dir,
            skip_existing=bool(run_cfg.get("skip_existing", True)),
            epochs=int(t["epochs"]),
            batch_size=int(t["batch_size"]),
            lr=float(t["lr"]),
            weight_decay=float(t.get("weight_decay", 0.05)),
            label_smoothing=float(t.get("label_smoothing", 0.1)),
            use_class_weights=bool(t.get("use_class_weights", False)),
            num_workers=int(t.get("num_workers", 4)),
            patience=int(t.get("patience", 12)),
            device=device,
        )

        # ---- 3) fuse: feature 확률 → 클래스 점수 ----
        # 이번 실행에서 확률을 얻은 feature 중 use_in_fuse가 켜진 것만 반영
        fuse_features = [a for a in fuse_wanted if a in trained["attributes"]]
        missing = [a for a in fuse_wanted if a not in trained["attributes"]]
        if missing:
            log(f"  ⚠ 모델이 없어 fuse에서 빠지는 feature: {missing} "
                f"(run.features로 학습하거나 use_in_fuse를 끄세요)")
        if not fuse_features:
            raise ValueError("fuse에 쓸 feature 모델이 하나도 없습니다")
        log(f"  최종 분류 반영 feature {len(fuse_features)}개: {fuse_features}")
        fuse_index = [table.attributes.index(a) for a in fuse_features]
        sub_target = table.target[:, fuse_index]
        weights = None
        if f["mode"] == "learned":
            # valid에서 feature 가중치를 학습 (test는 절대 쓰지 않는다)
            valid_matrix = match_probability_tensor(
                {a: trained["probabilities"]["valid"][a] for a in fuse_features},
                sub_target, fuse_features,
            )
            weights = learn_feature_weights(
                valid_matrix, trained["class_index"]["valid"], fuse_features,
                epochs=int(f.get("learn_epochs", 300)), lr=float(f.get("learn_lr", 0.05)), seed=seed,
            )
            log("  학습된 feature 가중치:")
            for name, w in sorted(zip(fuse_features, weights), key=lambda kv: -kv[1]):
                log(f"    {name:15s} {w:.4f}")
            write_json(run_dir / "feature_weights.json",
                       {n: float(w) for n, w in zip(fuse_features, weights)})

        for split in ("valid", "test"):
            probabilities = {a: trained["probabilities"][split][a] for a in fuse_features}
            result = fuse_scores(
                probabilities, sub_target, fuse_features,
                mode=f["mode"], weights=weights, min_match=int(f.get("min_match", 8)),
            )
            class_index = trained["class_index"][split]
            true_labels = [table.classes[i] for i in class_index]
            frame = scores_to_frame(result, table.classes, trained["paths"][split],
                                    true_labels, top_k=int(f.get("top_k", 2)))
            frame.to_csv(run_dir / f"predictions_{split}.csv", index=False)
            save_score_matrix(run_dir / f"scores_{split}.csv", result["scores"],
                              table.classes, trained["paths"][split])

            decided = ~result["rejected"]
            accuracy = float(frame["correct"].mean())
            accuracy_decided = float(frame.loc[decided, "correct"].mean()) if decided.any() else 0.0
            row = {
                "backbone": backbone_name, "mode": mode, "fuse_mode": f["mode"],
                "n_features": len(fuse_features), "split": split, "n": len(frame),
                "accuracy": accuracy,
                "accuracy_on_decided": accuracy_decided,
                "unknown_rate": float(result["rejected"].mean()),
                "mean_attribute_macro_f1": trained["metrics"][split]["mean_macro_f1"],
            }
            summary_rows.append(row)
            log(f"  [{split}] accuracy={accuracy:.4f} "
                f"(판정된 것만={accuracy_decided:.4f}, unknown={row['unknown_rate']:.3f}) "
                f"feature_macro_f1={row['mean_attribute_macro_f1']:.4f}")

            # ---- 시각화 (yaml visualize) ----
            viz_cfg = cfg.get("visualize", {}) or {}
            if viz_cfg.get("enabled", True):
                saved = visualize_run(
                    output_dir=run_dir,
                    split=split,
                    metrics=trained["metrics"][split],
                    probabilities=trained["probabilities"][split],
                    class_index=class_index,
                    table=table,
                    fuse_features=fuse_features,
                    scores=result["scores"],
                    match_counts=result["match_counts"],
                    classes=table.classes,
                    true_labels=true_labels,
                    pred_labels=frame["pred_label"].tolist(),
                    weights={n: float(w) for n, w in zip(fuse_features, weights)}
                             if weights is not None else None,
                    min_match=int(f.get("min_match", 8)) if f["mode"] == "min_match" else None,
                    which=viz_cfg.get("which",
                        ["performance", "weights", "confusions", "match", "score", "final"]),
                )
                if saved:
                    log(f"  [{split}] 시각화 {len(saved)}장 → {run_dir / 'visualization'}")

        write_json(run_dir / "spec_table.json", table.describe())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = pd.DataFrame(summary_rows)
    out_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_root / f"summary_{protocol}.csv", index=False)
    log(f"\nattributes 완료 → {out_root / f'summary_{protocol}.csv'}")
    with pd.option_context("display.width", 200):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
