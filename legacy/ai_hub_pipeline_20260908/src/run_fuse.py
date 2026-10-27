"""
Stage 5. Fuse 드라이버 (노션 6번)

세 브랜치의 [N, C] 점수를 하나로 합쳐 최종 분류한다.

  1) 각 브랜치 점수 로딩 → '클수록 좋음'으로 통일 → 행별 정규화
  2) 가중 결합 (manual 지정 또는 valid에서 자동 학습)
  3) open-set gate: metric이 open 클래스로 판정한 샘플은 metric만 사용
  4) threshold: 최고 점수가 기준 미만이면 __unknown__ (auto면 valid에서 결정)
  5) top1(기본) / top2 평가

실행:
  python src/run_fuse.py --config configs/06_fuse.yaml          # yaml 경로 사용
  python src/run_fuse.py --config configs/06_fuse.yaml \
      --metric      outputs/metric/closed27_open15_dinov2_supcon \
      --classifier  outputs/classifier/band_vit/closed27_open15_a8_l0.8_b6 \
      --attributes  outputs/attributes/closed27_open15_cnn_separate

브랜치 인자는 run 디렉토리 / best.pt / scores CSV 아무거나 받는다.
metric은 조합이 여러 개라 --metric-variant mean_cosine 으로 고르거나,
생략하면 그 run의 summary.csv에서 가장 좋은 조합을 자동으로 고른다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_utils import apply_overrides, as_list, load_config


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--protocol", choices=["closed27_open15", "full42"])
    parser.add_argument("--metric", help="metric run 디렉토리 / best.pt / scores CSV")
    parser.add_argument("--classifier", help="classifier run 디렉토리 / best.pt / scores CSV")
    parser.add_argument("--attributes", help="attributes run 디렉토리 / scores CSV")
    parser.add_argument("--metric-variant",
                        help="metric 조합 (예: mean_cosine). 생략하면 summary.csv에서 자동 선택")
    parser.add_argument("--labels-valid", help="valid 정답 라벨 CSV (path,label)")
    parser.add_argument("--labels-test", help="test 정답 라벨 CSV (path,label)")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[],
                        metavar="KEY=VALUE",
                        help="yaml 항목 덮어쓰기 (예: evaluation.top_k=2)")
    return parser.parse_args()


def _load_split(branch_cfg: dict, split: str, normalize_cfg: dict, log):
    """한 split에 대해 모든 브랜치 점수를 로딩·정규화·정렬."""
    from fuse.core import align_branches, load_branch_scores, normalize_scores

    loaded = []
    for name, cfg in branch_cfg.items():
        path = cfg["scores"].get(split)
        if not path:
            continue
        branch = load_branch_scores(
            name, path,
            higher_is_better=bool(cfg.get("higher_is_better", True)),
        )
        log(f"  [{split}] {name}: {len(branch)}장 × {len(branch.classes)}클래스 "
            f"({'클수록 좋음' if cfg.get('higher_is_better', True) else '작을수록 좋음 → 반전'})")
        loaded.append(branch)
    if not loaded:
        raise ValueError(f"{split}에 대한 브랜치 점수가 하나도 없습니다")

    paths, classes, aligned = align_branches(loaded)
    log(f"  [{split}] 정렬 결과: 공통 {len(paths)}장 × {len(classes)}클래스")

    # 브랜치마다 다루는 이미지가 다르면 교집합만 평가된다. 이때 정확도는
    # "더 쉬운 부분집합"에서 잰 값이라 브랜치 구성이 다른 실행끼리 비교하면 안 된다.
    widest = max((len(b) for b in loaded), default=0)
    if widest and len(paths) < widest:
        short = {b.name: len(b) for b in loaded if len(b) > len(paths)}
        log(f"  [{split}] WARNING 공통 {len(paths)}장 < 최대 {widest}장 — "
            f"교집합만 평가됩니다 (더 넓은 브랜치: {short}). "
            f"브랜치 구성이 다른 실행끼리 정확도를 직접 비교하지 마세요.")
    normalized = {
        name: normalize_scores(scores, normalize_cfg.get("method", "softmax"),
                               float(normalize_cfg.get("temperature", 1.0)))
        for name, scores in aligned.items()
    }
    return paths, classes, normalized


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.protocol:
        cfg["protocol"] = args.protocol
    apply_overrides(cfg, args.overrides)

    from fuse.resolve import resolve_branch
    from fuse.combine import (
        apply_openset_gate,
        apply_threshold,
        build_prediction_frame,
        combine,
        evaluate_rejection_quality,
        evaluate_topk,
        find_threshold_auto,
        learn_branch_weights,
    )
    from fuse.core import load_labels
    from fuse.visualize import visualize_run

    def log(msg):
        print(msg, flush=True)

    branch_cfg = cfg["branches"]

    # ---- enabled: false 인 브랜치는 제외 ----
    # 블록을 지워도 되지만, extends 상속에서는 부모의 항목을 지울 수 없고
    # "2브랜치 vs 3브랜치" 비교 자체가 실험 축이라 플래그로 끌 수 있게 한다.
    disabled = [n for n, e in branch_cfg.items() if not (e or {}).get("enabled", True)]
    for name in disabled:
        branch_cfg.pop(name)
    if disabled:
        log(f"enabled: false 로 제외된 브랜치: {disabled}")
    if not branch_cfg:
        raise ValueError("결합할 브랜치가 하나도 없습니다 (branches가 비었거나 전부 enabled: false)")

    # ---- 브랜치 입력 경로 확정 ----
    # 우선순위: CLI 인자 > yaml의 scores(파일 직접 지정) > yaml의 run(폴더에서 탐색)
    # 구체적인 지정이 이긴다 — scores를 적어 놓고 run 때문에 무시되면 알아채기 어렵다.
    cli_given = {"metric": args.metric, "classifier": args.classifier,
                 "attributes": args.attributes}
    for name in list(branch_cfg):
        entry = branch_cfg[name]
        explicit = entry.get("scores") or {}
        source = cli_given.get(name)
        origin = "CLI"
        if not source:
            if explicit.get("valid") and explicit.get("test"):
                if entry.get("run"):
                    log(f"[{name}] scores가 직접 지정돼 run은 무시합니다: {entry['run']}")
                log(f"[{name}] yaml scores: valid={explicit['valid']}")
                log(f"[{name}] yaml scores: test ={explicit['test']}")
                continue                   # yaml의 scores를 그대로 사용
            if explicit and not (explicit.get("valid") and explicit.get("test")):
                raise ValueError(
                    f"[{name}] scores 블록에는 valid와 test를 모두 적어야 합니다 "
                    f"(현재: {sorted(explicit)}). 한쪽만 있으면 fuse가 test로 "
                    f"가중치를 학습하게 됩니다."
                )
            source = entry.get("run")
            origin = "yaml run"
        if not source:
            raise ValueError(
                f"[{name}] 입력이 없습니다. yaml에 run 또는 scores를 적거나 "
                f"--{name} 로 넘기세요."
            )
        variant = args.metric_variant or entry.get("variant")
        entry.setdefault("scores", {})
        log(f"[{name}] {origin}: {source}")
        for split in ("valid", "test"):
            resolved, how = resolve_branch(name, source, split, metric_variant=variant)
            entry["scores"][split] = str(resolved)
            log(f"    {split:5s} → {how}")
    if args.labels_valid:
        cfg["labels"]["valid"] = args.labels_valid
    if args.labels_test:
        cfg["labels"]["test"] = args.labels_test

    weight_cfg = cfg["weights"]
    threshold_cfg = cfg["threshold"]
    eval_cfg = cfg["evaluation"]
    out_dir = Path(cfg["paths"]["output_dir"]) / cfg["output_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- open 클래스 목록 (gate 및 리포트용) ----
    open_classes = []
    open_manifest = cfg.get("openset", {}).get("open_manifest")
    if open_manifest and Path(open_manifest).is_file():
        open_classes = sorted(pd.read_csv(open_manifest)["label"].astype(str).unique().tolist())
    elif cfg.get("openset", {}).get("open_classes"):
        open_classes = list(cfg["openset"]["open_classes"])
    if open_classes:
        log(f"open 클래스 {len(open_classes)}개: {open_classes[:5]}"
            f"{'...' if len(open_classes) > 5 else ''}")

    # ---- valid: 가중치 학습 + threshold 자동 결정에만 사용 ----
    weights = dict(weight_cfg.get("manual", {}))
    threshold = None
    if weight_cfg["mode"] == "learned" or threshold_cfg["mode"] == "auto":
        log("\n=== valid (가중치/임계값 결정용) ===")
        v_paths, v_classes, v_scores = _load_split(branch_cfg, "valid", cfg["normalize"], log)
        v_labels = load_labels(cfg["labels"]["valid"])
        v_truth = [v_labels[p] for p in v_paths]
        class_to_idx = {c: i for i, c in enumerate(v_classes)}
        missing = sorted({t for t in v_truth if t not in class_to_idx})
        if missing:
            raise ValueError(f"valid 정답 라벨이 점수 컬럼에 없습니다: {missing}")
        v_true_index = np.asarray([class_to_idx[t] for t in v_truth])

        if weight_cfg["mode"] == "learned":
            weights = learn_branch_weights(
                v_scores, v_true_index,
                epochs=int(weight_cfg.get("learn_epochs", 400)),
                lr=float(weight_cfg.get("learn_lr", 0.05)),
                verbose=True,
            )
            log("  학습된 브랜치 가중치:")
            for name, w in sorted(weights.items(), key=lambda kv: -kv[1]):
                log(f"    {name:12s} {w:.4f}")

        if threshold_cfg["mode"] == "auto":
            v_fused = combine(v_scores, weights)
            if open_classes and "metric" in v_scores:
                v_fused, _ = apply_openset_gate(v_fused, v_scores["metric"], v_classes, open_classes)
            max_reject = threshold_cfg.get("max_reject_rate")
            threshold = find_threshold_auto(
                v_fused, v_true_index,
                max_reject_rate=float(max_reject) if max_reject is not None else None,
            )
            v_reject_rate = float((v_fused.max(axis=1) < threshold).mean())
            log(f"  자동 결정된 임계값: {threshold:.6f} "
                f"(valid에서 정답/오답 분리가 가장 잘 되는 지점, "
                f"valid reject율={v_reject_rate:.3f})")

    if weight_cfg["mode"] == "manual":
        log(f"\n수동 지정 가중치: {weights}")
    if threshold_cfg["mode"] == "manual":
        threshold = float(threshold_cfg["value"])
        log(f"수동 지정 임계값: {threshold}")
    elif threshold_cfg["mode"] == "none":
        threshold = None
        log("임계값 사용 안 함 (reject 없음)")

    # ---- test: 최종 결합 ----
    log("\n=== test (최종 결합) ===")
    paths, classes, scores = _load_split(branch_cfg, "test", cfg["normalize"], log)
    labels_map = load_labels(cfg["labels"]["test"])
    truth = [labels_map[p] for p in paths]

    fused = combine(scores, weights)
    pre_gate = fused.copy()                    # gate 효과 비교용
    gated = np.zeros(len(fused), dtype=bool)
    if open_classes:
        if "metric" not in scores:
            log("  ⚠ metric 브랜치가 없어 open-set gate를 건너뜁니다")
        else:
            fused, gated = apply_openset_gate(fused, scores["metric"], classes, open_classes)
            log(f"  open-set gate: {int(gated.sum())}/{len(gated)}장이 metric 단독 판정으로 전환")

    rejected = apply_threshold(fused, threshold)
    if threshold is not None:
        log(f"  threshold={threshold:.6f} → {int(rejected.sum())}/{len(rejected)}장 unknown 처리")

    # ---- 평가: top1 기본, top2는 옵션 ----
    results = {}
    top_k_list = as_list(eval_cfg.get("top_k", [1])) or [1]   # `top_k: 1` 스칼라도 허용
    for k in top_k_list:
        results[f"top{k}"] = evaluate_topk(fused, classes, truth, rejected, top_k=int(k))
        r = results[f"top{k}"]
        log(f"  top{k}: accuracy={r['accuracy']:.4f} "
            f"(판정된 것만={r['accuracy_on_decided']:.4f}, reject={r['reject_rate']:.3f})")

    frame = build_prediction_frame(paths, fused, classes, truth, rejected, gated,
                                   top_k=max(int(k) for k in top_k_list))
    frame.to_csv(out_dir / "predictions.csv", index=False)
    score_frame = pd.DataFrame(fused, columns=classes)
    score_frame.insert(0, "path", paths)
    score_frame.to_csv(out_dir / "scores.csv", index=False)

    # ---- open-set 거부 성능 (threshold와 무관한 지표) ----
    reject_cfg = cfg.get("reject_metrics", {}) or {}
    if reject_cfg.get("enabled", False) and open_classes:
        is_unknown = np.asarray([t in set(open_classes) for t in truth])
        rejection = evaluate_rejection_quality(
            fused, is_unknown,
            which=as_list(reject_cfg.get("scores", ["min_dist", "margin", "dist_ent"])))
        results["rejection"] = rejection
        if rejection.get("available"):
            parts = "  ".join(f"{n}={v['auroc']:.3f}" for n, v in rejection["scores"].items())
            log(f"  거부 AUROC — {parts}  (최고: {rejection['best_score']})")

    # ---- 시각화 (yaml visualize) ----
    viz_cfg = cfg.get("visualize", {}) or {}
    if viz_cfg.get("enabled", True):
        class_to_idx = {c: i for i, c in enumerate(classes)}
        saved = visualize_run(
            output_dir=out_dir,
            branch_scores=scores,
            fused=pre_gate,
            gated_scores=fused,
            true_index=np.asarray([class_to_idx[t] for t in truth]),
            is_open=np.asarray([t in set(open_classes) for t in truth]),
            classes=classes,
            true_labels=truth,
            pred_labels=frame["pred_label"].tolist(),
            weights=weights,
            weight_mode=weight_cfg["mode"],
            threshold=threshold,
            log=log,
            which=viz_cfg.get("which",
                ["weights", "comparison", "openset", "score", "confusion"]),
        )
        if saved:
            log(f"  시각화 {len(saved)}장 → {out_dir / 'visualization'}")

    summary = {
        "weights": weights,
        "weight_mode": weight_cfg["mode"],
        "threshold": threshold,
        "threshold_mode": threshold_cfg["mode"],
        "normalize": cfg["normalize"],
        "open_classes": open_classes,
        "openset_gated": int(gated.sum()),
        "branches": {n: c["scores"].get("test") for n, c in branch_cfg.items()},
        "results": results,
    }
    import json

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    topk_rows = [{"metric": k, **v} for k, v in results.items() if k.startswith("top")]
    pd.DataFrame(topk_rows).to_csv(out_dir / "summary.csv", index=False)
    log(f"\nfuse 완료 → {out_dir}")


if __name__ == "__main__":
    main()
