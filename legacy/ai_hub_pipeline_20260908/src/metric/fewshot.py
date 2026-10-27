"""Few-shot target adaptation — 별개 실험.

기본 평가(evaluate.py)는 open 클래스를 train split 전체로 갤러리에 등록한다.
이 실험은 "새 임플란트 시스템을 몇 장만 등록해도 인식되는가"를 본다:
클래스마다 support를 n장(1/5/10)만 뽑아 prototype을 만들고 나머지를 query로 쓴다.
support 뽑기를 여러 seed로 반복해 평균±표준편차를 보고한다 (재현성 검증).

출처: implantsj tools/evaluate_covaware_target_pool_5shot.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .core import compute_metrics, log, save_predictions, write_json
from .evaluate import embed_split
from .prototypes import build_prototypes
from .scoring import build_scorer, build_transform


def _episode(labels: np.ndarray, classes: Sequence[str], shots: int, rng) -> np.ndarray:
    """클래스마다 shots장을 support로 뽑아 인덱스 반환."""
    picks = []
    for c in classes:
        pool = np.flatnonzero(labels == c)
        if len(pool) <= shots:
            raise ValueError(
                f"클래스 {c!r}의 샘플이 {len(pool)}장뿐이라 {shots}-shot support를 뽑으면 query가 없습니다"
            )
        picks.append(rng.choice(pool, shots, replace=False))
    return np.concatenate(picks)


def run_fewshot(
    *,
    model,
    data_cfg: dict,
    eval_cfg: dict,
    shots_list: Sequence[int],
    seeds: Sequence[int],
    image_size: int,
    batch_size: int,
    num_workers: int,
    device,
    embedding_source: str,
    output_dir: str | Path,
) -> pd.DataFrame:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # target pool: 등록 대상 클래스의 모든 이미지를 한 풀로 모아 support/query를 나눈다
    manifest = data_cfg["target_manifest"]
    pool_splits = data_cfg.get("target_splits", ["train", "test"])
    chunks = [embed_split(model, manifest, s, image_size, batch_size, num_workers, device, embedding_source)
              for s in pool_splits]
    features = np.concatenate([c[0] for c in chunks])
    labels = np.concatenate([c[1] for c in chunks])
    paths = [p for c in chunks for p in c[2]]
    classes = sorted(set(labels.tolist()))
    log(f"few-shot pool: {len(features)}장 / {len(classes)}클래스 (splits={pool_splits})")

    proto_cfg = eval_cfg.get("prototype", {}) or {}
    methods = proto_cfg.get("methods", ["mean"])
    distances = eval_cfg.get("distances", ["cosine"])

    rows = []
    for shots in shots_list:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            support_idx = _episode(labels, classes, int(shots), rng)
            query_mask = np.ones(len(labels), bool)
            query_mask[support_idx] = False

            sup_z, sup_y = features[support_idx], labels[support_idx]
            qry_z, qry_y = features[query_mask], labels[query_mask]
            qry_paths = [p for p, keep in zip(paths, query_mask) if keep]

            # 변환/거리 파라미터는 support만으로 fit (query 정보 누출 방지)
            transform = build_transform(eval_cfg, seed).fit(sup_z, sup_y, classes)
            sup_zt, qry_zt = transform.apply(sup_z), transform.apply(qry_z)

            for method in methods:
                protos = build_prototypes(
                    sup_zt, sup_y, classes, method, seed=seed,
                    ss_count=int(proto_cfg.get("ss_count", 5)),
                    kmeans_clusters=int(proto_cfg.get("kmeans_clusters", 3)),
                )
                for distance in distances:
                    scorer = build_scorer(distance, eval_cfg, seed).fit(sup_zt, sup_y, classes)
                    matrix = scorer.score(qry_zt, protos)
                    best = matrix.argmin(axis=1)
                    pred = np.asarray(classes, dtype=object)[best].astype(str)

                    idx = {c: i for i, c in enumerate(classes)}
                    m = compute_metrics([idx[t] for t in qry_y], [idx[p] for p in pred], classes)
                    rows.append({
                        "shots": int(shots), "seed": int(seed),
                        "prototype": method, "distance": distance,
                        "n_query": int(query_mask.sum()),
                        "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                        "weighted_f1": m["weighted_f1"],
                    })
                    log(f"  shots={shots} seed={seed} {method}/{distance}: "
                        f"acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}")

                    if seed == seeds[0]:   # 대표 seed의 예측만 저장 (용량 절약)
                        save_predictions(
                            out / f"predictions_{shots}shot_{method}_{distance}.csv",
                            [{"path": p, "true_label": t, "pred_label": str(pr),
                              "score": float(s), "correct": str(pr) == t}
                             for p, t, pr, s in zip(qry_paths, qry_y, pred,
                                                    matrix[np.arange(len(best)), best])],
                        )

    detail = pd.DataFrame(rows)
    detail.to_csv(out / "fewshot_runs.csv", index=False)

    # seed 평균±표준편차 (재현성 검증)
    summary = (
        detail.groupby(["shots", "prototype", "distance"])
        .agg(seeds=("seed", "count"),
             accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
             macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"))
        .reset_index()
    )
    summary.to_csv(out / "fewshot_summary.csv", index=False)
    write_json(out / "config_used.json", {
        "shots": list(shots_list), "seeds": list(seeds),
        "prototype_methods": list(methods), "distances": list(distances),
        "embedding_source": embedding_source,
    })
    return summary
