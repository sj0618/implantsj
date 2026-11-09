"""Prototype 갤러리 평가 — 두 프로토콜 + 옵션들.

closed27_open15 (노션 <1> 실험):
  - 임베딩 학습은 closed 27클래스로만 (data leakage 방지)
  - 갤러리: closed train(27) + open support(15) → 42-way prototype
  - query: closed test + open test
  - 거리 파라미터(공분산/스펙트럼)도 closed train 임베딩으로만 fit
  - 결과: 전체 / closed-only / open-only 지표 + 예측 CSV(score 포함 → Fuse 입력)

full42 (노션 <2> 실험):
  - 42클래스 train으로 prototype을 만들고 test를 분류

옵션 (전부 yaml):
  eval.transform      none / l2norm / pca
  eval.distances      cosine / euclidean / mahalanobis / fractional / centered_angular
  eval.openset        none / threshold — score가 임계를 넘으면 unknown 판정
  eval.duplicate_free true면 train과 파일 내용이 같은 query를 제외한 지표도 산출
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .core import (
    as_list,
    compute_metrics,
    create_loader,
    load_manifest,
    log,
    save_confusion_matrix,
    save_predictions,
    write_json,
)
from .openset import evaluate_rejection
from .prototypes import build_prototypes
from .reference import audit_reference_leakage, load_reference_prototypes, scan_reference_dir
from .scoring import build_scorer, build_transform
from .visualize import visualize_run
from .train import extract_embeddings

UNKNOWN_LABEL = "__unknown__"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def embed_split(model, manifest, split, image_size, batch_size, num_workers, device, source):
    """manifest의 한 split 임베딩 → (features, str_labels, paths)."""
    df = load_manifest(manifest, split=split)
    classes = sorted(df["label"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    loader = create_loader(manifest, split, class_to_idx, image_size, batch_size, num_workers,
                           train=False, return_path=True)
    z, y, paths = extract_embeddings(model, loader, device, source=source)
    return z, np.asarray(classes, dtype=object)[y].astype(str), list(paths)


def _resolve_threshold(scores: np.ndarray, openset_cfg: dict) -> float | None:
    """openset.mode에 따라 unknown 판정 임계값 결정 (score는 작을수록 가까움)."""
    mode = (openset_cfg or {}).get("mode", "none")
    if mode == "none":
        return None
    if mode != "threshold":
        raise ValueError("eval.openset.mode must be none or threshold")
    if openset_cfg.get("value") is not None:
        return float(openset_cfg["value"])
    # value가 없으면 query score 분포의 percentile로 자동 결정
    percentile = float(openset_cfg.get("percentile", 95))
    return float(np.percentile(scores, percentile))


def _metric_block(true_labels, pred_labels, classes: Sequence[str]) -> dict:
    """unknown 예측을 포함해 지표 계산 (unknown은 항상 오답으로 집계)."""
    names = list(classes) + [UNKNOWN_LABEL]
    index = {c: i for i, c in enumerate(names)}
    t = np.asarray([index[x] for x in true_labels])
    p = np.asarray([index[x] for x in pred_labels])
    m = compute_metrics(t, p, names)
    return {k: m[k] for k in ("accuracy", "macro_f1", "weighted_f1")}


def evaluate_gallery(
    *,
    model,
    protocol: str,
    data_cfg: dict,
    eval_cfg: dict,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device,
    embedding_source: str,
    output_dir: str | Path,
    seed: int = 42,
) -> pd.DataFrame:
    """prototype 방법 × 거리 전체 조합을 평가하고 summary DataFrame 반환."""
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    def embed(manifest, split):
        return embed_split(model, manifest, split, image_size, batch_size, num_workers, device, embedding_source)

    # ---- 임베딩 추출 + 갤러리/쿼리 구성 ----
    if protocol == "closed27_open15":
        closed_manifest, open_manifest = data_cfg["closed_manifest"], data_cfg["open_manifest"]
        fit_z, fit_y, fit_paths = embed(closed_manifest, "train")     # 거리 파라미터 fit용 (closed만)
        open_z, open_y, open_paths = embed(open_manifest, "train")    # open 갤러리 등록용 support
        cq_z, cq_y, cq_p = embed(closed_manifest, "test")
        oq_z, oq_y, oq_p = embed(open_manifest, "test")

        closed_classes = sorted(set(fit_y.tolist()))
        open_classes = sorted(set(open_y.tolist()))
        overlap = set(closed_classes) & set(open_classes)
        if overlap:
            raise ValueError(f"closed/open 라벨이 겹칩니다 (leakage): {sorted(overlap)}")
        classes = closed_classes + open_classes
        gallery_z, gallery_y = np.concatenate([fit_z, open_z]), np.concatenate([fit_y, open_y])
        gallery_paths = fit_paths + open_paths
        # Stage 5 fuse가 가중치·임계값을 valid에서 정하므로 두 split 모두 평가한다
        queries = {"test": (np.concatenate([cq_z, oq_z]),
                            np.concatenate([cq_y, oq_y]), cq_p + oq_p)}
        try:
            vc_z, vc_y, vc_p = embed(closed_manifest, "valid")
            vo_z, vo_y, vo_p = embed(open_manifest, "valid")
            queries["valid"] = (np.concatenate([vc_z, vo_z]),
                                np.concatenate([vc_y, vo_y]), vc_p + vo_p)
        except Exception as exc:
            log(f"valid split 평가 생략: {type(exc).__name__}: {exc}")
        open_set = set(open_classes)
    elif protocol == "full42":
        full_manifest = data_cfg["full_manifest"]
        gallery_z, gallery_y, gallery_paths = embed(full_manifest, "train")
        queries = {"test": embed(full_manifest, "test")}
        try:
            queries["valid"] = embed(full_manifest, "valid")
        except Exception as exc:
            log(f"valid split 평가 생략: {type(exc).__name__}: {exc}")
        classes = sorted(set(gallery_y.tolist()))
        fit_z, fit_y = gallery_z, gallery_y
        open_set = set()
    else:
        raise ValueError(f"unknown protocol: {protocol}")

    # ---- [1] 임베딩 변환 (none / l2norm / pca) — fit 데이터로 학습 후 전부 적용 ----
    transform = build_transform(eval_cfg, seed).fit(fit_z, fit_y, classes)
    fit_zt = transform.apply(fit_z)
    gallery_zt = transform.apply(gallery_z)
    queries_t = {s: (transform.apply(z), y, p) for s, (z, y, p) in queries.items()}
    log(f"embedding transform: {transform.describe()} → "
        f"dim={queries_t['test'][0].shape[1]}, splits={sorted(queries_t)}")

    # ---- duplicate-free 마스크 (train과 파일 내용이 같은 query 제외) ----
    dup_free_mask = None
    if eval_cfg.get("duplicate_free", False):
        gallery_hashes = {_sha256(p) for p in gallery_paths if p}
        test_paths = queries_t["test"][2]
        dup_free_mask = np.asarray([_sha256(p) not in gallery_hashes for p in test_paths])
        log(f"duplicate-free: query {len(test_paths)}장 중 {int((~dup_free_mask).sum())}장이 "
            f"gallery와 내용 동일 → 별도 지표 산출")
        if not dup_free_mask.any():
            log("WARNING 모든 query가 gallery와 중복 — duplicate_free 지표를 건너뜁니다")
            dup_free_mask = None

    proto_cfg = eval_cfg.get("prototype", {}) or {}
    # 리스트가 아니어도 된다 — `methods: mean`, `distances: cosine` 처럼 하나만 적어도 된다
    methods = as_list(proto_cfg.get("methods", ["mean"]))
    distances = as_list(eval_cfg.get("distances", ["cosine"]))
    openset_cfg = eval_cfg.get("openset", {}) or {}

    # ---- 대표 이미지 prototype (method에 reference가 있을 때만) ----
    reference = None
    if "reference" in methods:
        reference_dir = proto_cfg.get("reference_dir")
        if not reference_dir:
            raise ValueError(
                "prototype methods에 reference가 있는데 "
                "eval.prototype.reference_dir 이 비어 있습니다"
            )
        mapped = scan_reference_dir(reference_dir, classes,
                                    extensions=proto_cfg.get("reference_extensions"))
        # 대표 이미지가 학습/평가 데이터에 섞여 있으면 prototype이 정답을 미리 본 셈이 된다
        if proto_cfg.get("reference_audit", True):
            audit_reference_leakage(
                mapped,
                {"gallery": gallery_paths,
                 **{f"query_{s}": p for s, (_, _, p) in queries_t.items()}},
                output_dir=out,
                strict=bool(proto_cfg.get("reference_audit_strict", True)),
            )
        from .reference import embed_reference

        raw = embed_reference(model, mapped, image_size=image_size, batch_size=batch_size,
                              num_workers=num_workers, device=device, source=embedding_source)
        # query와 같은 변환을 거쳐야 같은 공간에서 비교된다
        reference = {k: transform.apply(v) for k, v in raw.items()}

    spherical_cfg = bool(proto_cfg.get("cosine_spherical_mean", False))

    def make_prototypes(method: str, spherical: bool):
        return build_prototypes(
            gallery_zt, gallery_y, classes, method,
            seed=seed,
            ss_count=int(proto_cfg.get("ss_count", 5)),
            kmeans_clusters=int(proto_cfg.get("kmeans_clusters", 3)),
            reference=reference,
            reference_reduce=proto_cfg.get("reference_reduce", "mean"),
            spherical_mean=spherical,
            log=log,
        )

    summary_rows = []
    for method in methods:
        # 구면 평균은 cosine 거리에서만 의미가 있어, 필요할 때만 따로 만든다
        protos_plain = make_prototypes(method, False)
        protos_spherical = (make_prototypes(method, True)
                            if spherical_cfg and method in ("mean", "ss", "reference") else None)
        for distance in distances:
            # cosine에 구면 평균을 켜면 산술평균 결과를 **대체하지 않고** 둘 다 낸다.
            # (대체하면 summary의 distance 열은 그대로 cosine이라 어느 쪽 수치인지 알 수 없다)
            variants = [(protos_plain, "")]
            if protos_spherical is not None and distance == "cosine":
                variants.append((protos_spherical, "_spherical"))
            # scorer는 prototype이 아니라 거리에만 의존하므로 variant마다 다시 fit하지 않는다
            # (mahalanobis 공분산 추정은 비싸다)
            scorer = build_scorer(distance, eval_cfg, seed).fit(fit_zt, fit_y, classes)
            log(f"eval: prototype={method} scorer={scorer.describe()}"
                + (f" (+{len(variants) - 1} variant)" if len(variants) > 1 else ""))
            for protos, suffix in variants:
                tag = f"{method}_{distance}{suffix}"

                for split, (query_zt, query_y, query_paths) in sorted(queries_t.items()):
                    is_test = split == "test"
                    matrix = scorer.score(query_zt, protos)             # [N, C]
                    best = matrix.argmin(axis=1)
                    pred = np.asarray(classes, dtype=object)[best].astype(str)
                    best_score = matrix[np.arange(len(best)), best]

                    # ---- [4] open-set threshold: 임계 초과면 unknown ----
                    threshold = _resolve_threshold(best_score, openset_cfg)
                    rejected = best_score > threshold if threshold is not None else np.zeros(len(pred), bool)
                    final_pred = np.where(rejected, UNKNOWN_LABEL, pred)

                    # ---- 지표: 전체 / closed / open (+ duplicate-free) ----
                    is_open = np.asarray([t in open_set for t in query_y])
                    blocks = {"overall": _metric_block(query_y, final_pred, classes)}
                    for name, mask in (("closed", ~is_open), ("open", is_open)):
                        if mask.any():
                            blocks[name] = _metric_block(query_y[mask], final_pred[mask], classes)
                    if is_test and dup_free_mask is not None:
                        blocks["duplicate_free"] = _metric_block(
                            query_y[dup_free_mask], final_pred[dup_free_mask], classes
                        )

                    rows = [
                        {
                            "path": p, "true_label": t, "pred_label": str(pr),
                            "score": float(s),                      # 작을수록 가까움 (Fuse threshold 입력)
                            "rejected_unknown": bool(r),
                            "true_is_open_class": t in open_set,
                            "correct": str(pr) == t,
                        }
                        for p, t, pr, s, r in zip(query_paths, query_y, final_pred, best_score, rejected)
                    ]
                    save_predictions(out / f"predictions_{split}_{tag}.csv", rows)
                    # Stage 5 fuse 입력: [N, C] 거리 행렬 (작을수록 가까움 — fuse에서 방향 반전)
                    score_frame = pd.DataFrame(matrix, columns=list(classes))
                    score_frame.insert(0, "path", list(query_paths))
                    score_frame.to_csv(out / f"scores_{split}_{tag}.csv", index=False)
                    metrics_payload = {
                        "protocol": protocol,
                        "split": split,
                        "prototype": method,
                        "embedding_source": embedding_source,
                        "transform": transform.describe(),
                        "scorer": scorer.describe(),
                        "openset_threshold": threshold,
                        "unknown_rate": float(rejected.mean()),
                        **blocks,
                    }

                    # ---- open-set 거부 성능 (AUROC / FPR@95) ----
                    # (metrics json은 거부 지표까지 담은 뒤 저장한다)
                    # open 15는 임베딩 학습에 쓰이지 않았으므로 모델에겐 처음 보는 클래스다.
                    reject_cfg = eval_cfg.get("reject_metrics", {}) or {}
                    if reject_cfg.get("enabled", False) and open_set:
                        rejection = evaluate_rejection(
                            matrix, is_open,
                            which=as_list(reject_cfg.get(
                                "scores", ["min_dist", "margin", "dist_ent"])),
                            temperature=float(reject_cfg.get("temperature", 1.0)),
                        )
                        write_json(out / f"rejection_{split}_{tag}.json", rejection)
                        if rejection.get("available"):
                            parts = "  ".join(
                                f"{n}={v['auroc']:.3f}" for n, v in rejection["scores"].items())
                            log(f"    [{split}] 거부 AUROC — {parts}  "
                                f"(최고: {rejection['best_score']})")
                            if is_test:
                                for name, value in rejection["scores"].items():
                                    blocks.setdefault("reject", {})[f"{name}_auroc"] = value["auroc"]
                                    blocks["reject"][f"{name}_fpr95"] = value["fpr_at_95tpr"]

                    write_json(out / f"metrics_{split}_{tag}.json",
                               {**metrics_payload, **{k: v for k, v in blocks.items()}})

                    if not is_test:
                        continue        # valid는 fuse 입력용 점수까지만

                    # confusion matrix는 unknown 제외(실제 클래스 판정만)한 축으로 저장
                    keep = ~rejected
                    if keep.any():
                        idx = {c: i for i, c in enumerate(classes)}
                        save_confusion_matrix(
                            out / f"confusion_{tag}.csv", out / f"confusion_{tag}.png",
                            [idx[t] for t in query_y[keep]], [idx[str(p)] for p in final_pred[keep]], classes,
                        )

                    row = {"prototype": method + suffix, "distance": distance,
                           "variant": tag, "unknown_rate": float(rejected.mean())}
                    for part, values in blocks.items():
                        for key, value in values.items():
                            row[f"{part}_{key}"] = value
                    summary_rows.append(row)

                    # ---- 시각화 (yaml eval.visualize) ----
                    viz_cfg = eval_cfg.get("visualize", {}) or {}
                    if viz_cfg.get("enabled", False):
                        try:
                            visualize_run(
                                query_features=query_zt,
                                query_labels=query_y,
                                query_paths=query_paths,
                                protos=protos if viz_cfg.get("show_prototypes", True) else None,
                                classes=classes,
                                open_classes=sorted(open_set),
                                output_dir=out / "visualization",
                                tag=tag,
                                prototype_method=method,
                                reference_labels=list(reference) if reference else [],
                                reducer=viz_cfg.get("reducer", "tsne"),
                                max_samples=int(viz_cfg.get("max_samples", 2000)),
                                seed=seed,
                                pca_dim=int(viz_cfg.get("pca_dim", 50)),
                                perplexity=float(viz_cfg.get("perplexity", 30.0)),
                                n_neighbors=int(viz_cfg.get("n_neighbors", 15)),
                                min_dist=float(viz_cfg.get("min_dist", 0.1)),
                                best_scores=best_score,
                                correct=np.asarray([str(p) == t for p, t in zip(final_pred, query_y)]),
                            )
                        except Exception as exc:
                            # 시각화 실패가 평가 전체를 막지 않게 한다
                            log(f"    시각화 건너뜀: {type(exc).__name__}: {exc}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "summary.csv", index=False)
    return summary
