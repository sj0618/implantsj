#!/usr/bin/env python3
"""
미학습 제조사 평가를 Leave-One-Out 으로 다시 계산한다.

  python loo_eval.py --cond M3 M5 M7
  python loo_eval.py --all

왜 필요한가
-----------
E3 는 "9개로 학습한 모델이 closed 4개까지 포함해 13개를 분류"하는 시나리오다.
그런데 지금은 closed 86장 중 **test 13장만** 평가에 쓰고 나머지 73장은
프로토타입 재료로만 쓰인다. 재실행 변동이 ±0.09 에 달해 판정이 불가능했다.

핵심은 **M3/M5/M7 은 closed 4개를 학습에 전혀 쓰지 않았다**는 점이다.
모델 입장에서 closed 의 train/val/test 는 전부 똑같이 처음 보는 그림이다.
따라서 나누어 둘 이유가 없고, 자기 자신만 빼고 프로토타입을 만들면
**86장 전부를 평가에 쓸 수 있다** (6.6배).

  nucleoss 1번 평가 → 나머지 33장으로 대표 벡터
  nucleoss 2번 평가 → 나머지 33장으로 대표 벡터
  ...

자기 자신이 대표 벡터에 안 들어가므로 누수가 없고,
대표 벡터가 24장이 아닌 33장으로 만들어져 오히려 더 정확하다.

open 9개에는 적용할 수 없다 — 모델이 학습에 썼으므로 train/val 이미지는
외운 것이라 평가에 쓸 수 없다. open 은 그대로 test split 만 쓴다.

대표 지표
---------
**미학습 recall** 을 대표값으로 삼는다. 평가 집합이 open test 182장 +
closed 86장으로 바뀌면서 클래스 구성비가 달라지는데, precision 은 구성비에
따라 변하지만 recall 은 변하지 않기 때문이다. macro-F1 도 함께 기록하되
기존 E3 와 구성비가 달라 직접 비교하지 않는다.

거부(rejection) — 별개의 질문
-----------------------------
LOO 분류는 "미학습 브랜드에게 **대표 벡터를 쥐어주면** 자기 것을 찾아가나"를
묻는다(13지선다). 실전에서 먼저 필요한 건 다른 질문이다 —
**"애초에 아는 9개가 아니라는 걸 아나."**

이건 대표 벡터가 필요 없다. 아는 9개까지의 거리만 재서 "다 멀면 모름"으로
판정하면 된다. closed 프로토타입을 안 만드므로 fold 도 필요 없고,
**86장 전부를 곧바로 unknown 표본으로 쓴다.**

  known   = open test 182장
  unknown = closed 전체 86장
  점수    = 가장 가까운 open 프로토타입까지의 거리 (멀수록 unknown)
  → AUROC, FPR@95%TPR
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from dataset import load_split
from metrics import distance_matrix, per_class_prf
from prototype import EMB_ROOT, l2_normalize, load_embeddings

_ROOT = Path(__file__).parent
OUT_CSV = _ROOT / "results_loo.csv"
OUT_JSON = _ROOT / "results_loo_per_class.json"

# LOO 를 쓸 수 있는 조건 = closed 4개를 학습에 쓰지 않은 조건
LOO_CONDS = ("M1", "M3", "M5", "M7")
METRICS = ("l2", "cosine")

FIELDS = ["cond", "metric", "n_open_test", "n_closed_loo",
          "unseen_recall", "unseen_macro_f1", "macro_f1_13", "top1_13",
          "reject_auroc", "reject_fpr95"]


def is_loo_cond(cond):
    """M5b 처럼 뒤에 접미사가 붙은 변형도 허용한다 (M5b → M5)."""
    return cond.rstrip("abcdefghijklmnopqrstuvwxyz") in LOO_CONDS


def auroc(pos, neg):
    """P(pos > neg). pos=unknown 점수, neg=known 점수. 동점은 0.5로 센다."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # 동점은 평균 순위로 보정
    s = allv[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n_p, n_n = len(pos), len(neg)
    return float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def fpr_at_tpr(pos, neg, target_tpr=0.95):
    """unknown 을 target_tpr 만큼 잡을 때, known 을 얼마나 잘못 버리는가."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    thr = np.quantile(pos, 1 - target_tpr)   # 이 값 이상이면 unknown 판정
    return float((neg >= thr).mean())


def _prep(vecs, metric):
    """cosine 은 정규화한 뒤 평균 내야 노름 큰 샘플이 방향을 독점하지 않는다."""
    return l2_normalize(vecs) if metric == "cosine" else vecs


def _finish(p, metric):
    return l2_normalize(p) if metric == "cosine" else p


def gather(cond, meta, emb_root=EMB_ROOT):
    """→ (open train, open test, closed 전체) 각각 (vecs, labels)"""
    parts = {s: load_embeddings(cond, s, emb_root) for s in ("train", "val", "test")}
    open_c = set(meta["open_classes"])
    closed_c = set(meta["closed_classes"])

    tr_v, tr_l, _ = parts["train"]
    te_v, te_l, _ = parts["test"]

    m = np.isin(tr_l, list(open_c))
    open_train = (tr_v[m], tr_l[m])
    m = np.isin(te_l, list(open_c))
    open_test = (te_v[m], te_l[m])

    cv, cl = [], []
    for s in ("train", "val", "test"):
        v, l, _ = parts[s]
        m = np.isin(l, list(closed_c))
        cv.append(v[m])
        cl.append(l[m])
    closed_all = (np.concatenate(cv), np.concatenate(cl))
    return open_train, open_test, closed_all


def evaluate_loo(cond, meta, metric, emb_root=EMB_ROOT):
    classes = sorted(meta["classes"])
    closed = sorted(meta["closed_classes"])
    idx = {c: i for i, c in enumerate(classes)}

    (otr_v, otr_l), (ote_v, ote_l), (cl_v, cl_l) = gather(cond, meta, emb_root)

    # --- 클래스별 합과 개수. LOO 는 여기서 자기 자신만 빼면 된다.
    sums = np.zeros((len(classes), otr_v.shape[1]), dtype=np.float64)
    counts = np.zeros(len(classes), dtype=np.int64)
    for c in classes:
        if c in closed:                       # closed: 전체(train+val+test)
            v = _prep(cl_v[cl_l == c], metric)
        else:                                 # open: train 만 (나머지는 오염)
            v = _prep(otr_v[otr_l == c], metric)
        if len(v) == 0:
            raise ValueError(f"클래스 {c} 의 벡터가 없다.")
        sums[idx[c]] = v.sum(0)
        counts[idx[c]] = len(v)

    protos_full = np.stack([_finish((sums[i] / counts[i]).astype(np.float32), metric)
                            for i in range(len(classes))])

    # --- 평가 집합: open test (held-out) + closed 전체 (LOO)
    X = np.concatenate([ote_v, cl_v])
    y = np.array([idx[l] for l in np.concatenate([ote_l, cl_l])])
    is_loo = np.concatenate([np.zeros(len(ote_l), bool), np.ones(len(cl_l), bool)])

    dist = distance_matrix(X, protos_full, metric)

    # --- closed 샘플은 자기 클래스 프로토타입만 자기 자신을 빼고 다시 계산
    Xp = _prep(X, metric)
    for i in np.where(is_loo)[0]:
        c = y[i]
        n = counts[c]
        if n < 2:
            raise ValueError(f"클래스 {classes[c]} 가 1장뿐이라 LOO 불가")
        p = _finish(((sums[c] - Xp[i]) / (n - 1)).astype(np.float32), metric)
        dist[i, c] = distance_matrix(X[i:i + 1], p[None], metric)[0, 0]

    pred = dist.argmin(1)

    # --- 지표
    prf = per_class_prf(y, pred, len(classes))
    closed_idx = [idx[c] for c in closed]
    unseen_recall = float(np.mean([prf[c]["recall"] for c in closed_idx]))
    unseen_f1 = float(np.mean([prf[c]["f1"] for c in closed_idx]))

    # --- 거부: 아는 9개까지의 최소 거리만 쓴다 (closed 프로토타입 미사용)
    open_only = np.array([idx[c] for c in sorted(meta["open_classes"])])
    s_known = distance_matrix(ote_v, protos_full[open_only], metric).min(1)
    s_unknown = distance_matrix(cl_v, protos_full[open_only], metric).min(1)

    return {
        "cond": cond, "metric": metric,
        "n_open_test": int(len(ote_l)), "n_closed_loo": int(len(cl_l)),
        "unseen_recall": round(unseen_recall, 6),
        "unseen_macro_f1": round(unseen_f1, 6),
        "macro_f1_13": round(float(np.mean([prf[c]["f1"] for c in range(len(classes))])), 6),
        "top1_13": round(float((y == pred).mean()), 6),
        "reject_auroc": round(auroc(s_unknown, s_known), 6),
        "reject_fpr95": round(fpr_at_tpr(s_unknown, s_known), 6),
        "_per_class": {classes[c]: {k: (round(v, 6) if isinstance(v, float) else v)
                                    for k, v in prf[c].items()}
                       for c in range(len(classes))},
        "_scores": {"known": s_known.tolist(), "unknown": s_unknown.tolist()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="*", default=None, help=f"{list(LOO_CONDS)}")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--emb-root", default=EMB_ROOT)
    args = ap.parse_args()

    emb_root = Path(args.emb_root)
    if args.all:
        conds = [c for c in LOO_CONDS if (emb_root / c / "train.npz").is_file()]
    elif args.cond:
        conds = args.cond
    else:
        raise SystemExit("--cond 또는 --all 이 필요하다.")

    bad = [c for c in conds if not is_loo_cond(c)]
    if bad:
        raise SystemExit(
            f"{bad} 는 LOO 대상이 아니다. closed 4개를 학습에 쓴 조건이라 "
            f"train 이미지를 평가에 쓸 수 없다. 가능: {list(LOO_CONDS)}")

    meta = load_split()
    closed = sorted(meta["closed_classes"])
    rows, per_class, scores = [], {}, {}

    for cond in conds:
        print(f"[{cond}]")
        for metric in METRICS:
            r = evaluate_loo(cond, meta, metric, emb_root)
            per_class[f"{cond}|{metric}"] = r.pop("_per_class")
            scores[f"{cond}|{metric}"] = r.pop("_scores")
            rows.append(r)
            print(f"  {metric:<7} 미학습recall={r['unseen_recall']:.4f}  "
                  f"미학습F1={r['unseen_macro_f1']:.4f}  "
                  f"13개macroF1={r['macro_f1_13']:.4f}  |  "
                  f"거부AUROC={r['reject_auroc']:.4f}  FPR@95={r['reject_fpr95']:.4f}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(per_class, fh, ensure_ascii=False, indent=2)
    with open(_ROOT / "results_loo_scores.json", "w", encoding="utf-8") as fh:
        json.dump(scores, fh)

    # --- 클래스별 recall (미학습 4개)
    print(f"\n=== 미학습 4개 클래스별 recall (cosine)")
    print(f"  {'제조사':<10}{'전체장수':>8}{'기존test':>9}", end="")
    for c in conds:
        print(f"{c:>9}", end="")
    print()
    for cls in sorted(closed, key=lambda x: -meta["counts"][x]):
        n_old = len(meta["splits"][cls]["test"])
        print(f"  {cls:<10}{meta['counts'][cls]:>8}{n_old:>9}", end="")
        for c in conds:
            d = per_class.get(f"{c}|cosine", {}).get(cls)
            print(f"{d['recall']:>9.3f}" if d else f"{'-':>9}", end="")
        print()

    print(f"\n=== 거부(모르는 브랜드 판별) — AUROC 0.5=무작위, 1.0=완벽")
    print(f"  {'조건':<6}{'거리':<9}{'AUROC':>8}{'FPR@95TPR':>12}")
    for r in rows:
        print(f"  {r['cond']:<6}{r['metric']:<9}{r['reject_auroc']:>8.4f}{r['reject_fpr95']:>12.4f}")

    print(f"\n저장: {OUT_CSV.name}, {OUT_JSON.name}, results_loo_scores.json")


if __name__ == "__main__":
    main()
