#!/usr/bin/env python3
"""
프로토타입 거리 기반 분류 + 지표.

분류: 각 test 임베딩을 가장 가까운 제조사 프로토타입에 배정.
지표: Top-1 / Top-3 / macro-F1 (+ 클래스별 성능)

macro-F1 이 대표값이다. 클래스 불균형이 53:1(adin 266 : nobel 5)이라
Top-1 은 큰 클래스만 맞혀도 높게 나온다.

macro-F1 은 sklearn 없이 직접 계산한다. 의존성을 줄이려는 것도 있지만,
주된 이유는 zero-division 처리를 명시적으로 통제하기 위해서다 —
nobel 은 test 가 1장이라 예측이 한 번도 안 나오는 클래스가 생길 수 있고,
그때 precision 을 0 으로 둘지 제외할지가 macro 평균을 크게 바꾼다.
여기서는 **0으로 두고 클래스에 포함**한다 (sklearn zero_division=0 과 동일).
"""

import numpy as np

from prototype import build_prototypes, l2_normalize

TOPK = (1, 3)


def distance_matrix(vecs, protos, metric):
    """→ (N, C). 값이 **작을수록 가깝다** (cosine 도 거리로 변환해 통일)."""
    if metric == "cosine":
        return 1.0 - (l2_normalize(vecs) @ l2_normalize(protos).T)
    if metric == "l2":
        # ||a-b||^2 전개보다 수치적으로 안전한 직접 계산
        return np.linalg.norm(vecs[:, None, :] - protos[None, :, :], axis=2)
    raise ValueError(f"알 수 없는 metric: {metric}")


def topk_accuracy(dist, y_true, k):
    """dist (N,C), y_true (N,) 정수 인덱스."""
    k = min(k, dist.shape[1])
    topk = np.argsort(dist, axis=1)[:, :k]
    return float((topk == y_true[:, None]).any(axis=1).mean())


def per_class_prf(y_true, y_pred, n_classes):
    """→ {class_idx: {precision, recall, f1, support}}. zero-division 은 0."""
    out = {}
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[c] = {"precision": prec, "recall": rec, "f1": f1,
                  "support": tp + fn, "n_pred": tp + fp}
    return out


def macro_f1(y_true, y_pred, n_classes):
    prf = per_class_prf(y_true, y_pred, n_classes)
    return float(np.mean([prf[c]["f1"] for c in range(n_classes)]))


def evaluate(test_vecs, test_labels, train_vecs, train_labels, classes, metric):
    """한 셀의 전체 평가.

    프로토타입은 train 에서, 평가는 test 에서. 둘 다 `classes` 로 제한한다.
    test_labels 중 `classes` 에 없는 것은 미리 걸러서 넣어야 한다.
    """
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_true = np.array([class_to_idx[l] for l in test_labels], dtype=np.int64)

    protos = build_prototypes(train_vecs, train_labels, classes, metric)
    dist = distance_matrix(test_vecs, protos, metric)
    y_pred = dist.argmin(axis=1)

    res = {
        "n_test": int(len(y_true)),
        "n_classes": len(classes),
        "macro_f1": macro_f1(y_true, y_pred, len(classes)),
    }
    for k in TOPK:
        res[f"top{k}"] = topk_accuracy(dist, y_true, k)

    prf = per_class_prf(y_true, y_pred, len(classes))
    res["per_class"] = {
        classes[c]: {kk: (round(vv, 6) if isinstance(vv, float) else vv)
                     for kk, vv in prf[c].items()}
        for c in range(len(classes))
    }
    return res


def filter_to_classes(vecs, labels, classes):
    """classes 에 속한 샘플만 남긴다."""
    keep = np.isin(labels, list(classes))
    return vecs[keep], labels[keep]


# ---------------------------------------------------------------- selftest

def _selftest():
    rng = np.random.default_rng(0)
    classes = ["a", "b", "c"]
    centers = rng.normal(size=(3, 16)).astype(np.float32) * 5

    def sample(n, spread):
        v, l = [], []
        for i, c in enumerate(classes):
            v.append(centers[i] + rng.normal(size=(n, 16)).astype(np.float32) * spread)
            l += [c] * n
        return np.concatenate(v), np.array(l)

    tr_v, tr_l = sample(30, 0.1)
    te_v, te_l = sample(10, 0.1)

    for metric in ("l2", "cosine"):
        r = evaluate(te_v, te_l, tr_v, tr_l, classes, metric)
        assert r["top1"] == 1.0, f"{metric}: 잘 분리된 데이터인데 top1={r['top1']}"
        assert r["macro_f1"] == 1.0, f"{metric}: macro_f1={r['macro_f1']}"
        assert r["n_test"] == 30 and r["n_classes"] == 3

    # 완전 무작위면 top1 이 1/3 근처여야 한다
    rv = rng.normal(size=(300, 16)).astype(np.float32)
    rl = np.array([classes[i % 3] for i in range(300)])
    r = evaluate(rv, rl, rv, rl, classes, "cosine")
    assert 0.15 < r["top1"] < 0.55, f"무작위인데 top1={r['top1']}"

    # top3 >= top1 이어야 하고, 클래스가 3개면 top3 == 1.0
    assert r["top3"] >= r["top1"] and r["top3"] == 1.0

    # zero-division: 한 클래스가 한 번도 예측되지 않는 경우
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 1, 1])  # 클래스 2는 예측 0회
    prf = per_class_prf(y_true, y_pred, 3)
    assert prf[2]["precision"] == 0.0 and prf[2]["recall"] == 0.0
    assert prf[2]["f1"] == 0.0 and prf[2]["support"] == 2
    mf = macro_f1(y_true, y_pred, 3)
    assert abs(mf - (1.0 + 2 / 3 + 0.0) / 3) < 1e-9, mf

    # 불균형에서 macro_f1 이 top1 보다 낮게 나오는지 (지표가 의도대로 작동하는지)
    y_true = np.array([0] * 95 + [1] * 5)
    y_pred = np.array([0] * 100)  # 다수 클래스만 맞힘
    acc = float((y_true == y_pred).mean())
    mf = macro_f1(y_true, y_pred, 2)
    assert acc == 0.95 and mf < 0.5, f"acc={acc} macro_f1={mf}"

    # filter_to_classes
    v = np.arange(12, dtype=np.float32).reshape(6, 2)
    l = np.array(["a", "b", "c", "a", "b", "c"])
    fv, fl = filter_to_classes(v, l, ["a", "b"])
    assert fv.shape == (4, 2) and set(fl.tolist()) == {"a", "b"}

    print("metrics.py selftest 통과")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.print_help()
