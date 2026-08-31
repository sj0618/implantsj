#!/usr/bin/env python3
"""
현상 진단 3종. 전부 캐시된 임베딩만 쓴다 (재학습·GPU 불필요).

  python diagnostics.py --dup          # 1) train/test 근접 중복 검사
  python diagnostics.py --hub          # 2) hubness — M1 의 거부 AUROC < 0.5 설명
  python diagnostics.py --reject       # 3) 거부 점수 대안 비교
  python diagnostics.py --all

1) 근접 중복 (--dup)
   같은 원본 X-ray 에서 나온 crop 이 train 과 test 에 갈라져 있으면 성능이
   부풀려진다. 다른 버전 데이터(13label_2)에서는 실제로 있었다 —
   1443장이 고유 원본 1187개에서 나왔다. 지금 데이터는 파일명이 정규화돼
   있어 원본 추적이 불가능하므로, **그림 내용(임베딩)으로** 확인한다.
   학습 편향이 없는 M1(사전학습 그대로) 임베딩을 쓴다.

2) hubness (--hub)
   고차원 임베딩에서 일부 지점이 유독 많은 것의 최근접이 되는 현상.
   M1 의 거부 AUROC 가 0.394 로 무작위보다 나빴는데, 미학습 이미지가
   공간 중심부(허브)에 놓이면 이런 결과가 나온다. 프로토타입별 최근접
   선택 횟수를 세어 확인한다.

3) 거부 점수 대안 (--reject)
   지금은 "최근접 프로토타입까지의 거리" 하나뿐이다. OOD 탐지에서 흔히
   쓰는 다른 점수들과 비교한다.
     min_dist  : 최근접 거리 (현재)
     margin    : 1등과 2등 거리의 차 — 작으면 애매 = unknown
     dist_ent  : 거리를 softmax 로 바꾼 분포의 엔트로피
     head_msp  : 분류기 softmax 최대확률 (고전적 OOD 기준선)
     head_ent  : 분류기 softmax 엔트로피
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from dataset import load_split
from loo_eval import auroc, fpr_at_tpr, gather
from metrics import distance_matrix
from prototype import EMB_ROOT, l2_normalize, load_embeddings

_ROOT = Path(__file__).parent
CKPT_DIR = _ROOT / "ckpt"


# ---------------------------------------------------------------- 1. 근접 중복


def check_duplicates(cond="M1", thresholds=(0.99, 0.98, 0.95), topk=10):
    meta = load_split()
    tr_v, tr_l, tr_p = load_embeddings(cond, "train")
    te_v, te_l, te_p = load_embeddings(cond, "test")

    A, B = l2_normalize(te_v), l2_normalize(tr_v)
    sim = A @ B.T                       # (test, train)
    best = sim.max(1)
    arg = sim.argmax(1)

    print(f"=== train/test 근접 중복 검사 ({cond} 임베딩, test {len(te_l)}장)")
    q = np.quantile(best, [0.5, 0.9, 0.99, 1.0])
    print(f"  test 이미지별 'train 최근접 유사도' 분위수")
    print(f"    중앙값 {q[0]:.4f} / 90% {q[1]:.4f} / 99% {q[2]:.4f} / 최대 {q[3]:.4f}")
    for t in thresholds:
        n = int((best >= t).sum())
        print(f"    유사도 >= {t}: {n}장 ({n/len(best)*100:.1f}%)")

    order = np.argsort(-best)[:topk]
    print(f"\n  가장 비슷한 쌍 상위 {topk}")
    print(f"  {'유사도':>8}  {'같은클래스':>8}  test / train")
    for i in order:
        j = arg[i]
        same = "O" if te_l[i] == tr_l[j] else "X"
        print(f"  {best[i]:>8.4f}  {same:>8}  {te_p[i]}  |  {tr_p[j]}")

    print("\n  판정: 유사도 0.98 이상이 여러 장이면 같은 원본에서 나온 crop 일 수 있다.")
    print("        해당 쌍의 실제 이미지를 열어 확인할 것.")
    return {"max": float(best.max()), "n_over_98": int((best >= 0.98).sum())}


# ---------------------------------------------------------------- 2. hubness


def check_hubness(conds=("M1", "M3", "M5", "M7"), metric="cosine"):
    meta = load_split()
    open_c = sorted(meta["open_classes"])
    print(f"=== hubness — 9개 프로토타입이 최근접으로 뽑히는 횟수 ({metric})")
    print(f"  균등하면 각 프로토타입이 known 182/9={182/9:.1f}회, "
          f"unknown 86/9={86/9:.1f}회 뽑힌다.")

    out = {}
    for cond in conds:
        (otr_v, otr_l), (ote_v, ote_l), (cl_v, cl_l) = gather(cond, meta)
        prep = (lambda v: l2_normalize(v)) if metric == "cosine" else (lambda v: v)
        fin = prep if metric == "cosine" else (lambda p: p)
        protos = np.stack([fin(prep(otr_v[otr_l == c]).mean(0)) for c in open_c])

        nk = distance_matrix(ote_v, protos, metric).argmin(1)
        nu = distance_matrix(cl_v, protos, metric).argmin(1)
        ck, cu = Counter(nk.tolist()), Counter(nu.tolist())

        # 편중도: 최대 선택 횟수 / 기대값
        skew_k = max(ck.values()) / (len(nk) / len(open_c))
        skew_u = max(cu.values()) / (len(nu) / len(open_c))
        top_u = open_c[cu.most_common(1)[0][0]]
        share_u = cu.most_common(1)[0][1] / len(nu)

        print(f"\n  [{cond}] known 편중도 {skew_k:.2f}x / unknown 편중도 {skew_u:.2f}x")
        print(f"    unknown 을 가장 많이 빨아들인 프로토타입: {top_u} "
              f"({cu.most_common(1)[0][1]}/{len(nu)}장 = {share_u*100:.0f}%)")
        print(f"    {'브랜드':<10}{'known':>7}{'unknown':>9}")
        for i, c in enumerate(open_c):
            print(f"    {c:<10}{ck.get(i,0):>7}{cu.get(i,0):>9}")
        out[cond] = {"skew_known": skew_k, "skew_unknown": skew_u,
                     "top_unknown_class": top_u, "top_unknown_share": share_u}
    print("\n  판정: unknown 편중도가 known 보다 훨씬 크면 특정 프로토타입이 허브다.")
    return out


# ---------------------------------------------------------------- 3. 거부 점수


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _entropy(p, axis=-1):
    return -(p * np.log(p + 1e-12)).sum(axis=axis)


def _head_probs(cond, vecs):
    """ckpt 의 분류기를 임베딩에 적용해 softmax 확률을 얻는다. 없으면 None."""
    p = CKPT_DIR / f"{cond}.pt"
    if not p.is_file():
        return None
    import torch
    ck = torch.load(p, map_location="cpu")
    if "head" not in ck:
        return None
    W = ck["head"]["weight"].numpy()
    b = ck["head"]["bias"].numpy()
    return _softmax(vecs @ W.T + b, axis=1)


def reject_scores(cond, metric="cosine"):
    """→ {점수이름: (known 점수, unknown 점수)}. 값이 클수록 unknown."""
    meta = load_split()
    open_c = sorted(meta["open_classes"])
    (otr_v, otr_l), (ote_v, ote_l), (cl_v, cl_l) = gather(cond, meta)
    prep = (lambda v: l2_normalize(v)) if metric == "cosine" else (lambda v: v)
    fin = prep if metric == "cosine" else (lambda p: p)
    protos = np.stack([fin(prep(otr_v[otr_l == c]).mean(0)) for c in open_c])

    out = {}
    for name, X in (("known", ote_v), ("unknown", cl_v)):
        d = distance_matrix(X, protos, metric)
        s = np.sort(d, axis=1)
        out.setdefault("min_dist", {})[name] = s[:, 0]
        out.setdefault("margin", {})[name] = -(s[:, 1] - s[:, 0])   # 작을수록 애매
        out.setdefault("dist_ent", {})[name] = _entropy(_softmax(-d * 10, axis=1))
        pr = _head_probs(cond, X)
        if pr is not None:
            out.setdefault("head_msp", {})[name] = -pr.max(1)       # 낮을수록 unknown
            out.setdefault("head_ent", {})[name] = _entropy(pr)
    return out


def compare_reject(conds=("M1", "M3", "M5", "M7"), metric="cosine"):
    print(f"=== 거부 점수 대안 비교 ({metric}) — AUROC / FPR@95%TPR")
    names = ["min_dist", "margin", "dist_ent", "head_msp", "head_ent"]
    print(f"  {'조건':<6}" + "".join(f"{n:>20}" for n in names))
    out = {}
    for cond in conds:
        sc = reject_scores(cond, metric)
        cells, rec = [], {}
        for n in names:
            if n not in sc:
                cells.append(f"{'—':>20}")
                continue
            a = auroc(sc[n]["unknown"], sc[n]["known"])
            f = fpr_at_tpr(sc[n]["unknown"], sc[n]["known"])
            rec[n] = {"auroc": a, "fpr95": f}
            cells.append(f"{a:>13.3f}/{f:.2f}")
        print(f"  {cond:<6}" + "".join(cells))
        out[cond] = rec
    print("\n  min_dist=최근접거리  margin=1·2등 거리차  dist_ent=거리 softmax 엔트로피")
    print("  head_msp=분류기 최대확률  head_ent=분류기 엔트로피 (M1 은 분류기 없음)")
    return out


# ---------------------------------------------------------------- CLI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dup", action="store_true")
    ap.add_argument("--hub", action="store_true")
    ap.add_argument("--reject", action="store_true")
    ap.add_argument("--fix", action="store_true", help="hubness 보정 후 거부 성능")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--metric", default="cosine", choices=("l2", "cosine"))
    args = ap.parse_args()

    res = {}
    if args.dup or args.all:
        res["duplicates"] = check_duplicates()
        print()
    if args.hub or args.all:
        res["hubness"] = check_hubness(metric=args.metric)
        print()
    if args.reject or args.all:
        res["reject_variants"] = compare_reject(metric=args.metric)
        print()
    if args.fix or args.all:
        res["hub_fix"] = compare_hub_fix(metric=args.metric)
        print()
    if not res:
        ap.print_help()
        return
    with open(_ROOT / "results_diagnostics.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2, default=float)
    print("저장: results_diagnostics.json")




# ---------------------------------------------------------------- 4. hubness 보정


def hub_corrected(cond, metric="cosine", k=10):
    """hubness 를 보정한 거부 점수 3종. → {이름: {known, unknown}} (클수록 unknown)

    raw       : 최근접 프로토타입까지 거리 (보정 없음)
    proto_std : 프로토타입별로 거리를 z-score 화. 어떤 프로토타입이 원래 모든 것과
                가까우면(허브) 그 거리의 평균·표준편차로 나눠 공평하게 만든다.
                기준 분포는 open train 임베딩에서 구한다 (test 미사용).
    sample_std: 한 샘플의 9개 거리 안에서 표준화. "1등이 나머지보다 얼마나
                두드러지는가"를 재므로 샘플이 놓인 위치의 밀도에 덜 휘둘린다.
    csls      : Conneau et al. 2017. 유사도에서 양쪽의 국소 밀도를 빼준다.
                csls(x,p) = 2·sim(x,p) − r(x) − r(p)
                r(p)=프로토타입 p 와 기준집합 상위 k개의 평균 유사도,
                r(x)=샘플 x 와 9개 프로토타입의 평균 유사도.
                허브 프로토타입은 r(p)가 커서 자동으로 벌점을 받는다. (cosine 전용)
    """
    meta = load_split()
    open_c = sorted(meta["open_classes"])
    (otr_v, otr_l), (ote_v, ote_l), (cl_v, cl_l) = gather(cond, meta)
    prep = (lambda v: l2_normalize(v)) if metric == "cosine" else (lambda v: v)
    fin = prep if metric == "cosine" else (lambda p: p)
    protos = np.stack([fin(prep(otr_v[otr_l == c]).mean(0)) for c in open_c])

    # 기준 분포: open train (모델이 이미 본 데이터 — test 누수 없음)
    d_ref = distance_matrix(otr_v, protos, metric)          # (N_ref, 9)
    mu, sd = d_ref.mean(0), d_ref.std(0) + 1e-12
    if metric == "cosine":
        s_ref = 1.0 - d_ref                                  # 유사도
        kk = min(k, s_ref.shape[0])
        r_p = np.sort(s_ref, axis=0)[-kk:].mean(0)           # 프로토타입별 국소 밀도

    out = {}
    for name, X in (("known", ote_v), ("unknown", cl_v)):
        d = distance_matrix(X, protos, metric)
        out.setdefault("raw", {})[name] = d.min(1)
        out.setdefault("proto_std", {})[name] = ((d - mu) / sd).min(1)
        z = (d - d.mean(1, keepdims=True)) / (d.std(1, keepdims=True) + 1e-12)
        out.setdefault("sample_std", {})[name] = z.min(1)
        if metric == "cosine":
            s = 1.0 - d
            r_x = s.mean(1)
            out.setdefault("csls", {})[name] = -(2 * s - r_x[:, None] - r_p[None, :]).max(1)
    return out


def compare_hub_fix(conds=("M1", "M3", "M5", "M7"), metric="cosine"):
    names = ["raw", "proto_std", "sample_std"] + (["csls"] if metric == "cosine" else [])
    print(f"=== hubness 보정 후 거부 성능 ({metric}) — AUROC / FPR@95%TPR")
    print(f"  {'조건':<6}" + "".join(f"{n:>20}" for n in names))
    out = {}
    for cond in conds:
        sc = hub_corrected(cond, metric)
        cells, rec = [], {}
        for n in names:
            a = auroc(sc[n]["unknown"], sc[n]["known"])
            f = fpr_at_tpr(sc[n]["unknown"], sc[n]["known"])
            rec[n] = {"auroc": a, "fpr95": f}
            cells.append(f"{a:>13.3f}/{f:.2f}")
        print(f"  {cond:<6}" + "".join(cells))
        out[cond] = rec
    best = {c: max(out[c], key=lambda n: out[c][n]["auroc"]) for c in out}
    print(f"\n  조건별 최선: " + ", ".join(f"{c}={b}({out[c][b]['auroc']:.3f})"
                                          for c, b in best.items()))
    return out


if __name__ == "__main__":
    main()
