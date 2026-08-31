#!/usr/bin/env python3
"""
Mean-prototype 빌더.

프로토타입 = 해당 제조사 **train split** 임베딩의 평균.
(val/test 로 프로토타입을 만들면 누수다. 항상 train 만 쓴다.)

GT-prototype 은 폐기했다 — 소스였던 AI Hub Data 가 13개 클래스 중 3개만 덮고
(osstem 115 / dentium 36 / iti 6), 촬영 도메인도 달라 비교가 성립하지 않았다.

거리별로 평균 내는 방식이 다르다:
  l2     : 그냥 산술평균. 유클리드 공간의 중심.
  cosine : 각 벡터를 단위길이로 정규화한 뒤 평균 → 다시 정규화 (구면 평균).
           정규화 없이 평균 내면 노름이 큰 샘플이 방향을 독점한다.

E3(9개로 학습 → 13개 분류)에서는 학습에 쓰지 않은 closed 4개의 프로토타입도
그 train 임베딩으로 만든다. 모델이 그 클래스를 학습한 적은 없지만,
"대표 벡터를 만들 만큼의 예시는 있다"는 것이 프로토타입 분류의 전제다.
"""

from pathlib import Path

import numpy as np

EMB_ROOT = Path(__file__).parent / "embeddings"
METRICS = ("l2", "cosine")


def load_embeddings(cond, split, root=EMB_ROOT):
    """→ (vecs (N,D) float32, labels (N,) str, paths (N,) str)"""
    p = Path(root) / cond / f"{split}.npz"
    if not p.is_file():
        raise FileNotFoundError(
            f"{p} 없음. 먼저 `python extract.py --cond {cond}` 를 실행할 것.")
    z = np.load(p, allow_pickle=False)
    return z["vecs"].astype(np.float32), z["labels"].astype(str), z["paths"].astype(str)


def l2_normalize(x, eps=1e-12):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def build_prototypes(vecs, labels, classes, metric="l2"):
    """→ protos (C, D). classes 순서를 그대로 따른다.

    해당 클래스의 샘플이 하나도 없으면 에러 — 조용히 0벡터를 두면
    거리 계산이 말없이 이상해진다.
    """
    if metric not in METRICS:
        raise ValueError(f"metric 은 {METRICS} 중 하나여야 함: {metric}")
    protos = np.zeros((len(classes), vecs.shape[1]), dtype=np.float32)
    for i, c in enumerate(classes):
        m = labels == c
        n = int(m.sum())
        if n == 0:
            raise ValueError(f"클래스 '{c}' 의 샘플이 0개 — 프로토타입을 만들 수 없다.")
        v = vecs[m]
        if metric == "cosine":
            protos[i] = l2_normalize(l2_normalize(v).mean(0))
        else:
            protos[i] = v.mean(0)
    return protos


def prototype_stats(vecs, labels, classes, metric="l2"):
    """진단용: 클래스별 표본수 + 프로토타입 주변 산포(intra-class 평균거리).

    산포가 유독 큰 클래스는 Mean-prototype 이 이상치에 오염됐을 가능성이 있다.
    """
    protos = build_prototypes(vecs, labels, classes, metric)
    out = {}
    for i, c in enumerate(classes):
        v = vecs[labels == c]
        if metric == "cosine":
            d = 1.0 - l2_normalize(v) @ protos[i]
        else:
            d = np.linalg.norm(v - protos[i], axis=1)
        out[c] = {"n": int(v.shape[0]),
                  "intra_mean": float(d.mean()),
                  "intra_std": float(d.std())}
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="합성 데이터로 로직만 검증 (임베딩 파일 불필요)")
    ap.add_argument("--cond", default=None, help="실제 임베딩으로 산포 출력")
    args = ap.parse_args()

    if args.selftest:
        rng = np.random.default_rng(0)
        classes = ["a", "b", "c"]
        centers = rng.normal(size=(3, 8)).astype(np.float32) * 3
        vecs, labels = [], []
        for i, c in enumerate(classes):
            v = centers[i] + rng.normal(size=(20, 8)).astype(np.float32) * 0.1
            vecs.append(v)
            labels += [c] * 20
        vecs = np.concatenate(vecs)
        labels = np.array(labels)

        p_l2 = build_prototypes(vecs, labels, classes, "l2")
        assert p_l2.shape == (3, 8)
        err = np.abs(p_l2 - centers).max()
        assert err < 0.2, f"l2 프로토타입이 중심에서 너무 벗어남: {err}"

        p_cos = build_prototypes(vecs, labels, classes, "cosine")
        norms = np.linalg.norm(p_cos, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), f"cosine 프로토타입 노름 != 1: {norms}"

        # 노름이 큰 이상치 1개를 넣어도 cosine 프로토타입 방향은 거의 안 흔들려야 한다
        v2 = np.concatenate([vecs, (centers[0] * 50)[None]])
        l2b = np.append(labels, "a")
        p2 = build_prototypes(v2, l2b, classes, "cosine")
        drift = 1 - float(p2[0] @ p_cos[0])
        assert drift < 0.05, f"cosine 프로토타입이 이상치에 흔들림: {drift}"

        try:
            build_prototypes(vecs, labels, ["a", "b", "c", "없는클래스"], "l2")
        except ValueError:
            pass
        else:
            raise AssertionError("빈 클래스인데 에러가 안 났다")

        print("prototype.py selftest 통과")

    if args.cond:
        vecs, labels, _ = load_embeddings(args.cond, "train")
        classes = sorted(set(labels.tolist()))
        for metric in METRICS:
            print(f"\n[{args.cond} / {metric}] 클래스별 intra-class 산포")
            st = prototype_stats(vecs, labels, classes, metric)
            for c in sorted(st, key=lambda x: -st[x]["intra_mean"]):
                s = st[c]
                print(f"  {c:<10} n={s['n']:>3}  "
                      f"mean={s['intra_mean']:.4f}  std={s['intra_std']:.4f}")
