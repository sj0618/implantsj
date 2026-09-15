"""처음 보는(unseen) 임플란트 시스템 few-shot / 0-shot 평가 프로토콜 코어.

명세: docs/paper/FEWSHOT_ATTRIBUTE_SPEC.md §2(분할·에피소드·0-shot·지표), §3(분류기·특징 확률 출처),
§5(진단), §6.3(계약). numpy/pandas/sklearn/scipy 만 쓴다(torch 불필요).

  - make_splits / item_roles / check_split : label-unseen ∧ patient-unseen 분할과 불변식
  - sample_episode : unseen all-way K-shot 에피소드. 임베딩과 무관하게 (manifest, split, K, e, seed)
                     로만 정해지므로 모든 표현·분류기가 같은 에피소드 위에서 짝지어 비교된다.
  - classify       : cos / euc / ss / mt / mw (§3 표)
  - evaluate       : emb.npz + manifest + split -> 결과 dict (K-shot, GFSL, 특징표 채우기, 0-shot, 진단)
  - aggregate      : 평균 ± 95% CI (t 분포)

  python -m src.fewshot split    --manifest m.csv --out s.json [--n-splits 5 --min-groups-unseen 12 --n-unseen N --seed 0]
  python -m src.fewshot evaluate --emb e.npz --manifest m.csv --splits s.json --split-id 0 \
                                 --ks 1,3,5,10 --episodes 200 [--feature-table configs/class_features.csv] --out r.json
  python -m src.fewshot selftest
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
import zlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import solve_triangular

CLASSIFIERS = ('cos', 'euc', 'ss', 'mt', 'mtr', 'mw')
ROLES = ('train', 'val', 'test', 'unseen', 'excluded')


def _read_manifest(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={'label': str, 'group': str, 'class_key': str, 'domain': str})
    # 상대경로는 manifest 파일이 있는 폴더 기준으로 푼다(zip 으로 다른 PC·OS 에 옮긴 데이터를 그대로 쓰기 위해)
    base = Path(path).resolve().parent
    df['path'] = [p if Path(p).is_absolute() else str(base / p) for p in df['path'].astype(str)]
    return _prep(df)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ('path', 'label', 'group') if c not in df.columns]
    assert not missing, f'manifest 필수 열 없음: {missing}'
    assert df[['path', 'label', 'group']].notna().all().all(), 'path/label/group 에 빈 칸이 있다'
    df = df.reset_index(drop=True).copy()
    df['label'] = df['label'].astype(str)
    df['group'] = df['group'].astype(str)
    return df


# ---------------------------------------------------------------------------
# §2.1 분할
# ---------------------------------------------------------------------------
def make_splits(manifest_df, n_splits=5, n_unseen=None, min_groups_unseen=12, seed=0,
                val_frac=0.1, test_frac=0.1) -> list[dict]:
    """분할 r 은 rng(seed*1000 + r) 로만 정해진다. 역할은 (label, group) 으로 정의한다."""
    df = _prep(manifest_df)
    n_groups = df.groupby('label')['group'].nunique()
    classes = sorted(n_groups.index)
    pool = sorted(n_groups.index[n_groups >= min_groups_unseen])
    if n_unseen is None:
        n_unseen = int(round(len(classes) * 15 / 42))
    if n_unseen > len(pool):
        warnings.warn(f'n_unseen={n_unseen} > 자격 클래스 {len(pool)}개 (n_groups>={min_groups_unseen}); '
                      f'{len(pool)} 로 줄인다')
        n_unseen = len(pool)

    splits = []
    for r in range(n_splits):
        rng = np.random.default_rng(seed * 1000 + r)
        unseen = sorted(rng.choice(pool, n_unseen, replace=False).tolist())
        unseen_groups = sorted(df.loc[df['label'].isin(unseen), 'group'].unique())
        seen_df = df[~df['group'].isin(set(unseen_groups))]
        perm = rng.permutation(sorted(seen_df['group'].unique())).tolist()
        n_val, n_test = int(round(len(perm) * val_frac)), int(round(len(perm) * test_frac))
        roles = {g: ('val' if i < n_val else 'test' if i < n_val + n_test else 'train')
                 for i, g in enumerate(perm)}
        # 모든 seen 클래스가 train group 을 1개 이상 갖게 한다(perm 순서상 첫 group 을 train 으로)
        order = {g: i for i, g in enumerate(perm)}
        for c, gs in seen_df.groupby('label')['group'].unique().items():
            if not any(roles[g] == 'train' for g in gs):
                roles[min(gs, key=order.get)] = 'train'
        seen = [c for c in classes if c not in set(unseen)]
        no_train = sorted(set(seen) - set(seen_df['label']))
        if no_train:
            warnings.warn(f'split {r}: seen 인데 unseen group 에만 있어 train 항목이 없는 클래스 {no_train}')
        splits.append({'split_id': r, 'seed': seed, 'min_groups_unseen': min_groups_unseen,
                       'unseen_classes': unseen, 'seen_classes': seen,
                       'unseen_groups': unseen_groups, 'group_roles': roles})
    return splits


def item_roles(manifest_df, split) -> pd.Series:
    """label∈unseen→unseen; group∈unseen_groups→excluded; 그 외 group_roles[group] (없으면 NaN)."""
    lab = manifest_df['label'].astype(str)
    grp = manifest_df['group'].astype(str)
    roles = grp.map(split['group_roles']).astype(object)
    roles[grp.isin(set(split['unseen_groups']))] = 'excluded'
    roles[lab.isin(set(split['unseen_classes']))] = 'unseen'
    return roles


def check_split(manifest_df, split) -> None:
    """§2.1 불변식. 위반하면 AssertionError."""
    df = _prep(manifest_df)
    roles = item_roles(df, split)
    bad = df.loc[~roles.isin(ROLES), 'path']
    assert bad.empty, f'역할이 없는 항목 {len(bad)}개(분할과 manifest 의 group 불일치?): {bad.head(3).tolist()}'
    unseen, seen = set(split['unseen_classes']), set(split['seen_classes'])
    assert not unseen & seen, f'unseen/seen 클래스 중복: {sorted(unseen & seen)}'
    assert not set(split['group_roles']) & set(split['unseen_groups']), 'group_roles 와 unseen_groups 가 겹친다'
    assert (roles[df['label'].isin(unseen)] == 'unseen').all()
    for g, rs in roles.groupby(df['group']):
        rs = set(rs)
        ok = rs <= {'unseen', 'excluded'} or (len(rs) == 1 and rs <= {'train', 'val', 'test'})
        assert ok, f'group {g!r} 가 두 역할에 걸친다: {sorted(rs)}'


# ---------------------------------------------------------------------------
# §2.2 에피소드
# ---------------------------------------------------------------------------
def sample_episode(unseen_df, K, rng):
    """클래스마다 서로 다른 group K개 × group 당 1장 = 서포트, 그 클래스의 나머지 group 전부 = query.

    group 분리는 §2.2 대로 클래스 단위다(한 클래스의 서포트 group 은 그 클래스 query 에 없다).
    group 이 K+1 개 미만인 클래스는 이 K 에서 빠진다. 반환 인덱스는 unseen_df.index 값.
    """
    support, query, classes = [], [], []
    for c, g in unseen_df.groupby('label', sort=True):
        grp = g['group'].to_numpy(dtype=str)
        groups = np.unique(grp)
        if len(groups) < K + 1:
            continue
        chosen = rng.choice(groups, K, replace=False)
        idx = g.index.to_numpy()
        support += [rng.choice(idx[grp == s]) for s in chosen]
        query += idx[~np.isin(grp, chosen)].tolist()
        classes.append(c)
    return np.array(support, dtype=int), np.array(query, dtype=int), classes


# ---------------------------------------------------------------------------
# §3 분류기
# ---------------------------------------------------------------------------
def _l2(x):
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def _class_means(X, y):
    classes, inv = np.unique(y, return_inverse=True)
    M = np.zeros((len(classes), X.shape[1]))
    np.add.at(M, inv, X)
    return classes, M / np.bincount(inv)[:, None], inv


def _sqdist(Q, M):
    return (Q * Q).sum(1)[:, None] - 2 * Q @ M.T + (M * M).sum(1)[None]


def _maha(Q, M, P):
    A = Q @ P
    return (A * Q).sum(1)[:, None] - 2 * A @ M.T + ((M @ P) * M).sum(1)[None]


def classify(name, S, yS, Q, ctx=None):
    """서포트 (S, yS) 로 query Q 의 라벨을 예측한다. ctx: base_mean(ss), seen_cov / seen_prec(mw)."""
    ctx = ctx or {}
    S, Q = np.asarray(S, np.float64), np.asarray(Q, np.float64)
    if name == 'ss':
        S, Q = _l2(S - ctx['base_mean']), _l2(Q - ctx['base_mean'])
    elif name not in ('euc', 'mtr'):
        S, Q = _l2(S), _l2(Q)
    classes, M, inv = _class_means(S, np.asarray(yS))
    if name in ('euc', 'ss'):
        return classes[_sqdist(Q, M).argmin(1)]
    if name == 'cos':
        return classes[(Q @ _l2(M).T).argmax(1)]
    if name == 'mw':
        P = ctx.get('seen_prec')
        return classes[_maha(Q, M, np.linalg.inv(ctx['seen_cov']) if P is None else P).argmin(1)]
    # Simple CNAPS 헤드: Q_c = λΣ_c + (1-λ)Σ_task + βI, λ=K/(K+1), β=1.
    # mt 는 L2 공간이라 공분산 고유값(최대 0.07~0.14)이 β=1 보다 한 자릿수 작아 βI 가 지배해 euc 와 거의 같아진다.
    # mtr 은 원 논문처럼 정규화 전 특징 공간에서 같은 식을 쓴다(기준선 표는 mtr).
    if name in ('mt', 'mtr'):
        D = S.shape[1]
        task = np.cov(S, rowvar=False) if len(S) > 1 else np.zeros((D, D))
        d = np.empty((len(Q), len(classes)))
        for j in range(len(classes)):
            Sc = S[inv == j]
            lam = len(Sc) / (len(Sc) + 1)
            cov = lam * (np.cov(Sc, rowvar=False) if len(Sc) > 1 else 0.0) + (1 - lam) * task + np.eye(D)
            z = solve_triangular(np.linalg.cholesky(cov), (Q - M[j]).T, lower=True)
            d[:, j] = (z * z).sum(0)
        return classes[d.argmin(1)]
    raise ValueError(f'unknown classifier {name!r}; {CLASSIFIERS}')


# ---------------------------------------------------------------------------
# §2.4 통계
# ---------------------------------------------------------------------------
def aggregate(values: Sequence[float]) -> dict:
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    if n == 0:
        return {'mean': None, 'std': None, 'ci95_lo': None, 'ci95_hi': None, 'n': 0}
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if n > 1 else 0.0
    h = float(stats.t.ppf(0.975, n - 1) * std / math.sqrt(n)) if n > 1 else 0.0
    return {'mean': mean, 'std': std, 'ci95_lo': mean - h, 'ci95_hi': mean + h, 'n': n}


# ---------------------------------------------------------------------------
# 특징표 (§2.3, §3 특징 확률의 출처)
# ---------------------------------------------------------------------------
def _pick(P):
    """확률 argmax. 확률이 전부 0 인 행(학습에 값이 하나도 없던 feature)은 -1(예측 없음)."""
    out = P.argmax(1)
    out[P.max(1) <= 0] = -1
    return out


def _feature_rows(df, classes_all, feature_table):
    """label 코드 -> 특징표 정수 행(unknown=-1). 표의 class_key 열과 정확히 일치시킨다."""
    from .class_features import DEFAULT_TRAIN_ATTRIBUTES, build_targets, load_table

    assert df['class_key'].notna().all(), 'class_key 빈 칸이 있다'
    per_label = df.groupby('label')['class_key'].nunique()
    assert (per_label == 1).all(), f'label 하나에 class_key 여러 개: {per_label[per_label != 1].index[:5].tolist()}'
    table = load_table(feature_table)
    row_of = {k: i for i, k in enumerate(table['class_key'])}
    keys = sorted(df['class_key'].unique())
    missing = [k for k in keys if k not in row_of]
    if missing:
        raise ValueError(f'특징표 {feature_table} 에 없는 class_key: {missing}')
    ftab, target = build_targets(keys, {k: row_of[k] for k in keys}, table, DEFAULT_TRAIN_ATTRIBUTES)
    key_of = dict(zip(df['label'], df['class_key']))
    return ftab, target[[keys.index(key_of[c]) for c in classes_all]]


def _fill_acc(pred, truth, attrs):
    per = {}
    for ai, a in enumerate(attrs):
        m = truth[:, ai] >= 0
        per[a] = {'acc': float((pred[m, ai] == truth[m, ai]).mean()) if m.any() else None, 'n': int(m.sum())}
    accs = [v['acc'] for v in per.values() if v['acc'] is not None]
    return {'per_feature': per, 'mean': float(np.mean(accs)) if accs else None}


def _zero_shot_ceiling(R):
    """클래스 i 는 다른 모든 unseen 클래스 j 와 '둘 다 알려진' feature 중 하나 이상에서 값이 다르면 구분 가능.

    unknown 은 와일드카드로 보지 않는다(비교에서 뺄 뿐이다). 알려진 칸이 전부 같으면 원리상 구분 불가.
    """
    known = R >= 0
    ok = [bool(((R != R[i]) & known & known[i]).any(1)[np.arange(len(R)) != i].all()) for i in range(len(R))]
    return {'value': float(np.mean(ok)), 'n_distinguishable': int(sum(ok)), 'n_classes': len(R),
            'rule': 'distinct on >=1 attribute known for both classes, vs every other unseen class'}


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------
def evaluate(npz_path, manifest_csv, split, Ks, episodes=200, feature_table=None, seed=0,
             classifiers=CLASSIFIERS) -> dict:
    from sklearn.covariance import LedoitWolf
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    df = _read_manifest(manifest_csv)
    check_split(df, split)
    role = item_roles(df, split).to_numpy()
    tr, te, un = role == 'train', role == 'test', role == 'unseen'
    npz = np.load(npz_path, allow_pickle=False)
    pos = {p: i for i, p in enumerate(npz['path'].tolist())}
    idx = np.array([pos.get(p, -1) for p in df['path']])
    miss = df.loc[(tr | te | un) & (idx < 0), 'path']
    assert miss.empty, f'임베딩이 없는 manifest 항목 {len(miss)}개: {miss.head(3).tolist()}'
    emb = npz['emb'].astype(np.float64)
    X = np.zeros((len(df), emb.shape[1]))
    X[idx >= 0] = emb[idx[idx >= 0]]
    Z = _l2(X)

    lab = df['label'].to_numpy(dtype=str)
    classes_all = np.unique(lab)
    y = np.searchsorted(classes_all, lab)
    dom = df['domain'].to_numpy(dtype=str) if 'domain' in df.columns else None
    sid = split['split_id']
    sid = sid if isinstance(sid, int) else zlib.crc32(str(sid).encode())  # 예: split 'fixed'

    # seen train 통계: ss 의 기준 평균, mw 의 Σ_W(클래스별 중심화 L2 잔차 + Ledoit-Wolf), cos centroid
    seen_ids, seen_means, inv = _class_means(Z[tr], y[tr])
    resid = Z[tr] - seen_means[inv]
    lw = LedoitWolf(assume_centered=True).fit(resid)
    seen_cov = lw.covariance_
    ctx = {'base_mean': X[tr].mean(0), 'seen_cov': seen_cov, 'seen_prec': np.linalg.inv(seen_cov)}
    seen_cent = _l2(seen_means)

    # ---- 특징표: 확률 출처(head, probe)와 기준선(majority, nearest_seen)
    attr_out, probs, crow, attrs = None, {}, None, []
    if feature_table is not None and 'class_key' in df.columns:
        ftab, crow = _feature_rows(df, classes_all, feature_table)
        attrs, vocab = ftab.attributes, ftab.vocab
        heads = [a for a in attrs if f'attrprob__{a}' in npz.files]
        assert not heads or len(heads) == len(attrs), f'attrprob 가 일부 feature 에만 있다: {heads}'
        if heads:
            probs['head'] = {}
            for a in attrs:
                assert npz[f'attrvocab__{a}'].tolist() == vocab[a], f'{a}: npz attrvocab != 특징표 vocab'
                P = np.zeros((len(df), len(vocab[a])))
                P[idx >= 0] = npz[f'attrprob__{a}'][idx[idx >= 0]]
                probs['head'][a] = P
        probs['probe'], unreachable = {}, {}
        truth_un = crow[y[un]]
        for ai, a in enumerate(attrs):
            t = crow[y[tr], ai]
            keep = t >= 0
            vals = np.unique(t[keep])
            P = np.zeros((len(df), len(vocab[a])))
            if len(vals) == 1:
                P[:, vals[0]] = 1.0
            elif len(vals) > 1:
                clf = LogisticRegression(max_iter=2000).fit(Z[tr][keep], t[keep])
                P[np.ix_(un, clf.classes_)] = clf.predict_proba(Z[un])
            probs['probe'][a] = P
            lost = sorted(set(range(len(vocab[a]))) - set(vals.tolist()))
            unreachable[a] = {'values': [vocab[a][v] for v in lost],
                              'n_unseen_items': int(np.isin(truth_un[:, ai], lost).sum())}
        seen_codes = np.searchsorted(classes_all, [c for c in split['seen_classes'] if c in set(classes_all)])
        maj = []
        for ai in range(len(attrs)):
            v = crow[seen_codes, ai]
            v = v[v >= 0]
            maj.append(int(np.bincount(v).argmax()) if len(v) else -1)
        preds = {s: np.stack([_pick(P[un]) for P in pr.values()], 1) for s, pr in probs.items()}
        preds['majority'] = np.tile(maj, (int(un.sum()), 1))
        preds['nearest_seen'] = crow[seen_ids[(Z[un] @ seen_cent.T).argmax(1)]]
        attr_out = {'attributes': attrs, 'dropped_constant': ftab.dropped_constant, 'vocab': vocab,
                    'image_fill': {s: _fill_acc(p, truth_un, attrs) for s, p in preds.items()},
                    'probe_unreachable': unreachable, 'zero_shot': {}}
        # 0-shot: score_c(q) = mean_{a: row_c[a] 알려짐} log p_a(q)[row_c[a]]
        uc = np.unique(y[un])
        R = crow[uc]
        n_known = (R >= 0).sum(1)
        for s, pr in probs.items():
            score = np.zeros((int(un.sum()), len(uc)))
            for ai, a in enumerate(attrs):
                k = R[:, ai] >= 0
                score[:, k] += np.log(np.clip(pr[a][un], 1e-12, None))[:, R[k, ai]]
            score = np.where(n_known > 0, score / np.maximum(n_known, 1), -np.inf)
            yhat = uc[score.argmax(1)]
            attr_out['zero_shot'][s] = {
                'acc': float((yhat == y[un]).mean()),
                'f1': float(f1_score(y[un], yhat, labels=uc, average='macro', zero_division=0)),
                'n_queries': int(un.sum()), 'n_classes': len(uc)}
        attr_out['zero_shot_ceiling'] = _zero_shot_ceiling(R)

    # ---- K-shot 에피소드 (모든 분류기·표현이 같은 에피소드)
    udf = df[un]
    kshot = {}
    for K in Ks:
        eps = [sample_episode(udf, K, np.random.default_rng([seed, sid, K, e])) for e in range(episodes)]
        n_cls = len(eps[0][2]) if eps else 0
        out = {'n_classes': n_cls, 'n_episodes': 0}
        kshot[str(K)] = out
        if n_cls < 2:
            continue
        acc = {c: [] for c in classifiers}
        f1 = {c: [] for c in classifiers}
        dom_hits = {c: {} for c in classifiers}
        gfsl = {'acc_seen': [], 'acc_unseen': []}
        fill = {s: np.zeros((2, len(attrs))) for s in probs}
        for s, q, cls in eps:
            yq, labels = y[q], np.searchsorted(classes_all, cls)
            for c in classifiers:
                yhat = classify(c, X[s], y[s], X[q], ctx)
                acc[c].append(float((yhat == yq).mean()))
                f1[c].append(float(f1_score(yq, yhat, labels=labels, average='macro', zero_division=0)))
                if dom is not None:  # domain 별 query 정확도(에피소드 전체 합산)
                    for d in np.unique(dom[q]):
                        m = dom[q] == d
                        h = dom_hits[c].setdefault(d, [0, 0])
                        h[0] += int((yhat[m] == yq[m]).sum())
                        h[1] += int(m.sum())
            if te.any():
                ids_u, m_u, _ = _class_means(Z[s], y[s])
                ids, cent = np.concatenate([seen_ids, ids_u]), np.vstack([seen_cent, _l2(m_u)])
                a_s = float((ids[(Z[te] @ cent.T).argmax(1)] == y[te]).mean())
                a_u = float((ids[(Z[q] @ cent.T).argmax(1)] == yq).mean())
                gfsl['acc_seen'].append(a_s)
                gfsl['acc_unseen'].append(a_u)
            for src, pr in probs.items():  # 특징표 채우기(클래스, K): 서포트 확률 평균 -> argmax
                for c in np.unique(y[s]):
                    sc = s[y[s] == c]
                    for ai, a in enumerate(attrs):
                        if crow[c, ai] >= 0:
                            fill[src][0, ai] += _pick(pr[a][sc].mean(0, keepdims=True))[0] == crow[c, ai]
                            fill[src][1, ai] += 1
        out['n_episodes'] = len(eps)
        out['classifiers'] = {c: {'acc': aggregate(acc[c]), 'f1': aggregate(f1[c])} for c in classifiers}
        if dom is not None:
            for c in classifiers:
                out['classifiers'][c]['acc_by_domain'] = {d: {'acc': h / n, 'n': n}  # n: 에피소드 합산 query 수
                                                          for d, (h, n) in sorted(dom_hits[c].items())}
        if te.any():
            # H = 분할의 에피소드 평균 acc_seen·acc_unseen 의 조화평균(GZSL/GFSL 관례). 에피소드별 H 평균이 아니다.
            out['gfsl'] = {k: aggregate(v) for k, v in gfsl.items()}
            s_, u_ = out['gfsl']['acc_seen']['mean'], out['gfsl']['acc_unseen']['mean']
            out['gfsl']['H'] = {'mean': 2 * s_ * u_ / (s_ + u_) if s_ + u_ > 0 else 0.0,
                                'rule': 'harmonic mean of episode-mean acc_seen and acc_unseen'}
        if probs:
            out['attr_fill_class'] = {}
            for src, (hits, n) in fill.items():
                per = {a: {'acc': float(hits[i] / n[i]) if n[i] else None, 'n': int(n[i])}
                       for i, a in enumerate(attrs)}
                accs = [v['acc'] for v in per.values() if v['acc'] is not None]
                out['attr_fill_class'][src] = {'per_feature': per, 'mean': float(np.mean(accs)) if accs else None}

    # ---- §5 진단
    diag = {'delta_mw_cos': {k: 100 * (v['classifiers']['mw']['acc']['mean'] - v['classifiers']['cos']['acc']['mean'])
                             for k, v in kshot.items()
                             if v['n_episodes'] and {'mw', 'cos'} <= set(classifiers)}}
    uid, um, uinv = _class_means(Z[un], y[un])
    tr_w = (np.bincount(uinv, ((Z[un] - um[uinv]) ** 2).sum(1)) / np.bincount(uinv)).mean()
    tr_b = ((um - um.mean(0)) ** 2).sum(1).mean()
    diag['trace_ratio_unseen'] = float(tr_w / tr_b)
    # 유효 순위는 축소 없는 pooled Σ_W 로 잰다. LW 의 identity 성분은 n_train/D 에 따라 순위를 부풀린다(mw 만 LW 사용)
    ev = np.clip(np.linalg.eigvalsh(resid.T @ resid / len(resid)), 0, None)
    p = ev[ev > 0] / ev.sum()
    diag['eff_rank_seen'] = float(np.exp(-(p * np.log(p)).sum()))
    diag['lw_shrinkage'] = float(lw.shrinkage_)

    return {'split_id': split['split_id'], 'npz': str(npz_path), 'manifest': str(manifest_csv),
            'dim': int(emb.shape[1]), 'seed': seed, 'episodes': episodes, 'Ks': list(Ks),
            'n_items': {r: int((role == r).sum()) for r in ROLES},
            'kshot': kshot, 'attributes': attr_out, 'diagnostics': diag}


# ---------------------------------------------------------------------------
# selftest (합성 데이터, 데이터·torch 불필요)
# ---------------------------------------------------------------------------
def _synth_manifest(n_classes=28, n_groups=14, per_group=3) -> pd.DataFrame:
    rows = [(f'/syn/c{c:02d}_g{g:02d}_{i}.jpg', f'c{c:02d}', f'c{c:02d}_g{g:02d}', f'd{g % 2}')
            for c in range(n_classes) for g in range(n_groups) for i in range(per_group)]
    # 다중 라벨 group: 한 사진(group)에 이웃한 두 클래스가 같이 찍힌 경우
    rows += [(f'/syn/mix{c:02d}_{j}.jpg', f'c{c + j:02d}', f'mix{c:02d}', 'd0')
             for c in range(0, n_classes - 1, 2) for j in (0, 1)]
    return pd.DataFrame(rows, columns=['path', 'label', 'group', 'domain'])


def _synth_eval(tmp, name, df, emb, split, feature_table=None, classifiers=CLASSIFIERS, **npz_extra):
    m, e = Path(tmp) / f'{name}.csv', Path(tmp) / f'{name}.npz'
    df.to_csv(m, index=False)
    np.savez(e, path=df['path'].to_numpy(dtype=str), emb=emb.astype(np.float32), **npz_extra)
    return evaluate(e, m, split, (1, 5), episodes=40, feature_table=feature_table, classifiers=classifiers)


def selftest() -> None:
    import tempfile

    from .class_features import DEFAULT_TRAIN_ATTRIBUTES

    rng = np.random.default_rng(0)
    df = _synth_manifest()

    # (1) 분할: 결정적, JSON 왕복, §2.1 불변식, unseen group 의 seen 항목은 excluded
    splits = make_splits(df, n_splits=3, min_groups_unseen=12, seed=0)
    assert splits == make_splits(df, n_splits=3, min_groups_unseen=12, seed=0)
    assert json.loads(json.dumps(splits)) == splits
    n_excluded = 0
    for sp in splits:
        check_split(df, sp)
        roles = item_roles(df, sp)
        is_unseen = df['label'].isin(sp['unseen_classes'])
        assert len(sp['unseen_classes']) == 10
        in_ug = df['group'].isin(set(df.loc[is_unseen, 'group']))
        assert not roles[in_ug].isin(['train', 'val', 'test']).any()
        assert (roles[in_ug & ~is_unseen] == 'excluded').all()
        assert set(sp['seen_classes']) <= set(df.loc[roles == 'train', 'label'])
        n_excluded += int((roles == 'excluded').sum())
    assert n_excluded > 0, '합성 manifest 가 excluded 경로를 타지 않았다'
    bad = json.loads(json.dumps(splits[0]))
    g = df.loc[item_roles(df, bad) == 'excluded', 'group'].iloc[0]  # unseen·seen 이 섞인 group 을 train 으로 새게 한다
    bad['unseen_groups'].remove(g)
    bad['group_roles'][g] = 'train'
    try:
        check_split(df, bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('check_split 이 unseen group 누수를 못 잡았다')

    # (2) 에피소드: 같은 입력이면 동일, 클래스별 서포트/query group 분리, 서포트 group K개
    udf = df[item_roles(df, splits[0]) == 'unseen']
    for K in (1, 5, 12):
        s, q, cls = sample_episode(udf, K, np.random.default_rng([0, 0, K, 3]))
        s2, q2, cls2 = sample_episode(udf, K, np.random.default_rng([0, 0, K, 3]))
        assert np.array_equal(s, s2) and np.array_equal(q, q2) and cls == cls2
        assert len(cls) == 10 and len(s) == K * len(cls) and not set(s) & set(q)
        for c in cls:
            sg = set(df.loc[s, 'group'][df.loc[s, 'label'] == c])
            qg = set(df.loc[q, 'group'][df.loc[q, 'label'] == c])
            assert len(sg) == K and qg and not sg & qg
    assert sample_episode(udf, 15, rng)[2] == []  # group 15개인 클래스뿐 -> K=15 는 자격 없음

    # (3) Δ_W 부호: 모든 클래스가 공유하는 비등방 클래스 내 잡음이면 mw >> cos, 등방이면 차이 작음
    D = 32
    y = np.searchsorted(np.unique(df['label']), df['label'])
    mu = rng.normal(size=(28, D)) * 0.5
    rot = np.linalg.qr(rng.normal(size=(D, D)))[0]
    offset = np.eye(D)[0] * 6.0  # 공통 offset: L2 정규화를 거의 선형으로 만든다
    aniso = np.r_[[4.0] * 4, [0.2] * (D - 4)]
    iso = np.full(D, 1.0)
    with tempfile.TemporaryDirectory() as tmp:
        out = {}
        for name, sd in (('aniso', aniso), ('iso', iso)):
            emb = offset + mu[y] + (rng.normal(size=(len(df), D)) * sd) @ rot.T
            out[name] = _synth_eval(tmp, name, df, emb, splits[0])
        for name, r in out.items():
            accs = {c: round(v['acc']['mean'], 3) for c, v in r['kshot']['5']['classifiers'].items()}
            print(f'[selftest] {name}: K=5 acc {accs}, delta_mw_cos {r["diagnostics"]["delta_mw_cos"]}')
        assert out['aniso']['diagnostics']['delta_mw_cos']['5'] > 20
        assert abs(out['iso']['diagnostics']['delta_mw_cos']['5']) < 5
        k5 = out['aniso']['kshot']['5']
        assert set(k5['gfsl']) == {'acc_seen', 'acc_unseen', 'H'} and set(k5['classifiers']['cos']['acc_by_domain']) == {'d0', 'd1'}
        s_, u_ = k5['gfsl']['acc_seen']['mean'], k5['gfsl']['acc_unseen']['mean']
        assert abs(k5['gfsl']['H']['mean'] - 2 * s_ * u_ / (s_ + u_)) < 1e-12
        # 유효 순위(축소 없는 Σ_W): 큰 잡음 4축 비등방 < 등방
        er = {k: r['diagnostics']['eff_rank_seen'] for k, r in out.items()}
        assert er['aniso'] < 10 < er['iso'], er

        # (4) 특징 구조 합성: 클래스 임베딩 = feature 값 벡터의 합 -> 0-shot probe 가 우연보다 확실히 높다
        attrs = DEFAULT_TRAIN_ATTRIBUTES
        classes = np.unique(df['label'])
        codes = rng.integers(0, 2, size=(len(classes), len(attrs)))
        table = pd.DataFrame(np.where(codes == 1, 'v1', 'v0'), columns=attrs)
        table.insert(0, 'Name', classes)
        table.insert(0, 'class_key', classes)
        table.loc[0, 'Flange'] = 'unknown'
        table.to_csv(Path(tmp) / 'features.csv', index=False)
        W = rng.normal(size=(len(attrs), 2, D))
        emb = W[np.arange(len(attrs)), codes[y]].sum(1) + rng.normal(size=(len(df), D)) * 0.3
        head = {}  # 정답 one-hot 을 'head' 확률로 넣어 npz 정렬·vocab 검사 경로를 탄다
        for ai, a in enumerate(attrs):
            head[f'attrprob__{a}'] = np.eye(2)[codes[y, ai]]
            head[f'attrvocab__{a}'] = np.array(['v0', 'v1'])
        r = _synth_eval(tmp, 'attr', df.assign(class_key=df['label']), emb, splits[0],
                        feature_table=Path(tmp) / 'features.csv', classifiers=('cos', 'mw'), **head)
        at = r['attributes']
        zs = at['zero_shot']['probe']
        print(f"[selftest] 0-shot probe acc {zs['acc']:.3f} (chance {1 / zs['n_classes']:.3f}), "
              f"ceiling {at['zero_shot_ceiling']['value']:.2f}, image_fill mean "
              f"{ {s: round(v['mean'], 3) for s, v in at['image_fill'].items()} }")
        assert zs['acc'] > 1 / zs['n_classes'] + 0.3
        assert at['image_fill']['head']['mean'] == 1.0
        assert at['image_fill']['probe']['mean'] > at['image_fill']['majority']['mean']
        assert r['kshot']['5']['attr_fill_class']['head']['mean'] == 1.0

    # (5) aggregate: 손계산 값, n=1, 정규 표본 CI 커버리지
    a = aggregate([1.0, 2.0, 3.0])
    assert abs(a['mean'] - 2) < 1e-12 and abs(a['std'] - 1) < 1e-12
    assert abs(a['ci95_hi'] - (2 + 4.302652729911275 / math.sqrt(3))) < 1e-9
    assert aggregate([0.7])['ci95_lo'] == aggregate([0.7])['ci95_hi'] == 0.7
    cover = np.mean([(lambda g: g['ci95_lo'] <= 0 <= g['ci95_hi'])(aggregate(rng.normal(size=5)))
                     for _ in range(2000)])
    assert 0.93 < cover < 0.97, cover
    print('[selftest] OK')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser('fewshot', description='unseen few-shot / 0-shot 평가 프로토콜')
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('split')
    s.add_argument('--manifest', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--n-splits', type=int, default=5)
    s.add_argument('--min-groups-unseen', type=int, default=12)
    s.add_argument('--n-unseen', type=int, default=None)
    s.add_argument('--seed', type=int, default=0)
    e = sub.add_parser('evaluate')
    e.add_argument('--emb', required=True)
    e.add_argument('--manifest', required=True)
    e.add_argument('--splits', required=True)
    e.add_argument('--split-id', required=True)
    e.add_argument('--ks', default='1,3,5,10')
    e.add_argument('--episodes', type=int, default=200)
    e.add_argument('--feature-table', default=None)
    e.add_argument('--seed', type=int, default=0)
    e.add_argument('--out', required=True)
    sub.add_parser('selftest')
    a = p.parse_args()

    if a.cmd == 'split':
        df = _read_manifest(a.manifest)
        splits = make_splits(df, a.n_splits, a.n_unseen, a.min_groups_unseen, a.seed)
        for sp in splits:
            check_split(df, sp)
            counts = item_roles(df, sp).value_counts().to_dict()
            print(f"split {sp['split_id']}: unseen {len(sp['unseen_classes'])} / seen {len(sp['seen_classes'])} "
                  f"classes, items {counts}")
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(splits, ensure_ascii=False, indent=1))
    elif a.cmd == 'evaluate':
        splits = json.loads(Path(a.splits).read_text())
        split = next(sp for sp in splits if str(sp['split_id']) == a.split_id)
        res = evaluate(a.emb, a.manifest, split, [int(k) for k in a.ks.split(',')], a.episodes,
                       a.feature_table, a.seed)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
        for k, v in res['kshot'].items():
            if v['n_episodes']:
                print(f"K={k} ({v['n_classes']}-way): " + ', '.join(
                    f"{c} {r['acc']['mean']:.3f}" for c, r in v['classifiers'].items()))
    elif a.cmd == 'selftest':
        selftest()


if __name__ == '__main__':
    main()
