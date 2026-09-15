#!/usr/bin/env python3
"""few-shot 파일럿 실행·집계 (docs/paper/FEWSHOT_ATTRIBUTE_SPEC.md §6.5, §7).

  plan   : 트랙 manifest 로 unseen 분할 5개를 만들어 <out>/<track>/splits.json, splits_summary.csv 를 쓴다.
           트랙 B 는 crop manifest 로 unseen 클래스를 뽑은 뒤 full manifest 의 (label, group) 까지 덮게
           분할을 넓힌다(_cover). 그래야 arc_vits_m05_full 이 같은 splits.json 을 쓴다.
  run    : 레지스트리(TRACKS)의 방법 × 분할마다 train -> export -> evaluate.
           train/export 는 MPS 메모리를 매번 비우려고 `python3 -m src.fewshot_train` subprocess, evaluate 는 import.
           출력이 이미 있는 단계는 건너뛴다(재시작 가능). 실패는 errors.jsonl 에 남기고 다음 방법·분할로 넘어간다.
             동결: <out>/<track>/<method>/emb.npz (분할과 무관, 1회)
             학습: <out>/<track>/<method>/split<r>/{model.pt,history.json,summary.json,emb.npz,train.log}
             결과: <out>/<track>/<method>/split<r>.json      단계별 wall time: <out>/run.log
  report : <out>/REPORT.md, kcurve_<track>.png, tables/*.csv (tables/long.csv = 분할별 원값)

  python3 tools/fewshot_suite.py plan   --track B [--root .../data/fewshot_local] [--out .../outputs/fewshot]
  python3 tools/fewshot_suite.py run    --track S --methods in_vits,arcattr_vits_m05 --splits 0-4 [--epochs 10 --episodes 200]
  python3 tools/fewshot_suite.py report [--out .../outputs/fewshot]
  python3 tools/fewshot_suite.py selftest
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fewshot import (CLASSIFIERS, ROLES, _read_manifest, aggregate, check_split, evaluate,  # noqa: E402
                         item_roles, make_splits)

DEFAULT_ROOT = ROOT.parents[1] / 'data' / 'fewshot_local'
DEFAULT_OUT = ROOT.parent / 'outputs' / 'fewshot'

# ---------------------------------------------------------------------------
# 방법 레지스트리 (§7). 이 파일 한 곳에만 둔다.
# ---------------------------------------------------------------------------
N_SPLITS, SPLIT_SEED, TRAIN_SEED = 5, 0, 0
EPOCHS, BATCH, LR, HEAD_LR, EPISODES = 10, 32, 5e-5, 1e-3, 200
VITS = 'vit_small_patch16_224.augreg_in21k_ft_in1k'
VITB = 'vit_base_patch16_224.augreg2_in21k_ft_in1k'
DINO = 'vit_base_patch14_dinov2.lvd142m'
FROZEN = {'in_vitb': {'pretrained': VITB}, 'dino_vitb': {'pretrained': DINO}, 'in_vits': {'pretrained': VITS}}


def _arc(margin, model=VITS, **kw):
    return {'loss': 'arcface', 'model': model, 'margin': margin, **kw}


TRACKS = {
    'B': {'manifest': 'trackB/manifest_crop.csv', 'ks': (1, 3, 5, 10), 'min_groups_unseen': 12, 'epoch_size': 3000,
          'feature_table': None, 'ref': 'arc_vits_m05',
          'methods': {**FROZEN,
                      'ce_vits': {'loss': 'ce', 'model': VITS},
                      'proto_vits': {'loss': 'protonet', 'model': VITS},
                      'arc_vits_m05': _arc(0.5), 'arc_vits_m02': _arc(0.2), 'arc_vits_m00': _arc(0.0),
                      'arc_vitb_m05': _arc(0.5, VITB),
                      'arc_vits_m05_full': _arc(0.5, manifest='trackB/manifest_full.csv')}},  # 학습·평가 모두 full
    'S': {'manifest': 'trackS/manifest.csv', 'ks': (1, 2), 'min_groups_unseen': 1, 'epoch_size': 600,
          'feature_table': 'configs/class_features.csv', 'ref': 'arcattr_vits_m05',
          'methods': {**FROZEN,
                      'arc_vits_m05': _arc(0.5),
                      'arcattr_vits_m05': {'loss': 'arcface_attr', 'model': VITS, 'margin': 0.5, 'attr_weight': 1.0}}},
}

# 보고서 구성 (§3 기준선 칸, §4 ablation)
BASELINES = [('ImageNet ViT-B', 'in_vitb', 'cos'), ('DINOv2 ViT-B', 'dino_vitb', 'cos'), ('ProtoNet', 'proto_vits', 'euc'),
             ('SimpleShot (CE)', 'ce_vits', 'ss'), ('ArcFace 갤러리', 'arc_vits_m05', 'cos')]
ABLATIONS = [('margin (m = 0.0 / 0.2 / 0.5)', 'B', ['arc_vits_m00', 'arc_vits_m02', 'arc_vits_m05']),
             ('백본 크기 (ViT-S/16 vs ViT-B/16)', 'B', ['in_vits', 'in_vitb', 'arc_vits_m05', 'arc_vitb_m05']),
             ('crop 유무 (크롭 vs 원본 전체)', 'B', ['arc_vits_m05', 'arc_vits_m05_full']),
             ('특징 헤드 유무 (w = 0 / 1)', 'S', ['arc_vits_m05', 'arcattr_vits_m05'])]
SOURCES = ('head', 'probe', 'majority', 'nearest_seen')
# dataviz 기본 범주 팔레트 1~7번(밝은 배경 검증 통과). 색은 레지스트리 순서에 묶여 방법이 빠져도 바뀌지 않는다.
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
MARKERS = 'osD^vP*X'


def _manifests(cfg):
    return list(dict.fromkeys([cfg['manifest']] + [m['manifest'] for m in cfg['methods'].values() if 'manifest' in m]))


def _parse_ids(s):
    out = []
    for part in s.split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b or a) + 1))
    return out


def _log(out, msg):
    line = f'{datetime.now():%Y-%m-%d %H:%M:%S} {msg}'
    print(line, flush=True)
    with open(out / 'run.log', 'a', encoding='utf-8') as f:
        f.write(line + '\n')


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def _cover(split, allg) -> tuple[int, int]:
    """주 manifest 로 만든 split 을 여러 manifest 의 (label, group) 합집합에 맞춘다(제자리 수정).

    unseen 라벨 항목이 하나라도 있는 group 은 전부 unseen_groups 로 간다(주 manifest 에서 seen 역할이던
    group 이면 그 항목은 excluded 가 된다: §2.1 규칙 2 를 합집합에 적용). 주 manifest 에 없던 나머지 group 은
    make_splits 와 같은 비율(val 0.1 / test 0.1 / 나머지 train)로 역할을 받는다. 반환: (옮긴 group 수, 새 group 수).
    """
    ug = set(allg.loc[allg['label'].isin(split['unseen_classes']), 'group'])
    roles = split['group_roles']
    moved = sorted(ug & set(roles))
    for g in moved:
        del roles[g]
    new = sorted(set(allg['group']) - ug - set(roles))
    perm = np.random.default_rng([split['seed'], int(split['split_id']), 1]).permutation(new).tolist() if new else []
    n_val = n_test = int(round(len(perm) * 0.1))
    roles.update({g: 'val' if i < n_val else 'test' if i < n_val + n_test else 'train' for i, g in enumerate(perm)})
    split['unseen_groups'] = sorted(ug)
    lost = set(split['seen_classes']) - set(allg.loc[allg['group'].map(roles).eq('train'), 'label'])
    if lost:
        warnings.warn(f"split {split['split_id']}: train group 을 잃은 seen 클래스 {sorted(lost)}")
    return len(moved), len(new)


def plan(track, root, out):
    cfg = TRACKS[track]
    dfs = {m: _read_manifest(root / m) for m in _manifests(cfg)}
    splits = make_splits(dfs[cfg['manifest']], N_SPLITS, None, cfg['min_groups_unseen'], SPLIT_SEED)
    allg = pd.concat([d[['label', 'group']] for d in dfs.values()])
    rows = []
    for sp in splits:
        moved, added = _cover(sp, allg)
        for name, d in dfs.items():
            check_split(d, sp)
            roles = item_roles(d, sp)
            ng = d[roles == 'unseen'].groupby('label')['group'].nunique()
            rows.append({'split': sp['split_id'], 'manifest': name, 'n_unseen_classes': len(sp['unseen_classes']),
                         'n_seen_classes': len(sp['seen_classes']),
                         **{f'n_{r}': int((roles == r).sum()) for r in ROLES},
                         **{f'way_K{k}': int((ng >= k + 1).sum()) for k in cfg['ks']},
                         'groups_moved_to_unseen': moved, 'groups_added': added,
                         'unseen_classes': ' '.join(sp['unseen_classes'])})
    tdir = out / track
    tdir.mkdir(parents=True, exist_ok=True)
    path, text = tdir / 'splits.json', json.dumps(splits, ensure_ascii=False, indent=1)
    if path.exists() and json.loads(path.read_text()) != splits:
        raise SystemExit(f'{path} 가 이미 있고 지금 만든 분할과 다르다. 결과 폴더(--out)를 새로 쓰거나 지우고 다시 하라.')
    path.write_text(text)
    summ = pd.DataFrame(rows)
    summ.to_csv(tdir / 'splits_summary.csv', index=False)
    print(summ.drop(columns='unseen_classes').to_string(index=False))
    for sp in splits:
        print(f"split {sp['split_id']} unseen: {' '.join(sp['unseen_classes'])}")
    _log(out, f'plan track={track} -> {path} (check_split ok on {list(dfs)})')


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def _step(out, tag, target, fn) -> bool:
    """target 이 있으면 건너뛴다. 없으면 fn() 으로 만든다. 실패는 errors.jsonl 에 남기고 False."""
    track, method, split, step = tag
    head = f'track={track} method={method} split={split} step={step}'
    if target.exists():
        _log(out, f'{head} skip (exists)')
        return True
    t0 = time.time()
    try:
        fn()
        assert target.exists(), f'{target} 가 만들어지지 않았다'
    except Exception as e:  # 한 방법·분할의 실패가 나머지를 멈추지 않게 한다
        sec = time.time() - t0
        _log(out, f'{head} FAILED {sec:.1f}s: {(str(e).splitlines() or [type(e).__name__])[0]}')
        with open(out / 'errors.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps({'time': f'{datetime.now():%Y-%m-%d %H:%M:%S}', 'track': track, 'method': method,
                                'split': split, 'step': step, 'target': str(target), 'seconds': round(sec, 1),
                                'error': f'{type(e).__name__}: {e}',
                                'traceback': traceback.format_exc()[-3000:]}, ensure_ascii=False) + '\n')
        return False
    _log(out, f'{head} ok {time.time() - t0:.1f}s')
    return True


def _sub(argv, log_path):
    """python3 -m src.fewshot_train <argv> (cwd=ROOT), 출력은 log_path 에 덧붙인다."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-m', 'src.fewshot_train', *map(str, argv)]
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                           env={**os.environ, 'HF_HUB_OFFLINE': os.environ.get('HF_HUB_OFFLINE', '1')})
    if p.returncode:
        tail = [s for s in log_path.read_text(encoding='utf-8', errors='replace').splitlines() if s.strip()][-15:]
        raise RuntimeError(f'exit {p.returncode}: {tail[-1] if tail else ""} ({log_path})\n' + '\n'.join(tail))


def _stale(path, want) -> dict:
    """이미 있는 출력(summary.json 의 cfg 또는 결과 JSON)의 설정이 요청과 다르면 {키: (기존, 요청)}."""
    if not path.exists():
        return {}
    old = json.loads(path.read_text())
    old = old.get('cfg', old)
    return {k: (old.get(k), v) for k, v in want.items() if old.get(k) != v}


def _export(argv, target, log_path):
    """반쯤 쓴 emb.npz 가 '완료'로 보이지 않게 임시 이름으로 쓰고 옮긴다."""
    tmp = target.with_name('emb.partial.npz')
    return lambda: (_sub([*argv, '--out', tmp], log_path), os.replace(tmp, target))


def run(track, methods, split_ids, root, out, epochs=EPOCHS, epoch_size=None, episodes=EPISODES):
    cfg = TRACKS[track]
    unknown = set(methods) - set(cfg['methods'])
    assert not unknown, f'트랙 {track} 레지스트리에 없는 방법: {sorted(unknown)} (있는 것: {list(cfg["methods"])})'
    spath = out / track / 'splits.json'
    if not spath.exists():
        plan(track, root, out)
    splits = {sp['split_id']: sp for sp in json.loads(spath.read_text())}
    assert set(split_ids) <= set(splits), f'splits.json 에 없는 분할: {sorted(set(split_ids) - set(splits))}'
    ft = str(ROOT / cfg['feature_table']) if cfg['feature_table'] else None
    epoch_size = epoch_size or cfg['epoch_size']
    _log(out, f'run track={track} methods={",".join(methods)} splits={split_ids} epochs={epochs} '
              f'epoch_size={epoch_size} episodes={episodes}')

    for name in methods:
        m = cfg['methods'][name]
        mdir = out / track / name
        manifest = root / m.get('manifest', cfg['manifest'])
        ok_frozen = True
        if 'pretrained' in m:
            emb = mdir / 'emb.npz'
            ok_frozen = _step(out, (track, name, None, 'export'), emb,
                              _export(['export', '--manifest', manifest, '--pretrained', m['pretrained']],
                                      emb, mdir / 'export.log'))
        for r in split_ids:
            ok = ok_frozen
            target = mdir / f'split{r}.json'
            # 출력이 있으면 건너뛰므로, 다른 설정(예: smoke 의 epochs 1)으로 만든 출력을 조용히 재사용하지 않게 막는다
            stale = {**(_stale(mdir / f'split{r}' / 'summary.json',
                               {'epochs': epochs, 'epoch_size': epoch_size, 'seed': TRAIN_SEED})
                        if 'loss' in m and (mdir / f'split{r}' / 'model.pt').exists() else {}),
                     **_stale(target, {'episodes': episodes})}
            if stale:
                raise SystemExit(f'{track}/{name}/split{r} 의 기존 출력이 요청과 다른 설정이다 {stale}. '
                                 f'결과 폴더(--out)를 새로 쓰거나 그 출력을 지우고 다시 하라.')
            if 'loss' in m:
                rdir = mdir / f'split{r}'
                emb = rdir / 'emb.npz'
                argv = ['train', '--manifest', manifest, '--splits', spath, '--split-id', r, '--loss', m['loss'],
                        '--model-name', m['model'], '--margin', m.get('margin', 0.5),
                        '--attr-weight', m.get('attr_weight', 1.0), '--epochs', epochs, '--epoch-size', epoch_size,
                        '--batch-size', BATCH, '--lr', LR, '--head-lr', HEAD_LR, '--seed', TRAIN_SEED, '--out', rdir]
                if ft:
                    argv += ['--feature-table', ft]
                ok = (_step(out, (track, name, r, 'train'), rdir / 'model.pt', lambda: _sub(argv, rdir / 'train.log'))
                      and _step(out, (track, name, r, 'export'), emb,
                                _export(['export', '--manifest', manifest, '--ckpt', rdir / 'model.pt'],
                                        emb, rdir / 'train.log')))
            if not ok:
                continue

            def _eval(emb=emb, target=target, r=r):
                res = evaluate(emb, manifest, splits[r], cfg['ks'], episodes, ft, seed=0)
                tmp = target.with_suffix('.partial')
                tmp.write_text(json.dumps({'track': track, 'method': name, **res}, ensure_ascii=False, indent=1))
                os.replace(tmp, target)
            _step(out, (track, name, r, 'evaluate'), target, _eval)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _flatten(t, m, r, res):
    """결과 JSON 한 개 -> (track, method, split, metric, K, key, value) 행. 모든 표는 이 행의 분할 집계다."""
    rows = []

    def add(metric, k, key, v):
        rows.append((t, m, r, metric, str(k), key, np.nan if v is None else float(v)))

    for k, v in res['kshot'].items():
        add('n_way', k, '', v['n_classes'])
        for c, cv in v.get('classifiers', {}).items():
            add('acc', k, c, cv['acc']['mean'])
            add('f1', k, c, cv['f1']['mean'])
            add('acc_ep_hw', k, c, cv['acc']['ci95_hi'] - cv['acc']['mean'])  # 에피소드 수준 CI 반폭(부록)
            add('f1_ep_hw', k, c, cv['f1']['ci95_hi'] - cv['f1']['mean'])
            for d, dv in cv.get('acc_by_domain', {}).items():
                add('acc_dom', k, f'{c}:{d}', dv['acc'])
        for x, g in v.get('gfsl', {}).items():
            add(f'gfsl_{x}', k, '', g['mean'])
        for s, fv in v.get('attr_fill_class', {}).items():
            add('fill_class', k, s, fv['mean'])
    diag = res['diagnostics']
    for k, v in diag['delta_mw_cos'].items():
        add('delta_w', k, '', v)
    add('trace_ratio', '', '', diag['trace_ratio_unseen'])
    add('eff_rank', '', '', diag['eff_rank_seen'])
    add('lw_shrinkage', '', '', diag.get('lw_shrinkage'))
    at = res.get('attributes')
    if at:
        for s, fv in at['image_fill'].items():
            add('fill_image', '', f'{s}:mean', fv['mean'])
            for a, av in fv['per_feature'].items():
                add('fill_image', '', f'{s}:{a}', av['acc'])
        for s, z in at['zero_shot'].items():
            add('zs_acc', '', s, z['acc'])
            add('zs_f1', '', s, z['f1'])
            add('zs_ncls', '', s, z['n_classes'])
        add('zs_ceiling', '', '', at['zero_shot_ceiling']['value'])
        for a, u in at['probe_unreachable'].items():
            add('unreachable', '', a, u['n_unseen_items'])
    return rows


def _cell(s, scale=100.0, d=1):
    """분할 평균 ± 95% CI 반폭. 분할이 N_SPLITS 개보다 적으면 (n=k)."""
    if s is None or len(s) == 0:
        return '—'
    a = aggregate(s.tolist())
    txt = f"{a['mean'] * scale:.{d}f} ± {(a['ci95_hi'] - a['mean']) * scale:.{d}f}"
    return txt if a['n'] >= N_SPLITS else f"{txt} (n={a['n']})"


def _kcurve(out, t, cfg, v, present):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    groups = [('frozen', [m for m, s in cfg['methods'].items() if 'pretrained' in s]),
              ('trained on seen', [m for m, s in cfg['methods'].items() if 'loss' in s])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (title, gm) in zip(axes, groups):
        for i, m in enumerate(gm):  # 색·마커는 레지스트리 위치로 고정
            pts = [(K, aggregate(s.tolist())) for K in cfg['ks'] if len(s := v(t, m, 'acc', K, 'cos'))]
            if m not in present or not pts:
                continue
            ax.errorbar([k for k, _ in pts], [100 * a['mean'] for _, a in pts],
                        yerr=[100 * (a['ci95_hi'] - a['mean']) for _, a in pts], color=PALETTE[i],
                        marker=MARKERS[i], ms=6, lw=2, capsize=3, label=f"{m} (n={max(a['n'] for _, a in pts)})")
        ax.set_title(f'{title}', fontsize=10, color='#52514e')
        ax.set_xticks(cfg['ks'])
        ax.set_xlabel('K (support images per unseen class)')
        ax.grid(axis='y', color='#e6e5e0', lw=0.8)
        ax.spines[['top', 'right']].set_visible(False)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel('unseen accuracy, cos prototype (%)')
    fig.suptitle(f'Track {t}: unseen all-way K-shot accuracy (mean ± 95% CI over splits)', fontsize=11)
    fig.savefig(out / f'kcurve_{t}.png', dpi=150, bbox_inches='tight', facecolor='#fcfcfb')
    plt.close(fig)


def report(out, root):
    L, meta, attrs_of, dropped_of = [], [], {}, {}
    for t, cfg in TRACKS.items():
        for m in cfg['methods']:
            for r in range(N_SPLITS):
                p = out / t / m / f'split{r}.json'
                if not p.exists():
                    continue
                res = json.loads(p.read_text())
                L += _flatten(t, m, r, res)
                if res.get('attributes'):
                    attrs_of[t] = res['attributes']['attributes']
                    dropped_of[t] = res['attributes']['dropped_constant']
                sp = out / t / m / f'split{r}' / 'summary.json'
                sm = json.loads(sp.read_text()) if sp.exists() else {}
                meta.append({'track': t, 'method': m, 'split': r, 'episodes': res['episodes'], 'dim': res['dim'],
                             'n_unseen_items': res['n_items']['unseen'],
                             'epochs': sm.get('cfg', {}).get('epochs'), 'epoch_size': sm.get('cfg', {}).get('epoch_size'),
                             'best_epoch': sm.get('best_epoch'), 'best_val_acc': sm.get('best_val_acc'),
                             'train_s': round(sm['seconds']) if sm else None,
                             'protonet_excluded': ' '.join(sm['protonet']['ineligible']) if 'protonet' in sm else ''})
    L = pd.DataFrame(L, columns=['track', 'method', 'split', 'metric', 'K', 'key', 'value'])
    V = {k: g.set_index('split')['value'].astype(float) for k, g in L.groupby(['track', 'method', 'metric', 'K', 'key'])}

    def v(t, m, metric, k='', key=''):
        return V.get((t, m, metric, str(k), key), pd.Series(dtype=float)).dropna()

    tables = out / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    L.to_csv(tables / 'long.csv', index=False)
    md = ['# few-shot 파일럿 결과 보고서', '',
          f'- 생성 {datetime.now():%Y-%m-%d %H:%M}, 결과 루트 `{out}`, 데이터 루트 `{root}`. 명세: `docs/paper/FEWSHOT_ATTRIBUTE_SPEC.md`.',
          f'- 셀 = 분할 평균 ± 95% CI 반폭(t, df = n−1). 분할 값은 그 분할의 에피소드 평균이다(§2.4). '
          f'정확도·F1·특징 정확도 단위는 %. 분할이 {N_SPLITS}개보다 적으면 `(n=k)`. `—` = 결과 없음.',
          '- 분할별 원값은 `tables/long.csv`, 표마다 `tables/<이름>.csv`.', '']

    def tab(name, df, note=None):
        df.to_csv(tables / f'{name}.csv', index=False)
        if note:
            md.append(note)
            md.append('')
        md.append(df.to_markdown(index=False) if len(df) else '_(결과 없음)_')
        md.append('')

    present = {t: [m for m in cfg['methods'] if ((L['track'] == t) & (L['method'] == m)).any()]
               for t, cfg in TRACKS.items()}

    md += ['## 1. 데이터와 분할', '']
    for t, cfg in TRACKS.items():
        cc = root / Path(cfg['manifest']).parent / 'class_counts.csv'
        md += [f'### 트랙 {t} 클래스별 장수 (`{cc.name}`)', '']
        if cc.exists():
            tab(f'data_{t}', pd.read_csv(cc))
        else:
            md += [f'_{cc} 없음_', '']
        ss = out / t / 'splits_summary.csv'
        md += [f'### 트랙 {t} 분할 요약 (plan)', '']
        if ss.exists():
            tab(f'splits_{t}', pd.read_csv(ss), 'way_K = 그 K 에서 평가 가능한 unseen 클래스 수(group ≥ K+1). '
                                                 'groups_moved_to_unseen / groups_added = 다른 manifest 를 덮으려고 바꾼 group 수(_cover).')
        else:
            md += ['_splits_summary.csv 없음 (plan 을 돌리지 않았다)_', '']

    md += ['## 2. unseen K-shot 정확도와 macro-F1 (표현 × 분류기)', '',
           '`n` = 결과가 있는 분할 수, `way` = 분할별 N-way 범위.', '']
    for t, cfg in TRACKS.items():
        for K in cfg['ks']:
            for metric, title in (('acc', '정확도'), ('f1', 'macro-F1')):
                rows = []
                for m in present[t]:
                    way = v(t, m, 'n_way', K)
                    lo, hi = (int(way.min()), int(way.max())) if len(way) else (None, None)
                    rows.append({'method': m, 'n': len(v(t, m, metric, K, 'cos')),
                                 'way': '—' if lo is None else str(lo) if lo == hi else f'{lo}–{hi}',
                                 **{c: _cell(v(t, m, metric, K, c)) for c in CLASSIFIERS}})
                md += [f'### 트랙 {t}, K={K}, {title}', '']
                tab(f'kshot_{t}_K{K}_{metric}', pd.DataFrame(rows))
        if present[t]:
            _kcurve(out, t, cfg, v, present[t])
            md += [f'![트랙 {t} K-곡선](kcurve_{t}.png)', '']

    md += ['## 3. 기준선 표 (§3)', '',
           'Simple CNAPS 헤드 = 각 표현 + `mtr`(정규화 전 특징 공간의 계층 정규화 Mahalanobis 헤드만; FiLM 적응은 '
           '구현하지 않았다. L2 공간 변형 `mt` 는 βI 가 지배해 euc 와 거의 같아지므로 §2 표에만 싣는다). '
           '트랙 B 는 특징표가 없어 제안 = ArcFace(`arc_vits_m05`)이다. 정확도만 싣는다(F1 은 §2).', '']
    for t, cfg in TRACKS.items():
        base = BASELINES + [('제안', cfg['ref'], 'cos')] + [('Simple CNAPS 헤드', m, 'mtr') for m in cfg['methods']]
        seen, rows = set(), []
        for name, m, c in base:
            if m not in cfg['methods'] or (m, c) in seen:
                continue
            seen.add((m, c))
            row = {'기준선': name, 'method': m, 'classifier': c, **{f'K={K}': _cell(v(t, m, 'acc', K, c)) for K in cfg['ks']}}
            if cfg['feature_table']:
                row['0-shot (head)'] = _cell(v(t, m, 'zs_acc', '', 'head')) if name == '제안' else ''
            rows.append(row)
        md += [f'### 트랙 {t}', '']
        tab(f'baselines_{t}', pd.DataFrame(rows))

    md += ['## 4. Ablation (§4, cos 분류기 정확도)', '']
    for i, (title, t, ms) in enumerate(ABLATIONS):
        cfg = TRACKS[t]
        rows = []
        for m in ms:
            row = {'method': m, **{f'K={K}': _cell(v(t, m, 'acc', K, 'cos')) for K in cfg['ks']}}
            if cfg['feature_table']:
                row.update({'0-shot probe': _cell(v(t, m, 'zs_acc', '', 'probe')),
                            '0-shot head': _cell(v(t, m, 'zs_acc', '', 'head')),
                            '특징 채우기 probe': _cell(v(t, m, 'fill_image', '', 'probe:mean')),
                            '특징 채우기 head': _cell(v(t, m, 'fill_image', '', 'head:mean'))})
            rows.append(row)
        note = ('`arc_vits_m05_full` 은 manifest_full 로 학습·평가해 query 집합이 다르다. 짝지은 비교가 아니다.'
                if 'crop' in title else None)
        md += [f'### 트랙 {t}: {title}', '']
        tab(f'ablation_{i}_{t}', pd.DataFrame(rows), note)

    for t, cfg in TRACKS.items():
        if not cfg['feature_table']:
            continue
        attrs = attrs_of.get(t, [])
        md += [f'## 5. 트랙 {t} 특징표', '',
               f'`mean` = 상수 아닌 feature {len(attrs)}개의 평균. 제외(이 트랙 모든 클래스에서 값이 같음): '
               f"{', '.join(dropped_of.get(t, [])) or '없음'}.", '',
               '### 이미지 단위 특징표 채우기 (unseen 이미지, feature 별 정확도)', '',
               '`head` = 제안 모델의 attribute 헤드, `probe` = seen train L2 임베딩의 feature 별 로지스틱 회귀, '
               '`majority` = seen 클래스 최빈값, `nearest_seen` = 가장 가까운 seen centroid 의 행. '
               '`unreachable` = seen train 에 없던 값이 정답인 unseen 항목 수(probe 는 도달 불가, 분할 합).', '']
        for m in present[t]:
            rows = [{'feature': a, **{s: _cell(v(t, m, 'fill_image', '', f'{s}:{a}')) for s in SOURCES},
                     'unreachable': int(v(t, m, 'unreachable', '', a).sum()) if a != 'mean' else ''}
                    for a in attrs + ['mean']]
            md += [f'#### {m}', '']
            tab(f'attr_image_{t}_{m}', pd.DataFrame(rows))
        md += ['### 클래스 단위 특징표 채우기 (K 서포트 확률 평균 → 클래스 행, 상수 아닌 feature 평균)', '']
        rows = [{'method': m, 'source': s, **{f'K={K}': _cell(v(t, m, 'fill_class', K, s)) for K in cfg['ks']}}
                for m in present[t] for s in ('head', 'probe') if any(len(v(t, m, 'fill_class', K, s)) for K in cfg['ks'])]
        tab(f'attr_class_{t}', pd.DataFrame(rows))
        md += ['### 0-shot (특징표 한 행만, §2.3)', '',
               '`ceiling` = 알려진 feature 로 다른 모든 unseen 클래스와 구분되는 클래스 비율(원리상 상한).', '']
        rows = [{'method': m, 'source': s, 'acc': _cell(v(t, m, 'zs_acc', '', s)), 'macro-F1': _cell(v(t, m, 'zs_f1', '', s)),
                 'n_classes': _cell(v(t, m, 'zs_ncls', '', s), 1, 1), 'ceiling': _cell(v(t, m, 'zs_ceiling'))}
                for m in present[t] for s in ('head', 'probe') if len(v(t, m, 'zs_acc', '', s))]
        tab(f'zero_shot_{t}', pd.DataFrame(rows))

    md += ['## 6. 진단 (§5)', '',
           'Δ_W(K) = acc_mw − acc_cos (%p). trace ratio = unseen query 의 tr(S_W)/tr(S_B). '
           'eff rank = seen train 의 축소 없는 pooled Σ_W 유효 순위. LW shrinkage = `mw` 가 쓰는 Ledoit-Wolf 축소 강도.', '']
    for t, cfg in TRACKS.items():
        rows = [{'method': m, **{f'Δ_W K={K}': _cell(v(t, m, 'delta_w', K), 1.0) for K in cfg['ks']},
                 'trace ratio': _cell(v(t, m, 'trace_ratio'), 1.0, 3), 'eff rank': _cell(v(t, m, 'eff_rank'), 1.0),
                 'LW shrinkage': _cell(v(t, m, 'lw_shrinkage'), 1.0, 3)}
                for m in present[t]]
        md += [f'### 트랙 {t}', '']
        tab(f'diagnostics_{t}', pd.DataFrame(rows))

    md += ['## 7. GFSL (§2.2 보조, cos)', '', 'seen 프로토타입 ∪ unseen K-shot 프로토타입, query = seen test ∪ unseen query. '
           '분할의 H = 그 분할 에피소드 평균 acc_seen·acc_unseen 의 조화평균(에피소드별 H 의 평균이 아니다).', '']
    for t, cfg in TRACKS.items():
        rows = [{'method': m, 'K': K, **{x: _cell(v(t, m, f'gfsl_{x}', K)) for x in ('acc_seen', 'acc_unseen', 'H')}}
                for m in present[t] for K in cfg['ks'] if len(v(t, m, 'gfsl_H', K))]
        md += [f'### 트랙 {t}', '']
        tab(f'gfsl_{t}', pd.DataFrame(rows))

    md += ['## 8. 기준 방법 대비 짝지은 차이 (cos 정확도, %p)', '',
           '셀 = 같은 분할 위 차이(방법 − 기준)의 평균 ± 95% CI 반폭 (이긴 분할 수 / 공통 분할 수).', '']
    for t, cfg in TRACKS.items():
        ref, rows = cfg['ref'], []
        for m in present[t]:
            if m == ref:
                continue
            row = {'method': m}
            for K in cfg['ks']:
                d = (v(t, m, 'acc', K, 'cos') - v(t, ref, 'acc', K, 'cos')).dropna()
                a = aggregate(d.tolist())
                row[f'K={K}'] = (f"{100 * a['mean']:+.1f} ± {100 * (a['ci95_hi'] - a['mean']):.1f} "
                                 f"({int((d > 0).sum())}/{len(d)})") if len(d) else '—'
            rows.append(row)
        note = '`arc_vits_m05_full` 은 query 집합이 달라(원본 전체) 같은 에피소드 비교가 아니다.' if t == 'B' else None
        md += [f'### 트랙 {t} (기준: `{ref}`)', '']
        tab(f'paired_{t}', pd.DataFrame(rows), note)

    md += ['## 9. 트랙 B 도메인별 정확도 (cos)', '', '분할 값 = 그 분할 모든 에피소드의 도메인별 query 정확도(합산).', '']
    doms = sorted({k.split(':', 1)[1] for k in L.loc[(L['track'] == 'B') & (L['metric'] == 'acc_dom'), 'key']
                   if k.startswith('cos:')})
    rows = [{'method': m, 'K': K, **{d: _cell(v('B', m, 'acc_dom', K, f'cos:{d}')) for d in doms}}
            for m in present['B'] for K in TRACKS['B']['ks'] if len(v('B', m, 'acc', K, 'cos'))]
    tab('domain_B', pd.DataFrame(rows))

    md += ['## 10. 커버리지', '', '### 실행 설정 (결과 JSON 별)', '',
           '`protonet_excluded` = train 항목이 K+Q=10장 미만이라 ProtoNet 에피소드 학습에 한 번도 뽑히지 않은 seen 클래스'
           '(ce·arc 는 모든 seen 클래스로 학습한다).', '']
    tab('runs', pd.DataFrame(meta))
    rows = []
    for t, cfg in TRACKS.items():
        for m in cfg['methods']:
            miss = [r for r in range(N_SPLITS) if not (out / t / m / f'split{r}.json').exists()]
            if miss:
                rows.append({'track': t, 'method': m, 'missing splits': ', '.join(map(str, miss)),
                             'expected JSON': ', '.join(f'{t}/{m}/split{r}.json' for r in miss)})
    n_exp = sum(len(c['methods']) for c in TRACKS.values()) * N_SPLITS
    md += [f'### 없는 결과 JSON (있음 {len(meta)} / 기대 {n_exp})', '']
    tab('missing', pd.DataFrame(rows))
    errs = []
    ep = out / 'errors.jsonl'
    if ep.exists():
        for line in ep.read_text(encoding='utf-8').splitlines():
            e = json.loads(line)
            errs.append({'time': e['time'], 'track': e['track'], 'method': e['method'], 'split': e['split'],
                         'step': e['step'], 'error': e['error'].splitlines()[0][:200],
                         '지금 출력 있음': Path(e['target']).exists()})
    md += [f'### 오류 (`errors.jsonl`, {len(errs)}건)', '']
    tab('errors', pd.DataFrame(errs))

    md += ['## 부록 A. 에피소드 수준 95% CI (분할별, cos)', '',
           '셀 = 그 분할 E개 에피소드 평균 ± 95% CI 반폭(t, df = E−1). 본문 셀의 CI 는 분할 간 변동이다(§2.4).', '']
    rows = []
    for t, cfg in TRACKS.items():
        for m in present[t]:
            for K in cfg['ks']:
                a, ha = v(t, m, 'acc', K, 'cos'), v(t, m, 'acc_ep_hw', K, 'cos')
                f, hf = v(t, m, 'f1', K, 'cos'), v(t, m, 'f1_ep_hw', K, 'cos')
                rows += [{'track': t, 'method': m, 'K': K, 'split': r, 'acc': f'{100 * a[r]:.1f} ± {100 * ha[r]:.1f}',
                          'macro-F1': f'{100 * f[r]:.1f} ± {100 * hf[r]:.1f}'} for r in a.index]
    tab('episode_ci', pd.DataFrame(rows))

    (out / 'REPORT.md').write_text('\n'.join(md), encoding='utf-8')
    print(f"report -> {out / 'REPORT.md'} ({len(meta)}/{n_exp} JSON, errors {len(errs)})")


# ---------------------------------------------------------------------------
# selftest: _cover 가 두 manifest 모두에서 §2.1 불변식을 세우는지
# ---------------------------------------------------------------------------
def selftest():
    from src.fewshot import _synth_manifest

    crop = _synth_manifest()
    sp = make_splits(crop, 1, None, 12, 0)[0]
    u, s = sp['unseen_classes'][0], sp['seen_classes'][0]
    g_conf = next(g for g, r in sorted(sp['group_roles'].items()) if r == 'train')
    extra = pd.DataFrame([('/f/new_seen.jpg', s, 'new_seen'), ('/f/new_unseen.jpg', u, 'new_unseen'),
                          ('/f/conflict.jpg', u, g_conf)], columns=['path', 'label', 'group'])
    full = pd.concat([crop.iloc[::3], extra], ignore_index=True)
    try:
        check_split(full, sp)
    except AssertionError:
        pass
    else:
        raise AssertionError('crop 전용 분할이 full 에서 통과하면 안 된다(역할 없는 group)')
    before = dict(sp['group_roles'])
    moved, added = _cover(sp, pd.concat([crop, full])[['label', 'group']])
    assert (moved, added) == (1, 1), (moved, added)
    check_split(crop, sp)
    check_split(full, sp)
    assert g_conf in sp['unseen_groups'] and sp['group_roles']['new_seen'] == 'train'
    assert all(sp['group_roles'][g] == r for g, r in before.items() if g != g_conf)
    assert (item_roles(crop, sp)[crop['group'] == g_conf] == 'excluded').all()
    assert _parse_ids('0-2,4') == [0, 1, 2, 4] and _parse_ids('3') == [3]
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / 'summary.json'
        assert _stale(p, {'epochs': 10}) == {}
        p.write_text(json.dumps({'cfg': {'epochs': 1, 'epoch_size': 64, 'seed': 0}}))
        assert _stale(p, {'epochs': 10, 'epoch_size': 64, 'seed': 0}) == {'epochs': (1, 10)}
        p.write_text(json.dumps({'episodes': 200}))
        assert _stale(p, {'episodes': 200}) == {}
    print('[selftest] OK')


def main():
    p = argparse.ArgumentParser('fewshot_suite', description='few-shot 파일럿 실행·집계 (spec §6.5, §7)')
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(sp):
        sp.add_argument('--root', type=Path, default=DEFAULT_ROOT, help='fewshot_local (manifest 루트)')
        sp.add_argument('--out', '--results-root', dest='out', type=Path, default=DEFAULT_OUT, help='결과 루트')

    common(pl := sub.add_parser('plan'))
    pl.add_argument('--track', required=True, choices=TRACKS)
    common(rn := sub.add_parser('run'))
    rn.add_argument('--track', required=True, choices=TRACKS)
    rn.add_argument('--methods', default='all', help='쉼표 목록 또는 all')
    rn.add_argument('--splits', default=f'0-{N_SPLITS - 1}')
    rn.add_argument('--epochs', type=int, default=EPOCHS)
    rn.add_argument('--epoch-size', type=int, default=None, help='기본: 트랙 값(B 3000, S 600)')
    rn.add_argument('--episodes', type=int, default=EPISODES)
    common(sub.add_parser('report'))
    sub.add_parser('selftest')
    a = p.parse_args()

    if a.cmd == 'selftest':
        return selftest()
    out, root = a.out.resolve(), a.root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == 'plan':
        plan(a.track, root, out)
    elif a.cmd == 'run':
        methods = list(TRACKS[a.track]['methods']) if a.methods == 'all' else a.methods.split(',')
        run(a.track, methods, _parse_ids(a.splits), root, out, a.epochs, a.epoch_size, a.episodes)
    else:
        report(out, root)


if __name__ == '__main__':
    main()
