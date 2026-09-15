#!/usr/bin/env python3
"""few-shot 파일럿용 로컬 manifest 생성 (docs/paper/FEWSHOT_ATTRIBUTE_SPEC.md §1.2, §1.4, §6.2).

트랙 B: 로컬 YOLO 데이터셋 5개의 bbox 를 256 정사각 크롭으로 잘라 제조사(brand) 라벨로 모은다.
  group(환자 대용) = 출처군:환자키(patient_key: y12 기록번호, dddd 환자 이름, 그 외 stem)를 시작으로,
  원본 전체 이미지의 근접중복(dihedral dHash)을
  데이터셋 경계를 넘어 union-find 로 합친다. 같은 라벨 안의 근접중복 크롭/원본은 하나만 남긴다.
트랙 S: AI Hub 팬텀 사진 27 시스템을 자동 크롭해 configs/class_features.csv 의 class_key 로 모은다.

해시는 명세의 64-bit 대신 256-bit dHash(회색조 17x16)를 쓴다. 파노라마끼리 저해상도에서 닮아
64-bit 로는 무관한 사진이 과병합된다. 임계값은 dedup_report.json 에 남긴다.

  python3 tools/fewshot_build_local.py --data-root .../implant_project/data \
      --aihub-root ".../implantys/data/AI Hub Data" --out .../implant_project/data/fewshot_local
  python3 tools/fewshot_build_local.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipelines.yolo.crop_labeled import IMAGE_SUFFIXES, _class_names, _txt_boxes  # noqa: E402
from pipelines.yolo.crop_yolo import expand_square_crop  # noqa: E402

SIZE = 256
MIN_BOX = 16
CROP_DUP = 8          # 같은 라벨 안 크롭/원본 근접중복 (256-bit 중 비트 수)
SWEEP = (16, 12, 10, 8, 6, 4)

# (폴더, 출처군, 도메인). 같은 출처군 안에서 stem 이 같으면 같은 group 이다.
TRACK_B = [
    ('dddd', 'dddd', 'panoramic'),
    ('FULL_Bego_Bicon_ITI_3label', 'bbi', 'panoramic'),
    ('Dental Implants 2.0.v3i.yolov8', 'bbi', 'panoramic'),
    ('implant_yolo26_12label', 'y12', 'periapical'),
    ('realyolo26', 'ry26', 'catalog'),
]
# 기본 규칙(소문자, 비영숫자 -> '_')과 결과가 다른 것만 적는다. BEGO/Bego, STRAUMANN 등은 기본 규칙으로 같아진다.
# 라벨 단위는 제조사다(§1.4). 같은 제조사의 다른 이름을 따로 두면 한쪽이 unseen 일 때 다른 쪽이 seen 학습에 들어간다.
#   Hiossen = Osstem 미국 브랜드(realyolo26 Hiossen 파일명 65개 중 다수가 Osstem 제품코드: osstemssiii…, msn2510s25 는 osstem 에도 있다)
#   ITI / Straumann ITI = Straumann (BLX·TLX·tissue level)
#   Implance = AGS Medikal 제품(realyolo26 AGS Medikal 파일명 `…-implantswiss-implance-bone-level-…`)
ALIAS = {
    'ITI': 'straumann', 'Straumann ITI': 'straumann', 'Hiossen': 'osstem', 'IMPLANCE': 'ags_medikal',
    '3I': 'biomet_3i', 'Biomet 3i': 'biomet_3i',
    'BIOHORIZON': 'biohorizons', 'Nobel Biocare': 'nobel', 'Osstem Implant': 'osstem',
}
AIHUB = {
    1: 'osstem_gsii', 2: 'osstem_gsiii', 3: 'osstem_ssii', 4: 'osstem_tsiii', 5: 'osstem_usii',
    6: 'osstem_ssiii', 7: 'osstem_usiii', 8: 'dentium_implantium', 9: 'dentium_superline',
    10: 'dentium_simpleline', 11: 'neobiotech_isi', 12: 'neobiotech_isii', 13: 'neobiotech_isiii',
    14: 'neobiotech_eb', 15: 'shinhung_luna1', 20: 'megagen_anyridge', 21: 'megagen_exfeelinternal',
    22: 'megagen_exfeelexternal', 23: 'megagen_anyoneexternal', 24: 'megagen_anyoneinternal',
    26: 'cowellmedi_atlasinternal', 27: 'cowellmedi_atlasexternal', 31: 'straumann_tsstandard',
    33: 'straumann_bonelevel', 37: 'dentsply_xive', 38: 'zimmer_swissplus', 42: 'biohorizons_external',
}
RF_SUFFIX = re.compile(r'_(jpg|jpeg|png|bmp)\.rf\.[0-9a-f]+\.(jpg|jpeg|png)$', re.I)
DIHEDRAL = [None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270,
            Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM,
            Image.Transpose.TRANSPOSE, Image.Transpose.TRANSVERSE]


def canon_label(name: str) -> str:
    return ALIAS.get(name) or re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def image_stem(filename: str) -> str:
    """Roboflow `<stem>_<ext>.rf.<hash>.jpg` -> 정규화 stem (증강본·확장자 꼬리·copy 사본을 한 stem 으로)."""
    s = RF_SUFFIX.sub('', filename)
    if s == filename:
        s = Path(filename).stem
    prev = None
    while s != prev:
        prev = s
        s = re.sub(r'_(bmp|png|jpg|jpeg)$', '', s, flags=re.I)
        s = re.sub(r'(-copy(-\d+)?| copy)$', '', s, flags=re.I).rstrip('-_ ')
    return s.lower()


def patient_key(family: str, stem: str) -> tuple[str, str]:
    """초기 group = (key, 수준). 파일명에 환자 정보가 있는 출처군은 그것을 먼저 쓴다(§1.1 우선순위 ①).

    y12 : 파일명의 8자리 이상 숫자 = 환자/기록 번호(`bego-1001486781-pa-2`, `implance-ve-straumann-13535636170-2`)
    dddd: 환자 이름 + 촬영 번호(`deepika-2`, `ashok11`, `latha_46_m_20240608_143153_1`) -> 꼬리 숫자 제거.
          남는 글자가 3자 미만이면 멈춘다(`r10`, `is20220412_…` 같은 번호형 이름은 그대로).
          `dentium-superline-12` 같은 제품명+번호도 합쳐지지만 과병합은 보수적 방향이다(§9).
    그 외: 정규화 stem.
    """
    if family == 'y12' and (m := re.search(r'\d{8,}', stem)):
        return f'y12:{m.group()}', 'record_id'
    if family == 'dddd':
        base = stem
        while (t := re.sub(r'[-_]?\d+$', '', base)) != base and len(re.findall(r'[a-z]', t)) >= 3:
            base = t
        if base != stem:
            return f'dddd:{base}', 'patient_name'
    return f'{family}:{stem}', 'stem'


def specimen_token(stem: str, folder: str) -> str:
    """AI Hub 파일명 -> specimen. `(n)` 촬영번호, 폴더 접두, 공백을 떼고 크기 표기(5 == 5.0)를 맞춘다."""
    def norm(x: str) -> str:
        x = re.sub(r'\s+', '', re.sub(r'\s*\(\d+\)\s*$', '', x).lower())
        return re.sub(r'_ao[ie]$', '', x)       # 폴더/파일 끝 _AOI, _AOE 는 같은 제품 표기
    s, f = norm(stem), norm(folder)
    if not s.startswith(f):
        raise ValueError(f'folder prefix mismatch: {stem!r} in {folder!r}')
    s = re.sub(r'\d+(?:\.\d+)?', lambda m: f'{float(m.group()):g}', s[len(f):]).strip('_-')
    return s or 'base'


def dhash8(img: Image.Image) -> np.ndarray:
    """256-bit dHash 를 dihedral 8변환마다 계산 -> (8, 4) uint64."""
    g = img.convert('L')
    g.thumbnail((128, 128))
    out = []
    for t in DIHEDRAL:
        v = g if t is None else g.transpose(t)
        a = np.asarray(v.resize((17, 16), Image.Resampling.BILINEAR), dtype=np.int16)
        out.append(np.packbits(a[:, 1:] > a[:, :-1]).view(np.uint64))
    return np.stack(out)


def near_pairs(H: np.ndarray, thr: int, chunk: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """dist(i, j) = min_t popcount(H[i, 0] ^ H[j, t]). i != j, dist <= thr 인 (i, j, dist)."""
    ii, jj, dd = [], [], []
    for s in range(0, len(H), chunk):
        x = np.bitwise_count(H[s:s + chunk, None, None, 0, :] ^ H[None]).sum(-1, dtype=np.uint16).min(-1)
        i, j = np.nonzero(x <= thr)
        keep = (i + s) != j
        ii.append(i[keep] + s)
        jj.append(j[keep])
        dd.append(x[i[keep], j[keep]])
    return np.concatenate(ii), np.concatenate(jj), np.concatenate(dd)


def dedup_keep(H: np.ndarray, priority: list, thr: int = CROP_DUP) -> tuple[np.ndarray, np.ndarray]:
    """priority 오름차순으로 훑으며 이미 남긴 것과 dist <= thr 이면 버린다 -> (keep mask, removed_by)."""
    n = len(H)
    nbrs = defaultdict(list)
    for i, j in zip(*near_pairs(H, thr)[:2]):
        nbrs[i].append(j)
        nbrs[j].append(i)
    keep = np.zeros(n, bool)
    removed_by = np.full(n, -1)
    for i in sorted(range(n), key=lambda k: priority[k]):
        if removed_by[i] >= 0:
            continue
        keep[i] = True
        for j in nbrs[i]:
            if not keep[j] and removed_by[j] < 0:
                removed_by[j] = i
    return keep, removed_by


def components(n: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    g = coo_matrix((np.ones(len(i), np.int8), (i, j)), shape=(n, n))
    return connected_components(g, directed=False)[1]


def _uid(*parts: str) -> str:
    return hashlib.sha1('/'.join(parts).encode()).hexdigest()[:12]


# ---------------------------------------------------------------- track B

def _scan_b(task: tuple) -> dict:
    img_path, lbl_path, dataset, family, domain, split, names, out = task
    rec = {'img': img_path, 'dataset': dataset, 'family': family, 'domain': domain, 'split': split,
           'status': 'ok', 'n_boxes': 0, 'n_small': 0, 'crops': [], 'full': None, 'labels': []}
    if not Path(lbl_path).is_file():
        rec['status'] = 'no_label_file'
        return rec
    try:
        with Image.open(img_path) as im:
            rgb = im.convert('RGB')
        boxes = list(_txt_boxes(Path(lbl_path), rgb.size, names))
    except (ValueError, OSError) as exc:
        rec['status'] = f'error: {exc}'
        return rec
    rec['n_boxes'] = len(boxes)
    w, h = rgb.size
    uid = f'{family}_{_uid(dataset, split, Path(img_path).name)}'
    for box_index, (src, (x1, y1, x2, y2)) in enumerate(boxes):
        x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)
        if x2 - x1 < MIN_BOX or y2 - y1 < MIN_BOX:
            rec['n_small'] += 1
            continue
        label = canon_label(src)
        crop = expand_square_crop(rgb, (x1, y1, x2, y2), 0.1).resize((SIZE, SIZE), Image.Resampling.BICUBIC)
        path = Path(out) / 'crops' / label / f'{uid}_{box_index:02d}.jpg'
        path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(path, quality=95)
        rec['crops'].append((box_index, label, str(path), dhash8(crop)))
    if not boxes:
        rec['status'] = 'no_boxes'
    elif not rec['crops']:
        rec['status'] = 'all_small'
    else:
        rec['hash'] = dhash8(rgb)
        rec['labels'] = sorted({c[1] for c in rec['crops']})
        if len(rec['labels']) == 1:
            path = Path(out) / 'full' / rec['labels'][0] / f'{uid}.jpg'
            path.parent.mkdir(parents=True, exist_ok=True)
            rgb.resize((SIZE, SIZE), Image.Resampling.BICUBIC).save(path, quality=95)
            rec['full'] = str(path)
    return rec


def _sweep_stats(labels: np.ndarray, fam: np.ndarray, init: np.ndarray, thr: int, top: int = 10) -> dict:
    df = pd.DataFrame({'c': labels, 'fam': fam, 'init': init})
    g = df.groupby('c').agg(n=('init', 'size'), n_init=('init', 'nunique'), n_fam=('fam', 'nunique'))
    sizes = g['n']
    return {
        'threshold': thr,
        'n_groups': int(len(g)),
        'n_merged_components': int((g['n_init'] > 1).sum()),
        'n_cross_family_components': int((g['n_fam'] > 1).sum()),
        'size_hist': {k: int(v) for k, v in {'1': (sizes == 1).sum(), '2-5': sizes.between(2, 5).sum(),
                      '6-20': sizes.between(6, 20).sum(), '21-100': sizes.between(21, 100).sum(),
                      '>100': (sizes > 100).sum()}.items()},
        'top': [{'n_images': int(r.n), 'n_initial_groups': int(r.n_init), 'n_families': int(r.n_fam)}
                for r in g.sort_values('n', ascending=False).head(top).itertuples()],
    }


def build_track_b(data_root: Path, out: Path, merge_thr: int, crop_thr: int, workers: int) -> None:
    tasks, alias_rows = [], []
    for folder, family, domain in TRACK_B:
        root = data_root / folder
        names = _class_names(root / 'data.yaml')
        alias_rows += [(folder, n, canon_label(n)) for n in names.values()]
        for split in ('train', 'valid', 'test'):
            img_dir = root / split / 'images'
            if not img_dir.is_dir():
                continue
            for p in sorted(img_dir.iterdir()):
                if p.suffix.lower() in IMAGE_SUFFIXES:
                    tasks.append((str(p), str(root / split / 'labels' / f'{p.stem}.txt'),
                                  folder, family, domain, split, names, str(out)))
    print(f'[B] scanning {len(tasks)} images with {workers} workers', flush=True)
    with Pool(workers) as pool:
        recs = list(pool.imap(_scan_b, tasks, chunksize=8))

    per_ds = defaultdict(Counter)
    for r in recs:
        c = per_ds[r['dataset']]
        c['images'] += 1
        c[f"status_{r['status'].split(':')[0]}"] += 1
        c['boxes'] += r['n_boxes']
        c['boxes_dropped_small'] += r['n_small']
        c['crops'] += len(r['crops'])
        c['multi_label_images'] += len(r['labels']) > 1
    errors = [(r['img'], r['status']) for r in recs if r['status'].startswith('error')]
    ok = [r for r in recs if r['crops']]

    # --- group: family:stem 에서 시작해 원본 전체 근접중복으로 union
    n = len(ok)
    H = np.stack([r['hash'] for r in ok])
    fam = np.array([r['family'] for r in ok])
    keys = [patient_key(r['family'], image_stem(Path(r['img']).name)) for r in ok]
    init = np.array([k for k, _ in keys])
    level = pd.DataFrame(keys, columns=['key', 'level']).assign(family=fam)
    codes, first = np.unique(init, return_inverse=True)[1], {}
    same = np.array([first.setdefault(c, k) for k, c in enumerate(codes)])
    pi, pj, pd_ = near_pairs(H, max(SWEEP))
    sweep = []
    for thr in SWEEP:
        m = pd_ <= thr
        lab = components(n, np.r_[np.arange(n), pi[m]], np.r_[same, pj[m]])
        sweep.append(_sweep_stats(lab, fam, init, thr))
        s = sweep[-1]
        print(f"[B] merge thr {thr:2d}: groups {s['n_groups']}, merged {s['n_merged_components']}, "
              f"cross-family {s['n_cross_family_components']}, top sizes {[t['n_images'] for t in s['top']]}", flush=True)
    m = pd_ <= merge_thr
    lab = components(n, np.r_[np.arange(n), pi[m]], np.r_[same, pj[m]])
    name_of = pd.Series(init).groupby(lab).min()     # component 이름 = 구성원 초기 group 중 사전순 최소
    group = name_of.loc[lab].to_numpy()
    top10 = []
    for c, cnt in Counter(lab).most_common(10):
        idx = np.nonzero(lab == c)[0]
        top10.append({'group': name_of.loc[c], 'n_images': int(cnt), 'families': dict(Counter(fam[idx])),
                      'initial_groups': sorted(set(init[idx]))[:20],
                      'sample_images': [ok[k]['img'] for k in idx[:12]]})
    und = {(min(a, b), max(a, b)) for a, b in zip(pi[m], pj[m])}   # 거리가 비대칭일 수 있어 무방향 쌍으로 센다
    cross_edges = sum(fam[a] != fam[b] for a, b in und)

    # --- manifest + 같은 라벨 안 근접중복 제거 (valid/test 사본 우선: Roboflow 는 train 만 증강한다)
    crops = pd.DataFrame([
        {'path': path, 'label': label, 'group': group[k], 'dataset': r['dataset'], 'domain': r['domain'],
         'source_image': r['img'], 'box_index': bi, '_split': r['split'], '_h': hsh}
        for k, r in enumerate(ok) for bi, label, path, hsh in r['crops']])
    full = pd.DataFrame([
        {'path': r['full'], 'label': r['labels'][0], 'group': group[k], 'dataset': r['dataset'],
         'domain': r['domain'], 'source_image': r['img'], '_split': r['split'], '_h': r['hash']}
        for k, r in enumerate(ok) if r['full']])

    def dedup(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        keep_all = np.zeros(len(df), bool)
        by_label, cross = {}, Counter()
        for label, idx in df.groupby('label').indices.items():
            sub = df.iloc[idx]
            keep, rb = dedup_keep(np.stack(sub['_h'].to_numpy()),
                                  list(zip(sub['_split'] == 'train', sub['path'])), crop_thr)
            keep_all[idx] = keep
            by_label[label] = int((~keep).sum())
            g, ds = sub['group'].to_numpy(), sub['dataset'].to_numpy()
            cross.update(ds[j] for j in np.nonzero(~keep)[0] if g[j] != g[rb[j]])
        for p in df.loc[~keep_all, 'path']:
            Path(p).unlink()
        return df[keep_all].drop(columns=['_split', '_h']).reset_index(drop=True), {
            'threshold': crop_thr, 'n_before': int(len(df)), 'n_removed': int((~keep_all).sum()),
            'n_removed_cross_group': int(sum(cross.values())), 'removed_cross_group_by_dataset': dict(cross),
            'removed_by_label': {k: v for k, v in sorted(by_label.items()) if v}}

    crops, crop_rep = dedup(crops)
    full, full_rep = dedup(full)
    crops = crops.sort_values(['label', 'path']).reset_index(drop=True)
    full = full.sort_values(['label', 'path']).reset_index(drop=True)
    crops.to_csv(out / 'manifest_crop.csv', index=False)
    full.to_csv(out / 'manifest_full.csv', index=False)
    pd.DataFrame(alias_rows, columns=['dataset', 'source_label', 'label']).to_csv(out / 'alias_map.csv', index=False)

    cc = crops.groupby('label').agg(n_items=('path', 'size'), n_groups=('group', 'nunique'),
                                    datasets=('dataset', lambda s: ';'.join(sorted(set(s)))))
    cc.insert(2, 'n_full', full.groupby('label').size().reindex(cc.index).fillna(0).astype(int))
    for i, d in enumerate(('panoramic', 'periapical', 'catalog')):
        cc.insert(3 + i, f'n_{d}', crops[crops['domain'] == d].groupby('label').size().reindex(cc.index).fillna(0).astype(int))
    cc = cc.sort_values(['n_groups', 'n_items'], ascending=False)
    cc.to_csv(out / 'class_counts.csv')

    report = {
        'hash': 'dHash 256-bit (grayscale 17x16, horizontal gradient), distance = min Hamming over 8 dihedral variants',
        'min_box_side_px': MIN_BOX,
        'per_dataset': {k: dict(v) for k, v in per_ds.items()},
        'errors': errors[:50],
        'group_merge': {
            'initial_group': 'family:patient_key (y12 record number, dddd patient name, else normalized stem)',
            'initial_group_level': {f'{f}:{lv}': {'n_images': int(len(g)), 'n_keys': int(g['key'].nunique())}
                                    for (f, lv), g in level.groupby(['family', 'level'])},
            'threshold': merge_thr,
            'n_images': n, 'n_initial_groups': int(len(set(init))), 'n_final_groups': int(len(set(group))),
            'n_near_dup_pairs': len(und), 'n_cross_family_pairs': int(cross_edges),
            'sweep': sweep, 'top10': top10},
        'crop_dedup': crop_rep,
        'full_dedup': full_rep,
        'totals': {'n_crops': int(len(crops)), 'n_full': int(len(full)), 'n_labels': int(crops['label'].nunique()),
                   'n_groups': int(crops['group'].nunique()), 'n_labels_groups_ge_12': int((cc['n_groups'] >= 12).sum())},
    }
    (out / 'dedup_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[B] crops {len(crops)} / labels {cc.shape[0]} / groups {crops['group'].nunique()} / full {len(full)}; "
          f"crop dedup -{crop_rep['n_removed']}, full dedup -{full_rep['n_removed']}", flush=True)


# ---------------------------------------------------------------- track S

def otsu(g: np.ndarray) -> float:
    p = np.bincount(np.clip(g, 0, 255).astype(np.uint8).ravel(), minlength=256) / g.size
    w, mu = np.cumsum(p), np.cumsum(p * np.arange(256))
    return float(np.argmax((mu[-1] * w - mu) ** 2 / (w * (1 - w) + 1e-12)))


def autocrop_box(img: Image.Image) -> tuple[tuple[float, float, float, float], str]:
    """팬텀 사진의 임플란트 = 테두리에 닿지 않는 가장 큰 밝은 덩어리. 홀더/센서 막대는 테두리에 닿는다."""
    g = ndimage.gaussian_filter(np.asarray(img.convert('L'), np.float32), max(img.size) / 400)
    mask = g > otsu(g)
    lab, k = ndimage.label(mask)
    h, w = mask.shape
    if k:
        areas = ndimage.sum_labels(mask, lab, np.arange(1, k + 1))
        sls = ndimage.find_objects(lab)
        inner = [i for i, (sy, sx) in enumerate(sls)
                 if sy.start > 0 and sx.start > 0 and sy.stop < h and sx.stop < w]
        for how, cand in (('inner', inner), ('fallback_largest', list(range(k)))):
            if cand:
                i = max(cand, key=lambda c: areas[c])
                sy, sx = sls[i]
                return (sx.start, sy.start, sx.stop, sy.stop), how
    s = min(w, h) / 2
    return (w / 2 - s / 2, h / 2 - s / 2, w / 2 + s / 2, h / 2 + s / 2), 'fallback_center'


def build_track_s(aihub_root: Path, out: Path, crop_thr: int) -> None:
    table = pd.read_csv(ROOT / 'configs' / 'class_features.csv')
    missing = sorted(set(AIHUB.values()) - set(table['class_key']))
    assert not missing, f'class_key missing in class_features.csv: {missing}'
    rows, how_count, fallbacks = [], Counter(), []
    folders = sorted(p for p in aihub_root.iterdir() if p.is_dir())
    assert len(folders) == len(AIHUB), [p.name for p in folders]
    for folder in folders:
        key = AIHUB[int(folder.name.split('_')[0])]
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            with Image.open(p) as im:
                rgb = im.convert('RGB')
            box, how = autocrop_box(rgb)
            how_count[how] += 1
            if how != 'inner':
                fallbacks.append((str(p), how))
            crop = expand_square_crop(rgb, box, 0.15).resize((SIZE, SIZE), Image.Resampling.BICUBIC)
            path = out / 'images' / key / f'{key}_{_uid(folder.name, p.name)}.jpg'
            path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(path, quality=95)
            rows.append({'path': str(path), 'label': key, 'group': f'{key}:{specimen_token(p.stem, folder.name)}',
                         'class_key': key, 'dataset': 'aihub_sample', 'domain': 'phantom',
                         'source_image': str(p), '_h': dhash8(crop), '_how': how})
    df = pd.DataFrame(rows)
    keep_all, by_label, cross = np.zeros(len(df), bool), {}, 0
    for label, idx in df.groupby('label').indices.items():
        keep, rb = dedup_keep(np.stack(df['_h'].iloc[idx].to_numpy()), list(df['path'].iloc[idx]), crop_thr)
        keep_all[idx] = keep
        by_label[label] = int((~keep).sum())
        g = df['group'].iloc[idx].to_numpy()
        cross += int(sum(g[j] != g[rb[j]] for j in np.nonzero(~keep)[0]))
    for p in df.loc[~keep_all, 'path']:
        Path(p).unlink()
    kept = df[keep_all].reset_index(drop=True)
    kept.drop(columns=['_h', '_how']).to_csv(out / 'manifest.csv', index=False)
    cc = kept.groupby('label').agg(n_items=('path', 'size'), n_groups=('group', 'nunique'),
                                   n_phantom=('path', 'size'), datasets=('dataset', 'first'))
    cc.sort_values(['n_groups', 'n_items'], ascending=False).to_csv(out / 'class_counts.csv')
    (out / 'dedup_report.json').write_text(json.dumps({
        'hash': 'dHash 256-bit on the 256x256 autocrop, dihedral-min Hamming',
        'n_images': int(len(df)), 'n_kept': int(len(kept)),
        'crop_dedup': {'threshold': crop_thr, 'n_removed': int((~keep_all).sum()),
                       'n_removed_cross_specimen': cross,
                       'groups_lost': sorted(set(df['group']) - set(kept['group'])),
                       'removed_by_label': {k: v for k, v in sorted(by_label.items()) if v}},
        'n_groups_before_dedup': int(df['group'].nunique()),
        'autocrop': {'method': 'gaussian blur + Otsu + largest component not touching border, pad 0.15',
                     'counts': dict(how_count), 'fallbacks': fallbacks},
        'n_groups': int(kept['group'].nunique()),
        'n_classes_single_group': int((cc['n_groups'] == 1).sum()),
    }, ensure_ascii=False, indent=2))

    # 육안 점검용 몽타주: 남은 크롭 전부. 빨간 테두리 = autocrop fallback
    t, cols = 128, 16
    font = ImageFont.load_default(12)
    sheet = Image.new('RGB', (cols * t, -(-len(kept) // cols) * t), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, r in kept.iterrows():
        x, y = (i % cols) * t, (i // cols) * t
        with Image.open(r['path']) as im:
            sheet.paste(im.resize((t, t)), (x, y))
        draw.text((x + 2, y + 1), f"{i} {r['label'][:16]}", fill=(255, 255, 0), font=font)
        if r['_how'] != 'inner':
            draw.rectangle((x, y, x + t - 1, y + t - 1), outline=(255, 0, 0), width=3)
    sheet.save(out / 'autocrop_montage.jpg', quality=90)
    print(f"[S] images {len(df)} -> kept {len(kept)} / classes {cc.shape[0]} / groups {kept['group'].nunique()}; "
          f"autocrop {dict(how_count)}", flush=True)


def selftest() -> None:
    assert image_stem('R1-Copy_jpg.rf.0a1b2c.jpg') == 'r1'
    assert image_stem('noris-6-_jpeg_jpg.rf.deadbeef.jpg') == 'noris-6'
    assert image_stem('BioVent_png_jpg.rf.12ab.jpg') == image_stem('biovent_jpg.rf.34cd.jpg') == 'biovent'
    assert canon_label('3M ESPE') == '3m_espe' and canon_label('ITI') == canon_label('STRAUMANN') == 'straumann'
    assert canon_label('BEGO') == canon_label('Bego') == 'bego' and canon_label('Hiossen') == canon_label('osstem') == 'osstem'
    pk = patient_key
    assert pk('y12', 'bego-1001486781-pa-2')[0] == pk('y12', 'bego-1001486781')[0] == 'y12:1001486781'
    assert pk('y12', 'implance-ve-straumann-13535636170-2')[0] == pk('y12', 'straumann-13535636170-pa')[0]
    assert pk('y12', 'bego-3') == ('y12:bego-3', 'stem')
    assert pk('dddd', 'deepika-1')[0] == pk('dddd', 'deepika-2')[0] == pk('dddd', 'deepika')[0] == 'dddd:deepika'
    assert pk('dddd', 'mrityunjay1')[0] == pk('dddd', 'mrityunjay')[0] and pk('dddd', 'ashok-2')[0] == 'dddd:ashok'
    assert pk('dddd', 'latha_46_m_20240608_143153_1')[0] == pk('dddd', 'latha_46_m_20240608_143153')[0]
    assert pk('dddd', 'ashok-patil-2')[0] != pk('dddd', 'ashok')[0]
    assert pk('dddd', 'r10')[0] == 'dddd:r10' and pk('dddd', '12')[0] == 'dddd:12'
    assert pk('dddd', 'is20220412_130948_0949_00000019')[0] == 'dddd:is20220412_130948_0949_00000019'
    assert pk('bbi', 'bego-12') == ('bbi:bego-12', 'stem')   # bbi 의 `<brand>-N` 은 서로 다른 사진이다
    t = specimen_token
    assert t('06_Osstem_SSIII 5 (1)', '06_Osstem_SSIII') == t('06_Osstem_SSIII 5.0 (3)', '06_Osstem_SSIII') == '5'
    assert t('06_Osstem_SSIII4.5', '06_Osstem_SSIII') == t('06_Osstem_SSIII 4.5 (2)', '06_Osstem_SSIII')
    assert t('24_MegaGen_AnyOne_Internal_AOI (1)', '24_MegaGen_AnyOne_Internal_AOI') == \
        t('24_MegaGen_AnyOne_Internal (4)', '24_MegaGen_AnyOne_Internal_AOI') == 'base'
    assert t('12_NeoBiotech_ISII 4.0x10 2.5HEX (1)', '12_NeoBiotech_IS II') == '4x102.5hex'
    rng = np.random.default_rng(0)
    a = Image.fromarray(rng.integers(0, 255, (90, 120), dtype=np.uint8))
    b = Image.fromarray(rng.integers(0, 255, (90, 120), dtype=np.uint8))
    H = np.stack([dhash8(a), dhash8(a.transpose(Image.Transpose.ROTATE_90).transpose(Image.Transpose.FLIP_LEFT_RIGHT)),
                  dhash8(b)])
    i, j, d = near_pairs(H, 256)
    dist = {(x, y): z for x, y, z in zip(i, j, d)}
    assert dist[(0, 1)] == 0 and dist[(0, 2)] > 64, dist
    keep, rb = dedup_keep(H, [(True, 'a'), (False, 'b'), (True, 'c')])   # 0 은 train, 1 은 valid
    assert keep.tolist() == [False, True, True] and rb[0] == 1
    img = np.full((300, 400), 50, np.uint8)
    img[:30, 100:300] = 255          # 테두리에 닿는 홀더 막대 (더 크다)
    img[120:260, 180:220] = 240      # 임플란트
    box, how = autocrop_box(Image.fromarray(img))
    assert how == 'inner' and abs(box[0] - 180) <= 3 and abs(box[3] - 260) <= 3, (box, how)
    lab = components(4, np.array([0, 2]), np.array([1, 2]))
    assert lab[0] == lab[1] != lab[2] != lab[3]
    print('selftest ok')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data-root', type=Path)
    ap.add_argument('--aihub-root', type=Path)
    ap.add_argument('--out', type=Path)
    # 12 에서는 무관한 단일 임플란트 근접사진·회전 파노라마가 사슬로 이어져 88장/60장 component 가 생겼다(2026-09-15 육안 확인).
    ap.add_argument('--merge-thr', type=int, default=10, help='원본 전체 근접중복 group 병합 임계 (256-bit 중)')
    ap.add_argument('--crop-dup-thr', type=int, default=CROP_DUP, help='같은 라벨 안 크롭/원본 근접중복 임계 (256-bit 중)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    t0 = time.time()
    def fresh(track: str) -> Path:
        out = args.out / track
        shutil.rmtree(out, ignore_errors=True)   # 이 도구가 만든 산출물만 다시 만든다
        out.mkdir(parents=True)
        return out
    if args.data_root:
        build_track_b(args.data_root, fresh('trackB'), args.merge_thr, args.crop_dup_thr, args.workers)
    if args.aihub_root:
        build_track_s(args.aihub_root, fresh('trackS'), args.crop_dup_thr)
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
