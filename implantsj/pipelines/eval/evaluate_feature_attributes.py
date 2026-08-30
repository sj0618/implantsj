#!/usr/bin/env python3
"""학습에 없던 클래스의 13 feature 표를 채운다.

학습 때 정해진 attribute/값 어휘(vocab)를 그대로 쓴다. unseen 클래스의 정답 값이
학습 어휘에 없으면 그 칸은 구조적으로 맞힐 수 없으므로, 조용히 빼지 않고
`out_of_vocab` 으로 세어서 리포트한다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.class_features import UNKNOWN, FeatureTable, load_aliases, load_table, match_labels
from src.data import create_loader, load_manifest
from src.feature_attributes import evaluate_attributes
from src.models import AttributeModel, load_checkpoint
from src.utils import log, write_json


def main():
    p = argparse.ArgumentParser(description='처음 보는 클래스의 13 feature 값 예측 (VLM 미사용)')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifest', required=True, help='unseen 클래스가 든 manifest')
    p.add_argument('--split', default='test')
    p.add_argument('--feature-table', default=None)
    p.add_argument('--label-aliases', default=None)
    p.add_argument('--output-dir', default=None)
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--allow-cpu', action='store_true')
    a = p.parse_args()

    ckpt = load_checkpoint(a.checkpoint, map_location='cpu')
    for key in ('attributes', 'vocab', 'args'):
        if key not in ckpt:
            raise RuntimeError(f'{a.checkpoint} 는 feature-attribute 체크포인트가 아닙니다 (no {key}).')
    attributes, vocab = ckpt['attributes'], ckpt['vocab']
    table_path = a.feature_table or ckpt['args'].get('feature_table')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu' and not a.allow_cpu:
        raise RuntimeError('CUDA 를 못 잡았습니다. --allow-cpu 로 CPU 실행 가능.')

    # unseen 클래스를 표에 붙인다.
    df_all = load_manifest(a.manifest, split=a.split)
    labels = sorted(df_all['label'].astype(str).unique().tolist())
    table_df = load_table(table_path)
    mapping, unmatched = match_labels(labels, table_df, load_aliases(a.label_aliases))
    matched = [l for l in labels if l in mapping]
    if not matched:
        raise RuntimeError(f'표에 붙는 클래스가 없습니다. 데이터셋 클래스: {labels[:5]}')

    seen = set(ckpt['class_to_idx'])
    novel = [c for c in matched if c not in seen]
    log(f'{a.split} 클래스 {len(labels)}개 -> 표 매칭 {len(matched)}, 미매칭 {len(unmatched)}: {unmatched}')
    log(f'그중 학습에 없던 클래스 {len(novel)}개: {novel}')

    rows = table_df.iloc[[mapping[l] for l in matched]].reset_index(drop=True)
    target = np.full((len(matched), len(attributes)), -1, dtype='int64')
    oov = {a_: [] for a_ in attributes}
    for ci in range(len(matched)):
        for ai, attr in enumerate(attributes):
            v = rows.at[ci, attr]
            if not v or v == UNKNOWN:
                continue
            if v in vocab[attr]:
                target[ci, ai] = vocab[attr].index(v)
            else:
                oov[attr].append({'class': matched[ci], 'value': v})

    n_oov = sum(len(v) for v in oov.values())
    if n_oov:
        log(f'학습 어휘에 없는 정답 값 {n_oov}칸 -> 구조적으로 예측 불가. 평가에서 제외하되 리포트에 남깁니다.')

    class_to_idx = {c: i for i, c in enumerate(matched)}
    loader = create_loader(a.manifest, a.split, class_to_idx, a.image_size, a.batch_size,
                           a.num_workers, train=False, strong_aug=False, return_path=True)

    model = AttributeModel(ckpt['args']['model_name'], {k: len(vocab[k]) for k in attributes},
                           pretrained=False).to(device)
    model.load_state_dict(ckpt['model_state_dict'])

    table = FeatureTable(df=rows, attributes=attributes, vocab=vocab, dropped_constant=[])
    out_dir = Path(a.output_dir or (Path(a.checkpoint).parent.parent / 'unseen_eval'))
    out_dir.mkdir(parents=True, exist_ok=True)
    m = evaluate_attributes(model, loader, device, table, torch.from_numpy(target).to(device),
                            f'unseen_{a.split}', out_dir=out_dir)
    write_json(out_dir / 'unseen_setup.json', {
        'checkpoint': a.checkpoint, 'split': a.split,
        'matched_classes': matched, 'unmatched_classes': unmatched,
        'novel_classes_not_in_training': novel,
        'out_of_vocab_cells': oov, 'n_out_of_vocab_cells': n_oov,
        'total_cells': int(target.size), 'unknown_cells': int((target < 0).sum()) - n_oov,
    })
    log(f"mean_macro_f1={m['mean_macro_f1']:.4f} mean_accuracy={m['mean_accuracy']:.4f} "
        f"exact_row_match={m['exact_row_match']:.4f}")
    log(f'채워진 표: {out_dir}/metrics/unseen_{a.split}_filled_table.csv')


if __name__ == '__main__':
    main()
