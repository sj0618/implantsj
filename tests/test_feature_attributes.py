"""13 feature attribute 실험 self-check.

합성 이미지로 seen 학습 -> unseen 클래스 표 채우기까지 실제로 돌린다.
timm 이 없는 환경에서는 라벨 매핑 부분만 검사한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.class_features import build_targets, load_table, match_labels, normalize_key


def test_normalize_key_roman_and_prefix():
    assert normalize_key('03_Osstem_SSII') == normalize_key('Osstem_SSII')
    assert normalize_key('SS II') == normalize_key('SSII') == 'ss2'
    assert normalize_key('UF II') == normalize_key('UF2') == 'uf2'
    assert normalize_key('IS I') == normalize_key('ISI') == 'is1'
    assert normalize_key('3i') == '3i'      # 숫자 접두는 로마숫자로 보지 않는다
    assert normalize_key('IT') == 'it'      # 접미가 로마숫자가 아니면 그대로


def test_table_matches_closed_labels():
    df = load_table()
    assert len(df) == 45
    labels = [l.strip() for l in (ROOT / 'configs' / 'closed_labels.txt').read_text().splitlines() if l.strip()]
    mapping, unmatched = match_labels(labels, df)
    # 24/27. 나머지 3개(Luna/Hexplant/IT)는 표에 없거나 이름이 달라 alias 가 필요하다.
    assert len(mapping) == 24, (len(mapping), unmatched)
    assert set(unmatched) == {'Luna', 'Hexplant', 'IT'}


def test_targets_mask_unknown_and_drop_constant():
    df = load_table()
    labels = ['Warantec_Explant', 'Osstem_GSII', 'Osstem_SSII', 'Zimmer_TSV']
    mapping, unmatched = match_labels(labels, df)
    assert not unmatched
    table, target = build_targets(labels, mapping, df)
    assert 'Body_Type' in table.dropped_constant   # 45행 전부 Threaded
    assert 'Name' not in table.attributes          # 클래스명 자체라 기본 제외
    # Warantec Explant 은 표에서 '확인 불가' 가 많다 -> -1 로 마스킹돼야 한다
    assert (target[labels.index('Warantec_Explant')] == -1).any()
    # 값이 있는 칸은 반드시 그 attribute 어휘 안의 인덱스
    for ci in range(target.shape[0]):
        for ai, a in enumerate(table.attributes):
            assert -1 <= target[ci, ai] < len(table.vocab[a])


def test_end_to_end_train_and_unseen_eval(tmp_path):
    """합성 이미지로 학습 1 epoch -> 학습에 없던 클래스 표 채우기."""
    import pytest
    pytest.importorskip('timm')
    pytest.importorskip('torch')
    import numpy as np
    import pandas as pd
    from PIL import Image

    seen = ['Osstem_GSII', 'Osstem_SSII', 'Dentium_SuperLine', 'Megagen_AnyRidge']
    unseen = ['Straumann_BoneLevel', 'Zimmer_TSV']
    rows = []
    rng = np.random.default_rng(0)
    for cls in seen + unseen:
        for split in ('train', 'valid', 'test'):
            if cls in unseen and split != 'test':
                continue
            for k in range(2):
                d = tmp_path / 'imgs' / split / cls
                d.mkdir(parents=True, exist_ok=True)
                f = d / f'{k}.png'
                Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype='uint8')).save(f)
                rows.append({'path': str(f), 'label': cls, 'split': split})
    man_seen = tmp_path / 'seen.csv'
    pd.DataFrame([r for r in rows if r['label'] in seen]).to_csv(man_seen, index=False)
    man_unseen = tmp_path / 'unseen.csv'
    pd.DataFrame([r for r in rows if r['label'] in unseen]).to_csv(man_unseen, index=False)

    from src.feature_attributes import AttributeArgs, train_feature_attributes
    run_dir = train_feature_attributes(AttributeArgs(
        manifest=str(man_seen), output_root=str(tmp_path / 'out'),
        experiment_name='selfcheck', model_name='resnet18', pretrained=False,
        image_size=32, epochs=1, batch_size=4, num_workers=0, patience=1,
        amp=False, allow_cpu=True, strong_aug=False,
    ))
    ckpt = Path(run_dir) / 'checkpoints' / 'best.pt'
    assert ckpt.exists()
    assert (Path(run_dir) / 'label_setup.json').exists()

    import subprocess
    out = tmp_path / 'unseen_eval'
    r = subprocess.run(
        [sys.executable, '-m', 'pipelines.eval.evaluate_feature_attributes',
         '--checkpoint', str(ckpt), '--manifest', str(man_unseen), '--split', 'test',
         '--output-dir', str(out), '--num-workers', '0', '--image-size', '32', '--allow-cpu'],
        cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    filled = out / 'metrics' / 'unseen_test_filled_table.csv'
    assert filled.exists(), r.stdout + r.stderr
    t = pd.read_csv(filled)
    assert set(t['class']) == set(unseen)
    # 학습에 없던 클래스임이 리포트에 남아야 한다
    import json
    setup = json.loads((out / 'unseen_setup.json').read_text())
    assert set(setup['novel_classes_not_in_training']) == set(unseen)


if __name__ == '__main__':
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, '-m', 'pytest', __file__, '-q']))
