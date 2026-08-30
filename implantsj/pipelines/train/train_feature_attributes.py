#!/usr/bin/env python3
"""ViT backbone + attribute head 로 13 feature 값을 학습한다 (VLM 미사용)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.class_features import DEFAULT_TABLE
from src.feature_attributes import AttributeArgs, train_feature_attributes


def main():
    p = argparse.ArgumentParser(description='클래스->13feature 표를 정답으로 쓰는 attribute 학습')
    p.add_argument('--manifest', required=True)
    p.add_argument('--feature-table', default=str(DEFAULT_TABLE))
    p.add_argument('--label-aliases', default=None)
    p.add_argument('--attributes', default=None,
                   help='콤마 구분. 비우면 Name 제외 12개 (Name 은 클래스명 자체라 unseen 에서 정의상 불가)')
    p.add_argument('--output-root', default='/workspace/outputs')
    p.add_argument('--experiment-name', default='vit_feature_attr')
    p.add_argument('--model-name', default='vit_base_patch16_224')
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--drop-path-rate', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--no-strong-aug', action='store_true')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--allow-cpu', action='store_true')
    p.add_argument('--init-checkpoint', default=None)
    a = p.parse_args()
    train_feature_attributes(AttributeArgs(
        manifest=a.manifest, feature_table=a.feature_table, label_aliases=a.label_aliases,
        attributes=a.attributes, output_root=a.output_root, experiment_name=a.experiment_name,
        model_name=a.model_name, image_size=a.image_size, epochs=a.epochs,
        batch_size=a.batch_size, num_workers=a.num_workers, lr=a.lr,
        weight_decay=a.weight_decay, label_smoothing=a.label_smoothing, dropout=a.dropout,
        drop_path_rate=a.drop_path_rate, seed=a.seed, patience=a.patience,
        strong_aug=not a.no_strong_aug, amp=not a.no_amp, allow_cpu=a.allow_cpu,
        init_checkpoint=a.init_checkpoint,
    ))


if __name__ == '__main__':
    main()
