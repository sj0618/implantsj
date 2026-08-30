#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.metric_learning import MetricArgs, train_metric


def main():
    p = argparse.ArgumentParser(description='NO-VENV metric learning training: SupCon or triplet.')
    p.add_argument('--manifest', required=True)
    p.add_argument('--output-root', default='/workspace/implant_outputs')
    p.add_argument('--experiment-name', default='vit_metric_supcon')
    p.add_argument('--model-name', default='vit_base_patch16_224')
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--loss-name', choices=['supcon', 'triplet'], default='supcon')
    p.add_argument('--temperature', type=float, default=0.07)
    p.add_argument('--triplet-margin', type=float, default=0.2)
    p.add_argument('--embedding-dim', type=int, default=128)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--drop-path-rate', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--allow-cpu', action='store_true')
    a = p.parse_args()
    args = MetricArgs(
        manifest=a.manifest,
        output_root=a.output_root,
        experiment_name=a.experiment_name,
        model_name=a.model_name,
        image_size=a.image_size,
        epochs=a.epochs,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        lr=a.lr,
        weight_decay=a.weight_decay,
        loss_name=a.loss_name,
        temperature=a.temperature,
        triplet_margin=a.triplet_margin,
        embedding_dim=a.embedding_dim,
        dropout=a.dropout,
        drop_path_rate=a.drop_path_rate,
        seed=a.seed,
        patience=a.patience,
        allow_cpu=a.allow_cpu,
    )
    train_metric(args)


if __name__ == '__main__':
    main()
