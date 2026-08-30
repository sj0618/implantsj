#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.supervised import SupervisedArgs, train_supervised


def str2bool(v):
    if isinstance(v, bool):
        return v
    if str(v).lower() in {'1', 'true', 'yes', 'y'}:
        return True
    if str(v).lower() in {'0', 'false', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError('Boolean expected')


def main():
    p = argparse.ArgumentParser(description='NO-VENV supervised image classifier training.')
    p.add_argument('--manifest', required=True)
    p.add_argument('--output-root', default='/workspace/implant_outputs')
    p.add_argument('--experiment-name', default='vit_supervised')
    p.add_argument('--model-name', default='vit_base_patch16_224')
    p.add_argument('--pretrained', type=str2bool, default=True)
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
    p.add_argument('--amp', type=str2bool, default=True)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--monitor', default='macro_f1')
    p.add_argument('--use-class-weights', action='store_true')
    p.add_argument('--no-strong-aug', action='store_true')
    p.add_argument('--allow-cpu', action='store_true')
    p.add_argument('--resume-checkpoint', default=None)
    p.add_argument('--init-checkpoint', default=None)
    a = p.parse_args()
    args = SupervisedArgs(
        manifest=a.manifest,
        output_root=a.output_root,
        experiment_name=a.experiment_name,
        model_name=a.model_name,
        pretrained=a.pretrained,
        image_size=a.image_size,
        epochs=a.epochs,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        lr=a.lr,
        weight_decay=a.weight_decay,
        label_smoothing=a.label_smoothing,
        dropout=a.dropout,
        drop_path_rate=a.drop_path_rate,
        seed=a.seed,
        amp=a.amp,
        patience=a.patience,
        monitor=a.monitor,
        use_class_weights=a.use_class_weights,
        strong_aug=not a.no_strong_aug,
        allow_cpu=a.allow_cpu,
        resume_checkpoint=a.resume_checkpoint,
        init_checkpoint=a.init_checkpoint,
    )
    train_supervised(args)


if __name__ == '__main__':
    main()
