#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import build_manifest, detect_layout, make_stratified_splits, print_manifest_summary
from src.supervised import SupervisedArgs, train_supervised


def main():
    p = argparse.ArgumentParser(description='Fine-tune small dataset from a pretrained supervised/metric checkpoint.')
    p.add_argument('--data-root', required=True, help='Small dataset folder, e.g. /workspace/data/current_3class')
    p.add_argument('--manifest', required=True, help='Output/input small manifest path')
    p.add_argument('--checkpoint', required=True, help='Large pretrain checkpoint')
    p.add_argument('--output-root', default='/workspace/implant_outputs')
    p.add_argument('--experiment-name', default='finetune_small')
    p.add_argument('--model-name', default='vit_base_patch16_224')
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--label-smoothing', type=float, default=0.05)
    p.add_argument('--train-ratio', type=float, default=3)
    p.add_argument('--valid-ratio', type=float, default=1)
    p.add_argument('--test-ratio', type=float, default=7)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--rebuild-manifest', action='store_true')
    p.add_argument('--allow-cpu', action='store_true')
    a = p.parse_args()
    data_root = Path(a.data_root)
    manifest = Path(a.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if a.rebuild_manifest or not manifest.exists():
        layout = detect_layout(data_root)
        if layout == 'split_class_folder':
            build_manifest(data_root, manifest, layout='split_class_folder')
        else:
            unsplit = manifest.with_name(manifest.stem + '_unsplit.csv')
            build_manifest(data_root, unsplit, layout='class_folder')
            make_stratified_splits(unsplit, manifest, train=a.train_ratio, valid=a.valid_ratio, test=a.test_ratio, seed=a.seed)
    print_manifest_summary(manifest)
    args = SupervisedArgs(
        manifest=str(manifest),
        output_root=a.output_root,
        experiment_name=a.experiment_name,
        model_name=a.model_name,
        pretrained=False,
        image_size=a.image_size,
        epochs=a.epochs,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        lr=a.lr,
        weight_decay=a.weight_decay,
        label_smoothing=a.label_smoothing,
        seed=a.seed,
        amp=True,
        patience=10,
        monitor='macro_f1',
        use_class_weights=True,
        strong_aug=True,
        allow_cpu=a.allow_cpu,
        init_checkpoint=a.checkpoint,
    )
    train_supervised(args)


if __name__ == '__main__':
    main()
