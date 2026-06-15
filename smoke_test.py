#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import build_manifest, make_stratified_splits, print_manifest_summary
from src.supervised import SupervisedArgs, train_supervised


def make_synthetic(root: Path, classes=3, images_per_class=18):
    if root.exists():
        shutil.rmtree(root)
    for c in range(classes):
        d = root / f'class_{c}'
        d.mkdir(parents=True, exist_ok=True)
        for i in range(images_per_class):
            img = Image.new('RGB', (96, 96), (20 + c * 60, 20 + i % 20, 40))
            draw = ImageDraw.Draw(img)
            draw.rectangle([10 + c * 5, 10, 60, 60], outline=(255, 255, 255), width=3)
            draw.text((8, 72), f'{c}-{i}', fill=(255, 255, 255))
            img.save(d / f'{i:03d}.png')


def main():
    data_root = Path('/workspace/data/_implant_smoke')
    manifest_unsplit = Path('/workspace/data/manifests/_implant_smoke_unsplit.csv')
    manifest = Path('/workspace/data/manifests/_implant_smoke.csv')
    make_synthetic(data_root)
    build_manifest(data_root, manifest_unsplit, layout='class_folder')
    make_stratified_splits(manifest_unsplit, manifest, train=6, valid=2, test=2, seed=42)
    print_manifest_summary(manifest)
    args = SupervisedArgs(
        manifest=str(manifest),
        output_root='/workspace/implant_outputs',
        experiment_name='smoke_test_resnet18',
        model_name='resnet18',
        pretrained=False,
        image_size=96,
        epochs=1,
        batch_size=8,
        num_workers=0,
        lr=1e-3,
        weight_decay=1e-4,
        label_smoothing=0.0,
        dropout=0.0,
        drop_path_rate=0.0,
        patience=1,
        amp=False,
        allow_cpu=True,
    )
    run_dir = train_supervised(args)
    print('SMOKE_TEST_OK', run_dir)


if __name__ == '__main__':
    main()
