#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import create_loader, label_mapping, load_manifest
from src.models import create_classifier
from src.supervised import evaluate_classifier
from src.utils import make_run_dir, write_json


def main():
    p = argparse.ArgumentParser(description='Evaluate a supervised classifier checkpoint.')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    p.add_argument('--output-root', default='/workspace/implant_outputs')
    p.add_argument('--experiment-name', default='eval_classifier')
    p.add_argument('--model-name', default=None, help='If omitted, read from checkpoint args.')
    p.add_argument('--image-size', type=int, default=None, help='If omitted, read from checkpoint args or use 224.')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--allow-cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and not args.allow_cpu:
        raise RuntimeError('CUDA is not available. Pass --allow-cpu only for debugging.')

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    class_to_idx = {str(k): int(v) for k, v in ckpt.get('class_to_idx', {}).items()}
    if not class_to_idx:
        df = load_manifest(args.manifest, split=None)
        class_to_idx, _ = label_mapping(df)
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    ckpt_args = ckpt.get('args', {}) or {}
    model_name = args.model_name or ckpt_args.get('model_name') or ckpt_args.get('model') or 'vit_base_patch16_224'
    image_size = args.image_size or int(ckpt_args.get('image_size', 224))

    model = create_classifier(model_name, num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.to(device)

    run_dir = make_run_dir(args.output_root, args.experiment_name)
    write_json(run_dir / 'eval_args.json', vars(args))
    loader = create_loader(args.manifest, args.split, class_to_idx, image_size, args.batch_size, args.num_workers, train=False, return_path=True)
    metrics = evaluate_classifier(model, loader, device, idx_to_class, args.split, out_dir=run_dir)
    print('EVAL_RUN_DIR=', run_dir)
    print('accuracy=', metrics['accuracy'])
    print('macro_f1=', metrics['macro_f1'])


if __name__ == '__main__':
    main()
