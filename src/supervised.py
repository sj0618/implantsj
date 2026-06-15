from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast

from .data import class_weights_from_manifest, create_loader, label_mapping, load_manifest
from .metrics import compute_metrics, save_confusion_matrix, save_metrics_json, save_predictions
from .models import create_classifier, load_matching_weights, save_checkpoint
from .utils import count_parameters, log, make_run_dir, seed_everything, write_json


@dataclass
class SupervisedArgs:
    manifest: str
    output_root: str = '/workspace/implant_outputs'
    experiment_name: str = 'vit_supervised'
    model_name: str = 'vit_base_patch16_224'
    pretrained: bool = True
    image_size: int = 224
    epochs: int = 80
    batch_size: int = 32
    num_workers: int = 8
    lr: float = 5e-5
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    dropout: float = 0.1
    drop_path_rate: float = 0.1
    seed: int = 42
    amp: bool = True
    patience: int = 12
    monitor: str = 'macro_f1'
    use_class_weights: bool = False
    strong_aug: bool = True
    allow_cpu: bool = False
    resume_checkpoint: Optional[str] = None
    init_checkpoint: Optional[str] = None


def evaluate_classifier(model, loader, device, idx_to_class: Dict[int, str], split_name: str, out_dir: Optional[Path] = None) -> Dict:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    rows: List[Dict] = []
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, y, paths = batch
            else:
                x, y = batch
                paths = [''] * len(y)
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).cpu().numpy().tolist()
            true = y.numpy().tolist()
            y_true.extend(true)
            y_pred.extend(pred)
            for p, t, pr, prob in zip(paths, true, pred, probs.cpu().numpy()):
                rows.append({
                    'path': p,
                    'true_idx': int(t),
                    'true_label': idx_to_class[int(t)],
                    'pred_idx': int(pr),
                    'pred_label': idx_to_class[int(pr)],
                    'confidence': float(np.max(prob)),
                    'correct': bool(int(t) == int(pr)),
                })
    metrics = compute_metrics(y_true, y_pred, class_names)
    if out_dir is not None:
        save_metrics_json(out_dir / 'metrics' / f'{split_name}_metrics.json', metrics)
        save_confusion_matrix(
            out_dir / 'confusion_matrices' / f'{split_name}.csv',
            out_dir / 'confusion_matrices' / f'{split_name}.png',
            y_true,
            y_pred,
            class_names,
        )
        save_predictions(out_dir / 'metrics' / f'{split_name}_predictions.csv', rows)
    return metrics


def train_supervised(args: SupervisedArgs) -> Path:
    seed_everything(args.seed)
    import torch.optim as optim

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and not args.allow_cpu:
        raise RuntimeError('CUDA is not available. Pass --allow-cpu only for debugging.')

    full_df = load_manifest(args.manifest, split=None)
    class_to_idx, idx_to_class = label_mapping(full_df)
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    n_classes = len(class_names)

    run_dir = make_run_dir(args.output_root, args.experiment_name)
    write_json(run_dir / 'args.json', asdict(args))
    write_json(run_dir / 'class_to_idx.json', class_to_idx)

    log(f'Run dir: {run_dir}')
    log(f'Classes ({n_classes}): {class_names}')
    log(f'Model: {args.model_name}')
    log(f'Device: {device}')

    train_loader = create_loader(args.manifest, 'train', class_to_idx, args.image_size, args.batch_size, args.num_workers, train=True, strong_aug=args.strong_aug)
    valid_loader = create_loader(args.manifest, 'valid', class_to_idx, args.image_size, args.batch_size, args.num_workers, train=False, strong_aug=False, return_path=True)

    model = create_classifier(
        model_name=args.model_name,
        num_classes=n_classes,
        pretrained=args.pretrained,
        dropout=args.dropout,
        drop_path_rate=args.drop_path_rate,
    )
    if args.init_checkpoint:
        loaded, skipped = load_matching_weights(model, args.init_checkpoint, strict=False)
        log(f'Initialized from checkpoint: loaded={loaded}, skipped={skipped}, path={args.init_checkpoint}')
    model.to(device)
    log(f'Trainable parameters: {count_parameters(model):,}')

    if args.use_class_weights:
        weights = class_weights_from_manifest(args.manifest, class_to_idx, split='train').to(device)
    else:
        weights = None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = GradScaler(enabled=args.amp and device.type == 'cuda')

    start_epoch = 1
    best_score = -1.0
    best_metrics: Dict = {}
    history: List[Dict] = []

    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        if ckpt.get('optimizer_state_dict') is not None:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        best_score = float(ckpt.get('metrics', {}).get(args.monitor, -1.0))
        log(f'Resumed from {args.resume_checkpoint} at epoch {start_epoch}')

    patience_count = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_seen = 0
        t0 = time.time()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp and device.type == 'cuda'):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            bs = x.size(0)
            epoch_loss += float(loss.item()) * bs
            n_seen += bs
        scheduler.step()
        train_loss = epoch_loss / max(1, n_seen)
        valid_metrics = evaluate_classifier(model, valid_loader, device, idx_to_class, 'valid', out_dir=None)
        score = float(valid_metrics.get(args.monitor, valid_metrics.get('macro_f1', 0.0)))
        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'valid_accuracy': valid_metrics['accuracy'],
            'valid_macro_f1': valid_metrics['macro_f1'],
            'valid_weighted_f1': valid_metrics['weighted_f1'],
            'valid_macro_precision': valid_metrics['macro_precision'],
            'valid_macro_recall': valid_metrics['macro_recall'],
            'lr': optimizer.param_groups[0]['lr'],
            'seconds': time.time() - t0,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / 'metrics' / 'history.csv', index=False)
        log(f"epoch={epoch:03d} loss={train_loss:.4f} valid_acc={valid_metrics['accuracy']:.4f} valid_macro_f1={valid_metrics['macro_f1']:.4f} lr={row['lr']:.2e}")

        save_checkpoint(run_dir / 'checkpoints' / 'last.pt', model, optimizer, epoch, valid_metrics, asdict(args), class_to_idx)
        if score > best_score:
            best_score = score
            best_metrics = valid_metrics
            patience_count = 0
            save_checkpoint(run_dir / 'checkpoints' / 'best.pt', model, optimizer, epoch, valid_metrics, asdict(args), class_to_idx)
            save_metrics_json(run_dir / 'metrics' / 'best_valid_metrics.json', valid_metrics)
            log(f'New best {args.monitor}={best_score:.4f} at epoch={epoch}')
        else:
            patience_count += 1
            if patience_count >= args.patience:
                log(f'Early stopping at epoch={epoch}. Best {args.monitor}={best_score:.4f}')
                break

    # Final valid/test evaluation with best checkpoint.
    best_ckpt = torch.load(run_dir / 'checkpoints' / 'best.pt', map_location='cpu')
    model.load_state_dict(best_ckpt['model_state_dict'], strict=True)
    evaluate_classifier(model, valid_loader, device, idx_to_class, 'valid', out_dir=run_dir)
    try:
        test_loader = create_loader(args.manifest, 'test', class_to_idx, args.image_size, args.batch_size, args.num_workers, train=False, strong_aug=False, return_path=True)
        test_metrics = evaluate_classifier(model, test_loader, device, idx_to_class, 'test', out_dir=run_dir)
        log(f"TEST accuracy={test_metrics['accuracy']:.4f} macro_f1={test_metrics['macro_f1']:.4f}")
    except Exception as e:
        log(f'Skipped test evaluation: {repr(e)}')

    log(f'Finished. Best valid {args.monitor}={best_score:.4f}. Run dir: {run_dir}')
    return run_dir
