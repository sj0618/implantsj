from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from .data import ImageClassificationDataset, build_transforms, label_mapping, load_manifest
from .metrics import compute_metrics, save_confusion_matrix, save_metrics_json, save_predictions
from .models import MetricModel, save_checkpoint
from .utils import count_parameters, log, make_run_dir, seed_everything, write_json


class TwoViewDataset(Dataset):
    def __init__(self, manifest: str, split: str, class_to_idx: Dict[str, int], image_size: int):
        self.base1 = ImageClassificationDataset(manifest, split=split, class_to_idx=class_to_idx, transform=build_transforms(image_size, train=True, strong_aug=True), return_path=False)
        self.base2 = ImageClassificationDataset(manifest, split=split, class_to_idx=class_to_idx, transform=build_transforms(image_size, train=True, strong_aug=True), return_path=False)
    def __len__(self):
        return len(self.base1)
    def __getitem__(self, idx):
        x1, y = self.base1[idx]
        x2, _ = self.base2[idx]
        return x1, x2, y


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # features: [2B, D], labels: [2B]
        features = nn.functional.normalize(features, dim=1)
        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        mask = torch.eq(labels[:, None], labels[None, :]).float().to(features.device)
        self_mask = torch.eye(mask.size(0), device=features.device)
        mask = mask * (1.0 - self_mask)
        exp_logits = torch.exp(logits) * (1.0 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        loss = -mean_log_prob_pos.mean()
        return loss


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(embeddings, embeddings, p=2)
        labels_equal = labels[:, None] == labels[None, :]
        eye = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
        pos_mask = labels_equal & ~eye
        neg_mask = ~labels_equal
        hardest_pos = torch.where(pos_mask, dist, torch.zeros_like(dist)).max(dim=1).values
        hardest_neg = torch.where(neg_mask, dist, torch.full_like(dist, 1e6)).min(dim=1).values
        loss = nn.functional.relu(hardest_pos - hardest_neg + self.margin)
        return loss.mean()


@dataclass
class MetricArgs:
    manifest: str
    output_root: str = '/workspace/implant_outputs'
    experiment_name: str = 'vit_metric_supcon'
    model_name: str = 'vit_base_patch16_224'
    pretrained: bool = True
    image_size: int = 224
    epochs: int = 80
    batch_size: int = 32
    num_workers: int = 8
    lr: float = 3e-5
    weight_decay: float = 0.05
    loss_name: str = 'supcon'
    temperature: float = 0.07
    triplet_margin: float = 0.2
    embedding_dim: int = 128
    dropout: float = 0.1
    drop_path_rate: float = 0.1
    seed: int = 42
    amp: bool = True
    patience: int = 12
    allow_cpu: bool = False


def extract_embeddings(model, loader, device):
    model.eval()
    zs, ys, paths = [], [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, y, p = batch
            else:
                x, y = batch
                p = [''] * len(y)
            x = x.to(device, non_blocking=True)
            z = model(x).cpu().numpy()
            zs.append(z)
            ys.extend(y.numpy().tolist())
            paths.extend(list(p))
    return np.concatenate(zs, axis=0), np.asarray(ys), paths


def prototype_predict(train_z, train_y, query_z):
    classes = sorted(set(train_y.tolist()))
    protos = []
    for c in classes:
        proto = train_z[train_y == c].mean(axis=0)
        proto = proto / max(1e-12, np.linalg.norm(proto))
        protos.append(proto)
    protos = np.stack(protos, axis=0)
    q = query_z / np.maximum(1e-12, np.linalg.norm(query_z, axis=1, keepdims=True))
    sims = q @ protos.T
    pred_class_positions = sims.argmax(axis=1)
    pred = np.asarray([classes[i] for i in pred_class_positions])
    conf = sims.max(axis=1)
    return pred, conf


def evaluate_metric_model(model, manifest: str, class_to_idx: Dict[str, int], image_size: int, batch_size: int, num_workers: int, device, out_dir: Optional[Path] = None, split: str = 'valid'):
    from .data import create_loader
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    train_loader = create_loader(manifest, 'train', class_to_idx, image_size, batch_size, num_workers, train=False, return_path=True)
    query_loader = create_loader(manifest, split, class_to_idx, image_size, batch_size, num_workers, train=False, return_path=True)
    train_z, train_y, _ = extract_embeddings(model, train_loader, device)
    query_z, query_y, paths = extract_embeddings(model, query_loader, device)
    pred, conf = prototype_predict(train_z, train_y, query_z)
    metrics = compute_metrics(query_y, pred, class_names)
    if out_dir is not None:
        save_metrics_json(out_dir / 'metrics' / f'{split}_prototype_metrics.json', metrics)
        save_confusion_matrix(out_dir / 'confusion_matrices' / f'{split}_prototype.csv', out_dir / 'confusion_matrices' / f'{split}_prototype.png', query_y, pred, class_names)
        rows = []
        for p, t, pr, cf in zip(paths, query_y, pred, conf):
            rows.append({'path': p, 'true_idx': int(t), 'true_label': idx_to_class[int(t)], 'pred_idx': int(pr), 'pred_label': idx_to_class[int(pr)], 'similarity': float(cf), 'correct': bool(int(t)==int(pr))})
        save_predictions(out_dir / 'metrics' / f'{split}_prototype_predictions.csv', rows)
    return metrics


def train_metric(args: MetricArgs) -> Path:
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and not args.allow_cpu:
        raise RuntimeError('CUDA is not available. Pass --allow-cpu only for debugging.')

    df = load_manifest(args.manifest, split=None)
    class_to_idx, idx_to_class = label_mapping(df)
    run_dir = make_run_dir(args.output_root, args.experiment_name)
    write_json(run_dir / 'args.json', asdict(args))
    write_json(run_dir / 'class_to_idx.json', class_to_idx)
    log(f'Run dir: {run_dir}')
    log(f'Metric model: {args.model_name}, loss={args.loss_name}')

    ds = TwoViewDataset(args.manifest, 'train', class_to_idx, args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=True)
    model = MetricModel(args.model_name, embedding_dim=args.embedding_dim, pretrained=args.pretrained, dropout=args.dropout, drop_path_rate=args.drop_path_rate).to(device)
    log(f'Trainable parameters: {count_parameters(model):,}')

    if args.loss_name.lower() == 'supcon':
        criterion = SupConLoss(args.temperature)
    elif args.loss_name.lower() == 'triplet':
        criterion = BatchHardTripletLoss(args.triplet_margin)
    else:
        raise ValueError(f'Unknown metric loss: {args.loss_name}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = GradScaler(enabled=args.amp and device.type == 'cuda')

    best_score = -1.0
    patience_count = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        t0 = time.time()
        for x1, x2, y in loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp and device.type == 'cuda'):
                if args.loss_name.lower() == 'supcon':
                    x = torch.cat([x1, x2], dim=0)
                    labels = torch.cat([y, y], dim=0)
                    z = model(x)
                    loss = criterion(z, labels)
                else:
                    z = model(x1)
                    loss = criterion(z, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            bs = y.size(0)
            total_loss += float(loss.item()) * bs
            n += bs
        scheduler.step()
        train_loss = total_loss / max(1, n)
        valid_metrics = evaluate_metric_model(model, args.manifest, class_to_idx, args.image_size, args.batch_size, args.num_workers, device, out_dir=None, split='valid')
        score = valid_metrics['macro_f1']
        row = {'epoch': epoch, 'train_loss': train_loss, 'valid_accuracy': valid_metrics['accuracy'], 'valid_macro_f1': valid_metrics['macro_f1'], 'lr': optimizer.param_groups[0]['lr'], 'seconds': time.time()-t0}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / 'metrics' / 'history.csv', index=False)
        log(f"epoch={epoch:03d} loss={train_loss:.4f} valid_acc={valid_metrics['accuracy']:.4f} valid_macro_f1={valid_metrics['macro_f1']:.4f}")
        save_checkpoint(run_dir / 'checkpoints' / 'last.pt', model, optimizer, epoch, valid_metrics, asdict(args), class_to_idx)
        if score > best_score:
            best_score = score
            patience_count = 0
            save_checkpoint(run_dir / 'checkpoints' / 'best.pt', model, optimizer, epoch, valid_metrics, asdict(args), class_to_idx)
            save_metrics_json(run_dir / 'metrics' / 'best_valid_prototype_metrics.json', valid_metrics)
            log(f'New best valid prototype macro_f1={best_score:.4f}')
        else:
            patience_count += 1
            if patience_count >= args.patience:
                log(f'Early stopping at epoch={epoch}')
                break

    best = torch.load(run_dir / 'checkpoints' / 'best.pt', map_location='cpu')
    model.load_state_dict(best['model_state_dict'], strict=True)
    evaluate_metric_model(model, args.manifest, class_to_idx, args.image_size, args.batch_size, args.num_workers, device, out_dir=run_dir, split='valid')
    try:
        test_metrics = evaluate_metric_model(model, args.manifest, class_to_idx, args.image_size, args.batch_size, args.num_workers, device, out_dir=run_dir, split='test')
        log(f"TEST prototype accuracy={test_metrics['accuracy']:.4f} macro_f1={test_metrics['macro_f1']:.4f}")
    except Exception as e:
        log(f'Skipped test metric evaluation: {repr(e)}')
    return run_dir
