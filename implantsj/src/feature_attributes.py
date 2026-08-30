"""13 feature attribute 예측 실험 (ViT supervised, VLM 미사용).

정답은 `configs/class_features.csv` (클래스 -> 13 feature) 에서만 나온다.
목적은 클래스 이름을 맞히는 게 아니라, **처음 보는 클래스**의 이미지를 보고
그 클래스가 13 feature 각각에 대해 어떤 값을 갖는지 표를 채우는 것이다.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from .class_features import (
    DEFAULT_TABLE,
    UNKNOWN,
    build_targets,
    load_aliases,
    load_table,
    match_labels,
)
from .data import create_loader, load_manifest
from .models import AttributeModel, load_matching_weights
from .utils import count_parameters, log, make_run_dir, seed_everything, write_json


@dataclass
class AttributeArgs:
    manifest: str
    feature_table: str = str(DEFAULT_TABLE)
    label_aliases: Optional[str] = None
    attributes: Optional[str] = None          # 콤마 구분. 비우면 Name 제외 12개
    output_root: str = '/workspace/outputs'
    experiment_name: str = 'vit_feature_attr'
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
    strong_aug: bool = True
    allow_cpu: bool = False
    init_checkpoint: Optional[str] = None


# ---------------------------------------------------------------- 데이터 준비


def prepare_labels(
    manifest: str,
    feature_table: str,
    label_aliases: Optional[str] = None,
    attributes: Optional[Sequence[str]] = None,
    splits: Sequence[str] = ('train', 'valid', 'test'),
):
    """manifest 의 클래스를 표에 붙이고, 학습 가능한 attribute 만 남긴다.

    매칭 실패 클래스는 버리되 **개수와 이름을 그대로 돌려준다**. 조용히 사라지면
    분모가 조작되므로 호출부에서 반드시 기록한다.
    """
    df_all = load_manifest(manifest, split=None)
    labels = sorted(df_all['label'].astype(str).unique().tolist())
    table_df = load_table(feature_table)
    aliases = load_aliases(label_aliases)
    mapping, unmatched = match_labels(labels, table_df, aliases)
    matched = [l for l in labels if l in mapping]
    if not matched:
        raise RuntimeError(
            f'표({feature_table})에 붙는 클래스가 하나도 없습니다. '
            f'데이터셋 클래스 예: {labels[:5]}'
        )
    table, target = build_targets(matched, mapping, table_df, attributes)
    class_to_idx = {c: i for i, c in enumerate(matched)}
    return class_to_idx, table, target, unmatched


def _masked_ce(logits: Dict[str, torch.Tensor], target: torch.Tensor, attributes: List[str], smoothing: float):
    """unknown(-1) 은 loss 에서 뺀다. 정답을 모르는 칸을 틀렸다고 셀 수는 없다."""
    total = logits[attributes[0]].new_zeros(())
    counted = 0
    for ai, a in enumerate(attributes):
        t = target[:, ai]
        keep = t >= 0
        if not bool(keep.any()):
            continue
        total = total + nn.functional.cross_entropy(
            logits[a][keep], t[keep], label_smoothing=smoothing
        )
        counted += 1
    return total / max(counted, 1)


# ---------------------------------------------------------------------- 평가


def evaluate_attributes(model, loader, device, table, target_lut: torch.Tensor, split_name: str, out_dir: Optional[Path] = None) -> Dict:
    """attribute 별 정확도/macro-F1 + 클래스별 예측 표를 만든다."""
    from sklearn.metrics import f1_score

    model.eval()
    attrs = table.attributes
    per_attr_true: Dict[str, List[int]] = {a: [] for a in attrs}
    per_attr_pred: Dict[str, List[int]] = {a: [] for a in attrs}
    rows: List[Dict] = []
    # 클래스별 다수결로 "표 한 칸"을 채우기 위한 집계
    votes: Dict[str, Dict[str, Counter]] = {}
    idx_to_class = {i: c for c, i in loader.dataset.class_to_idx.items()}
    non_blocking = device.type == 'cuda'

    with torch.no_grad():
        for batch in loader:
            x, y, paths = batch if len(batch) == 3 else (*batch, [''] * len(batch[1]))
            x = x.to(device, non_blocking=non_blocking)
            logits = model(x)
            tgt = target_lut[y]
            for bi in range(len(y)):
                cls = idx_to_class[int(y[bi])]
                votes.setdefault(cls, {a: Counter() for a in attrs})
                row = {'path': paths[bi] if bi < len(paths) else '', 'class': cls}
                for ai, a in enumerate(attrs):
                    pi = int(logits[a][bi].argmax())
                    pv = table.vocab[a][pi]
                    votes[cls][a][pv] += 1
                    ti = int(tgt[bi, ai])
                    row[f'{a}_pred'] = pv
                    row[f'{a}_true'] = table.vocab[a][ti] if ti >= 0 else UNKNOWN
                    if ti >= 0:
                        per_attr_true[a].append(ti)
                        per_attr_pred[a].append(pi)
                rows.append(row)

    metrics: Dict[str, object] = {'split': split_name, 'per_attribute': {}}
    accs, f1s = [], []
    for a in attrs:
        yt, yp = per_attr_true[a], per_attr_pred[a]
        if not yt:
            metrics['per_attribute'][a] = {'n': 0, 'note': '정답이 전부 unknown 이라 평가 불가'}
            continue
        acc = float(np.mean(np.array(yt) == np.array(yp)))
        f1 = float(f1_score(yt, yp, average='macro', zero_division=0))
        metrics['per_attribute'][a] = {'n': len(yt), 'accuracy': acc, 'macro_f1': f1,
                                       'values': table.vocab[a]}
        accs.append(acc)
        f1s.append(f1)
    metrics['mean_accuracy'] = float(np.mean(accs)) if accs else 0.0
    metrics['mean_macro_f1'] = float(np.mean(f1s)) if f1s else 0.0
    # 13 feature 를 한 장에서 전부 맞힌 비율. 표를 통째로 채우는 실사용 기준.
    exact = [all(r[f'{a}_pred'] == r[f'{a}_true'] for a in attrs if r[f'{a}_true'] != UNKNOWN) for r in rows]
    metrics['exact_row_match'] = float(np.mean(exact)) if exact else 0.0

    if out_dir is not None:
        out_dir = Path(out_dir)
        (out_dir / 'metrics').mkdir(parents=True, exist_ok=True)
        write_json(out_dir / 'metrics' / f'{split_name}_attribute_metrics.json', metrics)
        pd.DataFrame(rows).to_csv(out_dir / 'metrics' / f'{split_name}_predictions.csv', index=False)
        # 클래스별 다수결 -> 채워진 표
        filled = []
        for cls, vv in sorted(votes.items()):
            rec = {'class': cls, 'n_images': sum(vv[attrs[0]].values())}
            ci = loader.dataset.class_to_idx[cls]
            for ai, a in enumerate(attrs):
                pv, n = vv[a].most_common(1)[0]
                ti = int(target_lut[ci, ai])
                rec[a] = pv
                rec[f'{a}_agree'] = round(n / max(sum(vv[a].values()), 1), 3)
                rec[f'{a}_true'] = table.vocab[a][ti] if ti >= 0 else UNKNOWN
            filled.append(rec)
        pd.DataFrame(filled).to_csv(out_dir / 'metrics' / f'{split_name}_filled_table.csv', index=False)
    return metrics


# ---------------------------------------------------------------------- 학습


def train_feature_attributes(args: AttributeArgs) -> Path:
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else ('cpu' if args.allow_cpu else 'cuda'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA 를 못 잡았습니다. --allow-cpu 를 주면 CPU 로 돕니다.')

    attrs_arg = [a.strip() for a in args.attributes.split(',')] if args.attributes else None
    class_to_idx, table, target, unmatched = prepare_labels(
        args.manifest, args.feature_table, args.label_aliases, attrs_arg
    )
    run_dir = make_run_dir(args.output_root, args.experiment_name)
    log(f'run_dir={run_dir}')
    log(f'클래스 {len(class_to_idx)}개 학습. 표에 못 붙은 클래스 {len(unmatched)}개: {unmatched}')
    if table.dropped_constant:
        log(f'값이 1종뿐이라 제외한 attribute: {table.dropped_constant}')
    log(f'학습 attribute {len(table.attributes)}개: {table.attributes}')

    target_lut = torch.from_numpy(target).to(device)
    write_json(run_dir / 'label_setup.json', {
        'classes': list(class_to_idx.keys()),
        'unmatched_classes': unmatched,
        'attributes': table.attributes,
        'dropped_constant_attributes': table.dropped_constant,
        'vocab': table.vocab,
        'unknown_cells': int((target < 0).sum()),
    })

    mk = lambda split, train: create_loader(
        args.manifest, split, class_to_idx, args.image_size, args.batch_size,
        args.num_workers, train=train, strong_aug=args.strong_aug and train, return_path=True,
    )
    train_loader = mk('train', True)
    manifest_splits = set(load_manifest(args.manifest, split=None)['split'].astype(str))
    valid_split = 'valid' if 'valid' in manifest_splits else 'test'
    if valid_split == 'test':
        log('valid split 이 없어 test 를 모니터링에 씁니다. 조기종료 기준이 test 에 노출됩니다.')
    valid_loader = mk(valid_split, False)

    head_sizes = {a: len(table.vocab[a]) for a in table.attributes}
    model = AttributeModel(args.model_name, head_sizes, pretrained=args.pretrained,
                           dropout=args.dropout, drop_path_rate=args.drop_path_rate).to(device)
    if args.init_checkpoint:
        loaded, skipped = load_matching_weights(model, args.init_checkpoint)
        log(f'init_checkpoint 에서 {loaded} 텐서 로드, {skipped} 건너뜀')
    log(f'params={count_parameters(model):,}')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    use_amp = args.amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best, best_epoch, history = -1.0, -1, []
    ckpt_dir = run_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, nb = 0.0, 0
        for batch in train_loader:
            x, y = batch[0].to(device, non_blocking=device.type == 'cuda'), batch[1].to(device)
            t = target_lut[y]
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = _masked_ce(model(x), t, table.attributes, args.label_smoothing)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.detach())
            nb += 1
        sched.step()

        vm = evaluate_attributes(model, valid_loader, device, table, target_lut, valid_split)
        score = float(vm['mean_macro_f1'])
        history.append({'epoch': epoch, 'train_loss': running / max(nb, 1),
                        'valid_mean_macro_f1': score, 'valid_mean_accuracy': vm['mean_accuracy'],
                        'valid_exact_row_match': vm['exact_row_match'],
                        'lr': opt.param_groups[0]['lr']})
        log(f"epoch={epoch:03d} loss={running/max(nb,1):.4f} "
            f"{valid_split}_mean_macro_f1={score:.4f} exact_row={vm['exact_row_match']:.4f}")

        torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                    'args': asdict(args), 'class_to_idx': class_to_idx,
                    'attributes': table.attributes, 'vocab': table.vocab}, ckpt_dir / 'last.pt')
        if score > best:
            best, best_epoch = score, epoch
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                        'args': asdict(args), 'class_to_idx': class_to_idx,
                        'attributes': table.attributes, 'vocab': table.vocab}, ckpt_dir / 'best.pt')
        pd.DataFrame(history).to_csv(run_dir / 'history.csv', index=False)
        if epoch - best_epoch >= args.patience:
            log(f'{args.patience} epoch 개선 없음. 조기종료.')
            break

    model.load_state_dict(torch.load(ckpt_dir / 'best.pt', map_location=device)['model_state_dict'])
    evaluate_attributes(model, valid_loader, device, table, target_lut, valid_split, out_dir=run_dir)
    if 'test' in manifest_splits and valid_split != 'test':
        evaluate_attributes(model, mk('test', False), device, table, target_lut, 'test', out_dir=run_dir)
    log(f'끝. best {valid_split}_mean_macro_f1={best:.4f} (epoch {best_epoch}). run_dir={run_dir}')
    return run_dir
