"""few-shot 논문 프로토콜의 seen 학습과 임베딩 추출 (docs/paper/FEWSHOT_ATTRIBUTE_SPEC.md §3, §4, §6.4).

  train  : 분할 r 에서 역할이 train 인 항목만으로 인코더(MetricModel)를 학습하고, 매 epoch seen val 의
           cosine 프로토타입 정확도로 최고 state 를 고른다. unseen·excluded·test 항목은 읽지 않는다.
             ce           : Linear(128, n_seen)(z)                                   -> SimpleShot 표현
             protonet     : N=min(10, 자격 seen) × (K=5 + Q=5) 에피소드, logits = -‖q − μ_c‖² (raw z)
             arcface      : ArcFaceHead(L2(z))
             arcface_attr : arcface + w · mean_a CE_a(Linear(L2(z)), unknown 무시)     (제안, §4)
  export : manifest 전체를 manifest 순서로 임베딩해 emb.npz 로 쓴다(src.fewshot.evaluate 의 입력).
           --ckpt 는 projector 출력 z(정규화 전 128d) + 특징 헤드 확률, --pretrained 는 동결 백본 pooled feature.

  python -m src.fewshot_train train  --manifest m.csv --splits s.json --split-id 0 --loss arcface_attr \
                                     --model-name vit_small_patch16_224.augreg_in21k_ft_in1k --out runs/x
  python -m src.fewshot_train export --manifest m.csv --ckpt runs/x/model.pt --out emb.npz
  python -m src.fewshot_train export --manifest m.csv --pretrained vit_base_patch14_dinov2.lvd142m --out emb.npz
  python -m src.fewshot_train selftest   # 합성 이미지 + 무작위 초기화 ViT-Tiny 로 train -> export -> evaluate 계약 확인
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .arcface import ArcFaceHead, _resolve_device
from .class_features import DEFAULT_TABLE
from .data import ImageClassificationDataset, build_transforms
from .fewshot import _feature_rows, _read_manifest, check_split, classify, item_roles
from .models import MetricModel, load_checkpoint
from .utils import log, seed_everything, write_json

LOSSES = ('ce', 'protonet', 'arcface', 'arcface_attr')
EMB_DIM, IMG = 128, 224
PROTO_WAY, PROTO_K, PROTO_Q = 10, 5, 5


def _build(cfg, n_seen, vocab, pretrained):
    """인코더 + loss 헤드. state_dict 키는 encoder.* / cls.* / arc.* / attr.*"""
    net = nn.Module()
    net.encoder = MetricModel(cfg['model_name'], EMB_DIM, pretrained=pretrained)
    if cfg['loss'] == 'ce':
        net.cls = nn.Linear(EMB_DIM, n_seen)
    if cfg['loss'].startswith('arcface'):
        net.arc = ArcFaceHead(EMB_DIM, n_seen, s=cfg['scale'], m=cfg['margin'])
    if vocab:
        net.attr = nn.ModuleDict({a: nn.Linear(EMB_DIM, len(v)) for a, v in vocab.items()})
    return net


def _z(net, x):
    return net.encoder.projector(net.encoder.forward_features(x))  # 정규화 전 z


def _transforms(backbone, train):
    """src.data.build_transforms 의 기하·증강을 그대로 쓰고 Normalize 만 백본 사전학습 통계로 바꾼다.

    build_transforms 는 ImageNet mean/std 고정인데 augreg ViT 는 0.5/0.5 로 사전학습됐다(DINOv2 는 ImageNet).
    반환: (transform, {'mean', 'std'})
    """
    from timm.data import resolve_model_data_config
    from torchvision import transforms

    c = resolve_model_data_config(backbone)
    t = build_transforms(IMG, train=train)
    assert isinstance(t.transforms[-1], transforms.Normalize), t
    t.transforms[-1] = transforms.Normalize(mean=c['mean'], std=c['std'])
    return t, {'mean': list(c['mean']), 'std': list(c['std'])}


class _Episodes:
    """ProtoNet batch_sampler. 배치 = N 클래스 × (K 서포트 + Q query), 클래스 블록 순서로 이어 붙인다."""

    def __init__(self, y, epoch_size, seed):
        self.by_class = {c: np.flatnonzero(y == c) for c in np.unique(y)}
        # 복원추출하면 같은 사진이 서포트와 query 에 동시에 들어가므로, K+Q 장 미만 클래스는 빼고 summary 에 남긴다
        self.eligible = sorted(c for c, idx in self.by_class.items() if len(idx) >= PROTO_K + PROTO_Q)
        if len(self.eligible) < 2:
            raise ValueError(f'protonet: train 항목이 K+Q={PROTO_K + PROTO_Q}장 이상인 seen 클래스가 '
                             f'{len(self.eligible)}개뿐이다(2개 이상 필요)')
        self.n_way = min(PROTO_WAY, len(self.eligible))
        self.n = epoch_size // (self.n_way * (PROTO_K + PROTO_Q))
        assert self.n >= 1, f'epoch_size={epoch_size} < 에피소드 1개({self.n_way}×{PROTO_K + PROTO_Q})'
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def __iter__(self):
        for _ in range(self.n):
            yield [int(i) for c in self.rng.choice(self.eligible, self.n_way, replace=False)
                   for i in self.rng.choice(self.by_class[c], PROTO_K + PROTO_Q, replace=False)]


def _loss(net, cfg, x, y, attrs, attr_t):
    """(loss, 정확도용 logits, 정확도용 target)"""
    z = _z(net, x)
    if cfg['loss'] == 'protonet':
        z = z.view(-1, PROTO_K + PROTO_Q, z.shape[1])
        proto, q = z[:, :PROTO_K].mean(1), z[:, PROTO_K:].reshape(-1, z.shape[2])
        logits = -((q[:, None] - proto[None]) ** 2).sum(-1)
        target = torch.arange(len(proto), device=x.device).repeat_interleave(PROTO_Q)
        return F.cross_entropy(logits, target), logits, target
    if cfg['loss'] == 'ce':
        logits = net.cls(z)
        return F.cross_entropy(logits, y), logits, y
    u = F.normalize(z, dim=1)
    logits = net.arc(u, y)
    loss = F.cross_entropy(logits, y)
    if attrs:
        t = attr_t[y]
        # 배치 안에서 전부 unknown 인 feature 는 CE 가 nan 이 되므로 뺀다
        parts = [F.cross_entropy(net.attr[a](u), t[:, ai], ignore_index=-1)
                 for ai, a in enumerate(attrs) if (t[:, ai] >= 0).any()]
        if parts:
            loss = loss + cfg['attr_weight'] * torch.stack(parts).mean()
    return loss, logits, y


@torch.no_grad()
def _select_acc(net, loader, sel_df, dev):
    """seen val 의 cosine 프로토타입 정확도(centroid = train 부분집합). 분류기는 src.fewshot.classify('cos')."""
    net.eval()
    Z = torch.cat([_z(net, x.to(dev)).float().cpu() for x, _ in loader]).numpy()
    lab = sel_df['label'].to_numpy(dtype=str)
    val = (sel_df['role'] == 'val').to_numpy()
    return float((classify('cos', Z[~val], lab[~val], Z[val]) == lab[val]).mean())


def _loader_kw(num_workers):
    return {'num_workers': num_workers, 'persistent_workers': num_workers > 0}


def train(cfg: dict, pretrained: bool = True) -> Path:
    t0 = time.time()
    cfg = {k: v for k, v in cfg.items() if k != 'cmd'}
    dev = _resolve_device(cfg['device'])
    seed_everything(cfg['seed'])
    out = Path(cfg['out'])
    out.mkdir(parents=True, exist_ok=True)

    df = _read_manifest(cfg['manifest'])
    split = next(sp for sp in json.loads(Path(cfg['splits']).read_text())
                 if str(sp['split_id']) == str(cfg['split_id']))
    check_split(df, split)
    role = item_roles(df, split)
    tr, va = df[role == 'train'], df[role == 'val']
    seen = sorted(tr['label'].unique())
    c2i = {c: i for i, c in enumerate(seen)}

    attrs, vocab, attr_t = [], {}, None
    if cfg['loss'] == 'arcface_attr' and cfg['attr_weight'] > 0:  # w=0 은 arcface 와 같다(§4) -> 헤드 없음
        keyed = df if 'class_key' in df.columns else df.assign(class_key=df['label'])
        ftab, rows = _feature_rows(keyed, seen, cfg['feature_table'])
        attrs, vocab = ftab.attributes, ftab.vocab
        assert attrs, '학습할 feature 가 없다(모두 상수)'
        attr_t = torch.as_tensor(rows, device=dev)

    # run_dir/items.csv: 학습(split=train) 과 모델 선택(split=select: 클래스당 train 최대 N장 + val) 항목.
    # select 의 train 행은 train 에도 있다(centroid 용 중복). 이 파일이 학습에 쓴 항목의 기록이다.
    rng = np.random.default_rng(cfg['seed'])
    cent = tr.loc[sorted(i for _, g in tr.groupby('label') for i in
                         rng.choice(g.index.to_numpy(), min(len(g), cfg['val_max_per_class']), replace=False))]
    cols = ['path', 'label', 'group', 'role', 'split']
    items = pd.concat([tr.assign(role='train', split='train'), cent.assign(role='train', split='select'),
                       va.assign(role='val', split='select')])[cols]
    assert not set(items['label']) & set(split['unseen_classes'])
    items.to_csv(out / 'items.csv', index=False)

    net = _build(cfg, len(seen), vocab, pretrained).to(dev)
    tf_train, norm = _transforms(net.encoder.backbone, train=True)
    ds_tr = ImageClassificationDataset(out / 'items.csv', 'train', c2i, tf_train)
    assert ds_tr.df['label'].isin(c2i).all(), 'items.csv 를 다시 읽으며 라벨 문자열이 바뀌었다(숫자형 라벨?)'
    y_tr = ds_tr.df['label'].map(c2i).to_numpy()
    kw = _loader_kw(cfg['num_workers'])
    proto = None
    if cfg['loss'] == 'protonet':
        proto = _Episodes(y_tr, cfg['epoch_size'], cfg['seed'])
        loader = DataLoader(ds_tr, batch_sampler=proto, **kw)
    else:
        assert cfg['epoch_size'] >= cfg['batch_size'], 'epoch_size < batch_size (drop_last 로 배치 0개)'
        sampler = WeightedRandomSampler(1.0 / np.bincount(y_tr)[y_tr], cfg['epoch_size'], replacement=True)
        loader = DataLoader(ds_tr, batch_size=cfg['batch_size'], sampler=sampler, drop_last=True, **kw)
    sel = None
    if len(va):
        ds_sel = ImageClassificationDataset(out / 'items.csv', 'select', c2i, _transforms(net.encoder.backbone, False)[0])
        sel = (DataLoader(ds_sel, batch_size=64, **kw), ds_sel.df)
    else:
        log('[fewshot_train] val 항목이 없다 -> 마지막 epoch 를 고른다')

    backbone = list(net.encoder.backbone.parameters())
    ids = {id(p) for p in backbone}
    opt = torch.optim.AdamW([{'params': backbone, 'lr': cfg['lr']},
                             {'params': [p for p in net.parameters() if id(p) not in ids], 'lr': cfg['head_lr']}],
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg['epochs']))
    log(f"[fewshot_train] {cfg['loss']} {cfg['model_name']} device={dev} seen={len(seen)} "
        f"train={len(tr)} val={len(va)} attrs={len(attrs)}"
        + (f' protonet {proto.n_way}-way x {proto.n} episodes' if proto else ''))

    best, best_ep, best_acc, hist, min_val = None, 0, None, [], 20
    for ep in range(1, cfg['epochs'] + 1):
        te = time.time()
        net.train()
        tot = ok = n = 0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            loss, logits, target = _loss(net, cfg, x, y, attrs, attr_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(target)
            ok += int((logits.argmax(1) == target).sum())
            n += len(target)
        lr_now = opt.param_groups[0]['lr']
        sched.step()
        val_acc = _select_acc(net, *sel, dev) if sel else None
        hist.append({'epoch': ep, 'loss': tot / max(1, n), 'train_acc': ok / max(1, n), 'val_acc': val_acc,
                     'lr': lr_now, 'seconds': time.time() - te})
        log(f"[fewshot_train] epoch {ep:03d} loss={hist[-1]['loss']:.4f} train_acc={hist[-1]['train_acc']:.3f} "
            f"val_acc={'-' if val_acc is None else f'{val_acc:.3f}'} ({hist[-1]['seconds']:.1f}s)")
        # 동률이면 뒤 epoch 를 고른다(앞을 고르면 val 이 작을 때 epoch 1~2 의 덜 학습된 모델이 뽑힌다).
        # val 이 min_val 장 미만이면 선택이 잡음이라 마지막 epoch 를 쓴다(트랙 S 는 val 5~9장).
        if val_acc is None or best_acc is None or val_acc >= best_acc or len(va) < min_val:
            best = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            best_ep, best_acc = ep, val_acc

    # model.pt 가 있으면 suite 가 '학습 완료'로 본다 -> 임시 이름으로 다 쓴 뒤 마지막에 옮긴다
    torch.save({'state_dict': best, 'cfg': cfg, 'class_to_idx': c2i, 'attributes': attrs, 'vocab': vocab,
                'best_epoch': best_ep, 'normalize': norm}, out / 'model.partial.pt')
    write_json(out / 'history.json', {'history': hist})
    summary = {'best_epoch': best_ep, 'best_val_acc': best_acc,
               'selection': ('seen val cosine prototype acc (동률이면 뒤 epoch)' if sel and len(va) >= min_val
                             else f'last epoch (val {len(va)}장 < {min_val})'),
               'n_train': len(tr), 'n_val': len(va), 'n_centroid': len(cent), 'n_seen': len(seen),
               'attributes': attrs, 'normalize': norm, 'device': dev, 'seconds': time.time() - t0, 'cfg': cfg}
    if proto:
        summary['protonet'] = {'n_way': proto.n_way, 'k': PROTO_K, 'q': PROTO_Q, 'episodes_per_epoch': proto.n,
                               'n_eligible': len(proto.eligible),
                               'ineligible': sorted(seen[c] for c in set(proto.by_class) - set(proto.eligible))}
    write_json(out / 'summary.json', summary)
    os.replace(out / 'model.partial.pt', out / 'model.pt')
    log(f"[fewshot_train] best epoch {best_ep} val_acc={best_acc} -> {out / 'model.pt'} ({summary['seconds']:.1f}s)")
    return out


@torch.no_grad()
def export(manifest, out, ckpt=None, pretrained=None, batch_size=64, num_workers=4, device='auto') -> Path:
    assert (ckpt is None) != (pretrained is None), '--ckpt 와 --pretrained 중 하나만'
    t0 = time.time()
    dev = _resolve_device(device)
    attrs, vocab = [], {}
    if ckpt:
        obj = load_checkpoint(ckpt)
        attrs, vocab = obj['attributes'], obj['vocab']
        net = _build(obj['cfg'], len(obj['class_to_idx']), vocab, pretrained=False)
        net.load_state_dict(obj['state_dict'])
    else:
        import timm
        net = timm.create_model(pretrained, pretrained=True, num_classes=0,
                                **({'img_size': IMG} if 'dinov2' in pretrained else {}))
    net.to(dev).eval()
    tf, norm = _transforms(net.encoder.backbone if ckpt else net, train=False)
    ds = ImageClassificationDataset(manifest, None, {}, tf)
    ds.df = _read_manifest(manifest)  # 상대경로를 manifest 기준으로 푼 같은 행(평가의 path 와 문자열이 같아야 맞춰진다)
    embs, probs = [], {a: [] for a in attrs}
    for x, _ in DataLoader(ds, batch_size=batch_size, **_loader_kw(num_workers)):
        x = x.to(dev)
        z = _z(net, x) if ckpt else net(x)
        for a in attrs:
            probs[a].append(F.softmax(net.attr[a](F.normalize(z, dim=1)), dim=1).float().cpu())
        embs.append(z.float().cpu())
    arrays = {'path': ds.df['path'].to_numpy(dtype=str), 'emb': torch.cat(embs).numpy().astype(np.float32)}
    for a in attrs:
        arrays[f'attrprob__{a}'] = torch.cat(probs[a]).numpy().astype(np.float32)
        arrays[f'attrvocab__{a}'] = np.array(vocab[a], dtype=str)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    log(f"[fewshot_train] export {len(ds)} items emb{arrays['emb'].shape} attrs={len(attrs)} normalize={norm} -> {out} "
        f"({time.time() - t0:.1f}s)")
    return Path(out)


# ---------------------------------------------------------------------------
# selftest: 에피소드 샘플러 불변식 + 합성 이미지로 train -> export -> fewshot.evaluate 계약
# ---------------------------------------------------------------------------
def selftest() -> None:
    import tempfile

    from PIL import Image, ImageDraw

    from .class_features import ATTRIBUTES
    from .fewshot import evaluate, make_splits

    y = np.repeat(np.arange(4), [12, 10, 9, 15])  # 클래스 2 는 9장 < K+Q -> 자격 없음
    ep = _Episodes(y, 95, seed=0)
    assert (ep.n_way, len(ep)) == (3, 3)
    for b in ep:
        blocks = y[b].reshape(3, PROTO_K + PROTO_Q)
        assert len(set(b)) == len(b) and (blocks == blocks[:, :1]).all() and 2 not in blocks

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rows = []
        for c in range(6):
            color = tuple(int(v) for v in rng.integers(80, 256, 3))
            for g in range(6):
                for i in range(3):
                    im = Image.new('RGB', (64, 64), tuple(int(v) for v in rng.integers(0, 60, 3)))
                    x0, y0 = (int(v) for v in rng.integers(2, 12, 2))
                    ImageDraw.Draw(im).rectangle([x0, y0, x0 + 10 + 8 * c, y0 + 40], fill=color)
                    p = tmp / f'c{c}_g{g}_{i}.png'
                    im.save(p)
                    rows.append((str(p), f'c{c}', f'c{c}_g{g}', f'k{c}'))
        df = pd.DataFrame(rows, columns=['path', 'label', 'group', 'class_key'])
        df.to_csv(tmp / 'm.csv', index=False)
        table = pd.DataFrame(np.where(rng.integers(0, 2, (6, len(ATTRIBUTES))) == 1, 'v1', 'v0'), columns=ATTRIBUTES)
        table.insert(0, 'class_key', [f'k{c}' for c in range(6)])
        table.loc[0, 'System'] = 'unknown'
        table.to_csv(tmp / 'features.csv', index=False)
        splits = make_splits(df, n_splits=1, n_unseen=2, min_groups_unseen=1, seed=0)
        (tmp / 's.json').write_text(json.dumps(splits))

        for loss in ('ce', 'protonet', 'arcface_attr'):
            args = ['train', '--manifest', str(tmp / 'm.csv'), '--splits', str(tmp / 's.json'), '--split-id', '0',
                    '--loss', loss, '--model-name', 'vit_tiny_patch16_224', '--epochs', '2', '--epoch-size', '40',
                    '--batch-size', '8', '--num-workers', '0', '--feature-table', str(tmp / 'features.csv'),
                    '--out', str(tmp / loss)]
            train(vars(_parser().parse_args(args)), pretrained=False)
            s = json.loads((tmp / loss / 'summary.json').read_text())
            assert s['best_epoch'] in (1, 2) and s['n_val'] > 0 and s['best_val_acc'] is not None
            assert s['normalize']['mean'] == [0.5] * 3 and not (tmp / loss / 'model.partial.pt').exists()
            if loss == 'protonet':
                assert s['protonet']['n_eligible'] + len(s['protonet']['ineligible']) == s['n_seen']
            items = pd.read_csv(tmp / loss / 'items.csv')
            roles = item_roles(df, splits[0])
            assert set(items['path']) <= set(df.loc[roles.isin(['train', 'val']), 'path'])
            export(tmp / 'm.csv', tmp / f'{loss}.npz', ckpt=tmp / loss / 'model.pt', num_workers=0)
            npz = np.load(tmp / f'{loss}.npz')
            assert npz['path'].tolist() == df['path'].tolist() and npz['emb'].shape == (len(df), EMB_DIM)
            r = evaluate(tmp / f'{loss}.npz', tmp / 'm.csv', splits[0], (1,), episodes=5,
                         feature_table=tmp / 'features.csv', classifiers=('cos',))
            assert r['kshot']['1']['n_episodes'] == 5
            assert ('head' in r['attributes']['image_fill']) == (loss == 'arcface_attr')
            if loss == 'arcface_attr':
                P = npz[f"attrprob__{s['attributes'][0]}"]
                assert np.allclose(P.sum(1), 1, atol=1e-4)
    print('[selftest] OK')


def _parser():
    p = argparse.ArgumentParser('fewshot_train', description='seen 학습과 emb.npz 추출 (spec §6.4)')
    sub = p.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--manifest', required=True)
    t.add_argument('--splits', required=True)
    t.add_argument('--split-id', required=True)
    t.add_argument('--loss', required=True, choices=LOSSES)
    t.add_argument('--model-name', required=True)
    t.add_argument('--margin', type=float, default=0.5)
    t.add_argument('--scale', type=float, default=30.0)
    t.add_argument('--attr-weight', type=float, default=1.0)
    t.add_argument('--feature-table', default=str(DEFAULT_TABLE))
    t.add_argument('--epochs', type=int, default=10)
    t.add_argument('--epoch-size', type=int, default=3000)
    t.add_argument('--batch-size', type=int, default=32)
    t.add_argument('--lr', type=float, default=5e-5)
    t.add_argument('--head-lr', type=float, default=1e-3)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--device', default='auto')
    t.add_argument('--num-workers', type=int, default=4)
    t.add_argument('--val-max-per-class', type=int, default=30, help='선택용 centroid 에 쓰는 클래스당 train 장수')
    t.add_argument('--out', required=True)
    e = sub.add_parser('export')
    e.add_argument('--manifest', required=True)
    src = e.add_mutually_exclusive_group(required=True)
    src.add_argument('--ckpt')
    src.add_argument('--pretrained', help='timm 이름. 동결 백본 pooled feature')
    e.add_argument('--out', required=True)
    e.add_argument('--batch-size', type=int, default=64)
    e.add_argument('--num-workers', type=int, default=4)
    e.add_argument('--device', default='auto')
    sub.add_parser('selftest')
    return p


def main() -> None:
    a = _parser().parse_args()
    if a.cmd == 'train':
        train(vars(a))
    elif a.cmd == 'export':
        export(a.manifest, a.out, a.ckpt, a.pretrained, a.batch_size, a.num_workers, a.device)
    else:
        selftest()


if __name__ == '__main__':
    main()
