from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn


def create_classifier(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.0,
    drop_path_rate: float = 0.0,
):
    import timm
    kwargs = {'pretrained': pretrained, 'num_classes': num_classes}
    if dropout is not None:
        kwargs['drop_rate'] = dropout
    if drop_path_rate is not None:
        kwargs['drop_path_rate'] = drop_path_rate
    return timm.create_model(model_name, **kwargs)


class MetricModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        embedding_dim: int = 128,
        pretrained: bool = True,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=dropout,
            drop_path_rate=drop_path_rate,
        )
        in_dim = getattr(self.backbone, 'num_features', None)
        if in_dim is None:
            raise RuntimeError(f'Could not infer num_features for {model_name}')
        self.projector = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, embedding_dim),
        )

    def forward_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        z = self.projector(self.forward_features(x))
        return nn.functional.normalize(z, dim=1)


def unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for key in ['model_state_dict', 'state_dict', 'model']:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def load_matching_weights(model: nn.Module, checkpoint_path: str | Path, strict: bool = False) -> Tuple[int, int]:
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = unwrap_state_dict(ckpt)
    model_state = model.state_dict()
    filtered = {}
    skipped = 0
    for k, v in state.items():
        kk = k
        if kk.startswith('module.'):
            kk = kk[len('module.'):]
        if kk.startswith('model.'):
            kk = kk[len('model.'):]
        if kk in model_state and tuple(model_state[kk].shape) == tuple(v.shape):
            filtered[kk] = v
        else:
            # metric checkpoints may store backbone.xxx while classifier expects xxx
            if kk.startswith('backbone.'):
                alt = kk[len('backbone.'):]
                if alt in model_state and tuple(model_state[alt].shape) == tuple(v.shape):
                    filtered[alt] = v
                    continue
            skipped += 1
    model.load_state_dict(filtered, strict=False)
    return len(filtered), skipped


def save_checkpoint(path: str | Path, model, optimizer, epoch: int, metrics: Dict, args: Dict, class_to_idx: Dict[str, int]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'metrics': metrics,
        'args': args,
        'class_to_idx': class_to_idx,
    }, path)


def load_checkpoint(path: str | Path):
    return torch.load(path, map_location='cpu')


def get_class_to_idx_from_checkpoint(path: str | Path) -> Dict[str, int]:
    ckpt = torch.load(path, map_location='cpu')
    if 'class_to_idx' not in ckpt:
        raise KeyError('Checkpoint has no class_to_idx.')
    return {str(k): int(v) for k, v in ckpt['class_to_idx'].items()}
