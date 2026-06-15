from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def add_project_root_to_path(file: str) -> Path:
    root = Path(file).resolve().parent
    if root.name == 'src':
        root = root.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def make_run_dir(output_root: str | Path, experiment_name: str) -> Path:
    root = Path(output_root)
    run_dir = root / f'{timestamp()}_{experiment_name}'
    for sub in ['checkpoints', 'logs', 'metrics', 'confusion_matrices', 'attention_maps', 'embeddings']:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)


class TeeLogger:
    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.log_path.open('a', encoding='utf-8')

    def write(self, text: str) -> None:
        sys.__stdout__.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self) -> None:
        sys.__stdout__.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def log(msg: str) -> None:
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)


def require_torch_cuda(require_cuda: bool = True):
    import torch
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available. Use a GPU RunPod or pass --allow-cpu.')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None
