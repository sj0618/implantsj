#!/usr/bin/env python3
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    print('Python:', sys.version)
    print('Executable:', sys.executable)
    print('CWD:', Path.cwd())
    print('Project root:', ROOT)
    print('PYTHONPATH:', os.environ.get('PYTHONPATH', ''))
    mods = ['torch', 'torchvision', 'timm', 'numpy', 'pandas', 'sklearn', 'matplotlib', 'PIL', 'yaml', 'tqdm', 'src']
    failed = []
    for name in mods:
        try:
            mod = importlib.import_module(name)
            print('OK', f'{name:<12}', getattr(mod, '__version__', ''))
        except Exception as e:
            failed.append((name, repr(e)))
            print('FAIL', f'{name:<12}', repr(e))
    try:
        import torch
        print('cuda_available:', torch.cuda.is_available())
        if torch.cuda.is_available():
            print('cuda_device_count:', torch.cuda.device_count())
            print('cuda_device_0:', torch.cuda.get_device_name(0))
    except Exception:
        pass
    if failed:
        raise SystemExit(1)
    print('ENV_OK')


if __name__ == '__main__':
    main()
