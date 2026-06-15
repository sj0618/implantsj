#!/usr/bin/env python3
"""
NO-VENV installer for RunPod Web Terminal.

Usage:
  python3 install_requirements.py
  python3 install_requirements.py --install-torch cu126

This script installs packages into the current container Python. It does not create or use a virtual environment.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

BASE_PACKAGES = [
    'numpy',
    'pandas',
    'scikit-learn',
    'matplotlib',
    'umap-learn',
    'Pillow',
    'PyYAML',
    'tqdm',
    'timm',
    'openai',
    'python-dotenv',
]


def run(cmd):
    print('+', ' '.join(cmd), flush=True)
    return subprocess.run(cmd, check=False)


def pip_install(args):
    cmd = [sys.executable, '-m', 'pip', 'install'] + args
    p = run(cmd)
    if p.returncode == 0:
        return
    # Debian/Ubuntu PEP 668 fallback.
    cmd = [sys.executable, '-m', 'pip', 'install', '--break-system-packages'] + args
    p = run(cmd)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--install-torch', choices=['none', 'cu126', 'cu121', 'cpu'], default='none', help='Do not install torch by default. Use only if torch import is broken.')
    parser.add_argument('--upgrade', action='store_true', default=True)
    args = parser.parse_args()

    pip_install(['--upgrade', 'pip', 'setuptools', 'wheel'])

    if args.install_torch != 'none':
        run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torch', 'torchvision', 'torchaudio'])
        if args.install_torch == 'cpu':
            pip_install(['torch', 'torchvision'])
        else:
            pip_install(['torch', 'torchvision', '--index-url', f'https://download.pytorch.org/whl/{args.install_torch}'])

    pip_install(['--upgrade'] + BASE_PACKAGES)

    print('\nChecking imports...')
    import importlib
    mods = ['torch', 'torchvision', 'timm', 'numpy', 'pandas', 'sklearn', 'matplotlib', 'PIL', 'yaml', 'tqdm']
    failed = []
    for m in mods:
        try:
            mod = importlib.import_module(m)
            print('OK', m, getattr(mod, '__version__', ''))
        except Exception as e:
            print('FAIL', m, repr(e))
            failed.append((m, e))
    if failed:
        print('\nSome imports failed. If torch/torchvision failed, rerun with:')
        print('  python3 install_requirements.py --install-torch cu126')
        raise SystemExit(1)
    print('INSTALL_OK')


if __name__ == '__main__':
    main()
