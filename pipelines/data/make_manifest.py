#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import build_manifest, print_manifest_summary


def main():
    p = argparse.ArgumentParser(description='Build image manifest CSV from class-folder or split-class-folder data.')
    p.add_argument('--data-root', required=True, help='Example: /workspace/data/large_multiclass')
    p.add_argument('--out', required=True, help='Output CSV path')
    p.add_argument('--layout', default='auto', choices=['auto', 'class_folder', 'split_class_folder'])
    args = p.parse_args()
    build_manifest(args.data_root, args.out, layout=args.layout)
    print_manifest_summary(args.out)


if __name__ == '__main__':
    main()
