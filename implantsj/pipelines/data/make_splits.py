#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import make_stratified_splits, print_manifest_summary


def main():
    p = argparse.ArgumentParser(description='Create stratified train/valid/test split in a manifest CSV.')
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--train', type=float, default=8)
    p.add_argument('--valid', type=float, default=1)
    p.add_argument('--test', type=float, default=1)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    make_stratified_splits(args.manifest, args.out, train=args.train, valid=args.valid, test=args.test, seed=args.seed)
    print_manifest_summary(args.out)


if __name__ == '__main__':
    main()
