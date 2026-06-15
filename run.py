#!/usr/bin/env python3
"""Small Python-only command dispatcher.
Examples:
  python3 run.py install
  python3 run.py check
  python3 run.py supcon7 --data-root /workspace/implant_python_only_final/data/7label --epochs 80
  python3 run.py eval3metric7 --checkpoint /workspace/implant_python_only_final/outputs/<run>/checkpoints/best.pt
  python3 run.py supcon6 --data-root /workspace/data/large_multiclass --epochs 80
  python3 run.py vit7 --data-root /workspace/data/large_multiclass --epochs 80
  python3 run.py vlm5 --data-root /workspace/data/large_multiclass --output data/manifests/vlm_5feature_6label.jsonl --resume
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMMANDS = {
    'install': 'install_requirements.py',
    'check': 'check_environment.py',
    'manifest': 'make_manifest.py',
    'split': 'make_splits.py',
    'supcon7': 'train_vit_7label_supcon.py',
    'eval3metric7': 'evaluate_3label_metric_from_7label_supcon.py',
    'supcon6': 'train_vit_6label.py',
    'eval3metric6': 'evaluate_3label_metric_from_6label.py',
    'vit7': 'train_vit_7label.py',
    'vlm5': 'generate_vlm_features_azure_6label.py',
    'vlm5json': 'generate_vlm_features_azure_6label.py',
    'train': 'train_supervised.py',
    'metric': 'train_metric.py',
    'finetune': 'finetune_small_data.py',
    'eval': 'evaluate.py',
    'attention': 'visualize_attention.py',
    'summary': 'summarize_runs.py',
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print('Usage: python3 run.py <command> [args...]')
        print('Commands:')
        for k in COMMANDS:
            print(' ', k)
        raise SystemExit(2)
    cmd = sys.argv[1]
    script = ROOT / COMMANDS[cmd]
    args = [sys.executable, str(script)] + sys.argv[2:]
    raise SystemExit(subprocess.call(args, cwd=str(ROOT)))


if __name__ == '__main__':
    main()
