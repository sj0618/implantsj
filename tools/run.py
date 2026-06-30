#!/usr/bin/env python3
"""Small Python-only command dispatcher.
Examples:
  python3 tools/run.py install
  python3 tools/run.py check
  python3 tools/run.py supcon7 --data-root /workspace/implant_python_only_final/data/7label --epochs 80
  python3 tools/run.py eval3metric7 --checkpoint /workspace/implant_python_only_final/outputs/<run>/checkpoints/best.pt
  python3 tools/run.py supcon6 --data-root /workspace/data/large_multiclass --epochs 80
  python3 tools/run.py vit7 --data-root /workspace/data/large_multiclass --epochs 80
  python3 tools/run.py vlm5 --data-root /workspace/data/large_multiclass --output data/manifests/vlm_5feature_6label.jsonl --resume
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMANDS: dict[str, str] = {
    'install': 'tools.setup.install_requirements',
    'check': 'tools.setup.check_environment',
    'manifest': 'pipelines.data.make_manifest',
    'split': 'pipelines.data.make_splits',
    'supcon7': 'pipelines.train.train_vit_7label_supcon',
    'eval3metric7': 'pipelines.eval.evaluate_3label_metric_from_7label_supcon',
    'supcon6': 'pipelines.train.train_vit_6label',
    'eval3metric6': 'pipelines.eval.evaluate_3label_metric_from_6label',
    'vit7': 'pipelines.train.train_vit_7label',
    'vlm5': 'pipelines.vlm.generate_vlm_features_azure_6label',
    'vlm5json': 'pipelines.vlm.generate_vlm_features_azure_6label',
    'train': 'pipelines.train.train_supervised',
    'metric': 'pipelines.train.train_metric',
    'finetune': 'pipelines.train.finetune_small_data',
    'eval': 'pipelines.eval.evaluate',
    'attention': 'pipelines.visualize.visualize_attention',
    'summary': 'pipelines.report.summarize_runs',
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print('Usage: python3 tools/run.py <command> [args...]')
        print('Commands:')
        for k in COMMANDS:
            print(' ', k)
        raise SystemExit(2)
    cmd = sys.argv[1]
    module = COMMANDS[cmd]
    args = [sys.executable, '-m', module] + sys.argv[2:]
    raise SystemExit(subprocess.call(args, cwd=str(REPO_ROOT)))


if __name__ == '__main__':
    main()
