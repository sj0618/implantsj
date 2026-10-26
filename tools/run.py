#!/usr/bin/env python3
"""Single Python command dispatcher for implantsj.

`run.sh` (the interactive menu) calls into this file; you can also call it
directly. Every command below is a thin alias for `python3 -m <module>`, so
the underlying pipeline scripts keep their own `--flags`.

  python3 tools/run.py check
  python3 tools/run.py manifest --data-root /workspace/data/7label --out m.csv
  python3 tools/run.py train  --manifest m.csv --model-name resnet50.a1_in1k
  python3 tools/run.py metric --manifest m.csv --model-name vit_base_patch16_224
  python3 tools/run.py maha   --checkpoint best.pt --manifest m.csv --output-dir out/
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 명령 이름 -> 실행할 python 모듈.
COMMANDS: dict[str, str] = {
    # 환경
    'install': 'tools.setup.install_requirements',
    'check': 'tools.setup.check_environment',
    # 데이터 준비
    'manifest': 'pipelines.data.make_manifest',
    'split': 'pipelines.data.make_splits',
    # 폐쇄망 원본(TL1/TS1/VL1/VS1) -> ImageFolder + manifest
    'prepare': 'pipelines.data.prepare_closed_network_dataset',
    # 학습 (모델은 --model-name 으로 교체: vit / resnet / convnext ...)
    'train': 'pipelines.train.train_supervised',
    'metric': 'pipelines.train.train_metric',
    'finetune': 'pipelines.train.finetune_small_data',
    # 평가
    'eval': 'pipelines.eval.evaluate',
    'maha': 'tools.evaluate_unseen_mahalanobis',
    # YOLO 검출 / crop 전처리
    'yolotrain': 'pipelines.yolo.train_yolo',
    'yolocrop': 'pipelines.yolo.crop_yolo',
    'yolotest': 'pipelines.yolo.test_yolo',
    # 시각화 / 리포트
    'attention': 'pipelines.visualize.visualize_attention',
    'tsne': 'pipelines.visualize.visualize_classifier_tsne',
    'summary': 'pipelines.report.summarize_runs',
    'reproduce': 'tools.reproduce_reports',
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help', 'help'):
        print('Usage: python3 tools/run.py <command> [args...]')
        print('Commands:')
        for name, module in COMMANDS.items():
            print(f'  {name:<10} -> {module}')
        raise SystemExit(0 if len(sys.argv) >= 2 else 2)
    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f'Unknown command: {cmd}', file=sys.stderr)
        print('Run "python3 tools/run.py --help" for the list.', file=sys.stderr)
        raise SystemExit(2)
    args = [sys.executable, '-m', COMMANDS[cmd]] + sys.argv[2:]
    raise SystemExit(subprocess.call(args, cwd=str(REPO_ROOT)))


if __name__ == '__main__':
    main()
