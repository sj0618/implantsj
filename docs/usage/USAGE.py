#!/usr/bin/env python3

USAGE_TEXT: str = """Canonical command examples for the reorganized `implantsj` layout.

Run dispatcher:
  python3 tools/run.py install
  python3 tools/run.py check
  python3 tools/run.py manifest --help
  python3 tools/run.py split --help
  python3 tools/run.py metric --help
  python3 tools/run.py maha --help
  python3 tools/run.py train --help
  python3 tools/run.py eval --help
  python3 tools/run.py attention --help
  python3 tools/run.py summary --help

Direct module execution also works from the repo root:
  python3 -m pipelines.train.train_vit_7label --help
  python3 -m pipelines.train.train_vit_7label_supcon --help
  python3 -m pipelines.eval.evaluate_3label_metric_from_7label --help
  python3 -m pipelines.visualize.visualize_attention --help
  python3 -m pipelines.vlm.generate_vlm_features_azure_6label --help

Shared code and packages:
  src/          shared utilities and model code
  two_stage/    confusion-group refinement pipeline
  tests/        smoke tests and runnable regression checks
"""

print(USAGE_TEXT)
