# implantsj

Dental implant classification / visualization playground.

## Layout

- `src/` — shared dataset, model, metric, and training utilities
- `pipelines/data/` — manifest and split builders
- `pipelines/train/` — supervised, metric-learning, fine-tune, and ViT training workflows
- `pipelines/eval/` — evaluation, prediction, and transfer-learning checks
- `pipelines/visualize/` — attention overlays and embedding plots
- `pipelines/vlm/` — VLM feature extraction and fusion experiments
- `pipelines/report/` — run summaries
- `tools/` — setup and command dispatch
- `tests/` — smoke tests and runnable regression checks
- `two_stage/` — confusion-group refinement pipeline
- `docs/` — usage notes, analysis, and VLM references

## Current research notes

- `docs/analysis/RESEARCH_PLAN_TSNE_UMAP.md`
- `docs/analysis/PROJECT_ANALYSIS.md`
- `docs/analysis/EXPERIMENT_SUMMARY.md`

## Quick start

See `docs/usage/USAGE.py` for copy-paste command examples.
