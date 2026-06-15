#!/usr/bin/env python3
"""
Python-only, no-venv RunPod experiment project.

Install packages into the RunPod container Python:
  cd /workspace/implant_python_only_final
  python3 install_requirements.py
  python3 check_environment.py

Train 7-label -> 7-label ViT high-score baseline:
  python3 train_vit_7label.py \
    --data-root /workspace/data/large_multiclass \
    --manifest /workspace/data/manifests/large_multiclass.csv \
    --model-name vit_base_patch16_224 \
    --epochs 80 \
    --batch-size 32

Train 7-label SupCon ViT metric model:
  python3 train_vit_7label_supcon.py \
    --data-root /workspace/implant_python_only_final/data/7label \
    --manifest /workspace/implant_python_only_final/data/manifests/supcon_vit_7label.csv \
    --model-name vit_base_patch16_224 \
    --epochs 80 \
    --batch-size 16

Evaluate 7-label SupCon checkpoint on the 3-label metric prototype task:
  python3 evaluate_3label_metric_from_7label_supcon.py \
    --checkpoint /workspace/implant_python_only_final/outputs/<run>/checkpoints/best.pt \
    --data-root /workspace/implant_python_only_final/data/3label \
    --manifest /workspace/implant_python_only_final/data/manifests/3label.csv \
    --batch-size 32

Smoke test the 7-label SupCon -> 3-label metric evaluator:
  python3 test_7label_supcon_to_3label_metric.py

Train 6-label SupCon ViT metric model:
  python3 train_vit_6label.py \
    --data-root /workspace/data/large_multiclass \
    --manifest /workspace/data/manifests/large_multiclass_supcon_6label.csv \
    --model-name vit_base_patch16_224 \
    --epochs 80 \
    --batch-size 16 \
    --rebuild-manifest

Evaluate that SupCon checkpoint on the 3-label metric prototype task:
  python3 evaluate_3label_metric_from_6label.py \
    --checkpoint /workspace/implant_outputs/<run>/checkpoints/best.pt \
    --data-root /workspace/implant_python_only_final/data/3label \
    --manifest /workspace/implant_python_only_final/data/manifests/3label.csv \
    --batch-size 32

Evaluate best checkpoint:
  python3 evaluate.py \
    --checkpoint /workspace/implant_outputs/<run>/checkpoints/best.pt \
    --manifest /workspace/data/manifests/large_multiclass.csv \
    --split test

Visualize saliency maps:
  python3 visualize_attention.py \
    --checkpoint /workspace/implant_outputs/<run>/checkpoints/best.pt \
    --manifest /workspace/data/manifests/large_multiclass.csv \
    --split test \
    --max-samples 64

Visualize classifier t-SNE:
  python3 visualize_classifier_tsne.py \
    --checkpoint /workspace/implant_outputs/<run>/checkpoints/best.pt \
    --manifest /workspace/data/manifests/large_multiclass.csv \
    --split all

Visualize metric t-SNE:
  python3 visualize_metric_tsne.py \
    --checkpoint /workspace/implant_outputs/<run>/checkpoints/best.pt \
    --manifest /workspace/data/manifests/large_multiclass.csv \
    --split all

Generate UMAP beside existing t-SNE artifacts:
  python3 visualize_umap_from_tsne.py \
    --outputs-root /workspace/implant_python_only_final/outputs \
    --overwrite

No virtualenv is created or required. All scripts run with python3 directly.
"""

print(__doc__)
