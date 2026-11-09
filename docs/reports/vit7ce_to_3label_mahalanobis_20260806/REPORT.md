# 7-label CE → 3-label Mean prototype distance comparison

## Result

| Distance | Test Accuracy | Test Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|
| Euclidean | 73.09% | 48.74% | 59.35% |
| Manhattan | 73.32% | 48.92% | 59.43% |
| Cosine | 73.32% | 48.83% | 59.43% |
| Mahalanobis | 95.07% | 84.19% | 87.72% |

## Validation sanity check

| Distance | Validation Accuracy | Validation Macro-F1 |
|---|---:|---:|
| Euclidean | 72.42% | 53.88% |
| Manhattan | 73.32% | 54.75% |
| Cosine | 74.89% | 56.31% |
| Mahalanobis | 95.29% | 85.99% |

## Protocol

- Source: cropped 7-label train/valid, ImageNet-pretrained ViT-Base, CE fine-tuning for 30 epochs, seed 2026.
- Selected checkpoint: epoch 11 (minimum source validation loss 0.43493).
- Target: cropped Bego/Bicon/ITI. Target-train Mean prototypes; target-train pooled within-class Ledoit-Wolf covariance; target-test query.
- Mahalanobis implementation check versus SciPy: maximum absolute error 1.137e-13.

## Exact-duplicate-free test sensitivity

| Distance | N | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|---:|
| Euclidean | 441 | 72.79% | 48.66% | 59.24% |
| Manhattan | 441 | 73.02% | 48.85% | 59.33% |
| Cosine | 441 | 73.02% | 48.76% | 59.33% |
| Mahalanobis | 441 | 95.01% | 84.18% | 87.71% |

- Removed 5 test images whose SHA-256 exactly matched a target-train image.

## Important interpretation

- This is not zero-shot: target train labels are used to fit the prototypes and covariance.
- The original target split contains five exact train/test image duplicates; the sensitivity result above removes all five.
- Historical 7→3 results (34.86%/29.26%, 35.78%/29.88%, 37.61%/34.75%) used different saved runs/protocols; their checkpoints are unavailable locally, so this is a fresh comparison, not an exact rerun.
