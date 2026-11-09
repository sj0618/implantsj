# 42-class FT backbone: Mean prototype × four distances

- Prototype: target train class-wise mean.
- Query: target test.
- Mahalanobis: train-only pooled within-class covariance with Ledoit-Wolf shrinkage.
- This is supervised target-gallery transfer, not zero-shot.
- Mahalanobis implementation matched SciPy's direct formula within 5e-14; the same ranking reproduced on validation.

| Dataset | Distance | Accuracy | Macro-F1 | Majority baseline |
|---|---|---:|---:|---:|
| 12label | mahalanobis | 77.34% | 70.16% | 34.57% |
| 12label | cosine | 36.72% | 36.08% | 34.57% |
| 12label | euclidean | 35.16% | 35.82% | 34.57% |
| 12label | manhattan | 33.79% | 34.60% | 34.57% |
| 3label | mahalanobis | 98.43% | 95.65% | 86.10% |
| 3label | cosine | 91.70% | 76.93% | 86.10% |
| 3label | manhattan | 90.13% | 73.57% | 86.10% |
| 3label | euclidean | 89.46% | 73.52% | 86.10% |
| 7label | mahalanobis | 61.70% | 26.89% | 51.77% |
| 7label | manhattan | 17.73% | 13.33% | 51.77% |
| 7label | euclidean | 18.44% | 13.23% | 51.77% |
| 7label | cosine | 15.60% | 11.55% | 51.77% |

## Warnings

- 3label contains 5 exact train↔test duplicate hashes.
- 7label test contains 6/7 gallery labels; nobel has zero test support.
- The local dataset is 12label; no class was removed to manufacture an 11label result.
- The internal 42-class training manifest is unavailable, so training-to-target overlap cannot be excluded.

## Mahalanobis validation sanity check

| Dataset | Accuracy | Macro-F1 |
|---|---:|---:|
| 3label | 97.53% | 89.84% |
| 7label | 67.13% | 36.77% |
| 12label | 75.39% | 68.37% |
