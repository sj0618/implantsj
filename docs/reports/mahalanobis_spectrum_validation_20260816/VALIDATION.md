# Mahalanobis spectrum validation

## Scope

This report evaluates four fixed experimental scorers against pooled Ledoit-Wolf on cached embeddings. The candidates were proposed after these caches had already been inspected, so every result below is exploratory post-selection evidence. It is not a fresh confirmatory estimate and it does not change automatic method selection.

The 3-label conditions use ten train supports per class for both prototypes and covariance, with the complete test split as query. Fresh-cache duplicate-free rows exclude exact train/test content duplicates and are shown beside the all-row results. The strict condition uses the stored five-support-per-class episodes, 25 supports and 127 queries for each seed.

## Fresh 3-label results

| condition | evaluation_view | method | accuracy | delta_accuracy | macro_f1 | delta_macro_f1 |
|---|---|---|---|---|---|---|
| fresh_backbone | all_rows | pooled_ledoit_wolf | 69.36% | 0.00% | 52.93% | 0.00% |
| fresh_backbone | all_rows | fractional_0.5 | 70.33% | 0.97% | 53.60% | 0.67% |
| fresh_backbone | all_rows | fractional_0.75 | 69.43% | 0.07% | 52.87% | -0.06% |
| fresh_backbone | all_rows | centered_angular_0.5 | 74.22% | 4.86% | 55.77% | 2.84% |
| fresh_backbone | all_rows | centered_angular_0.75 | 73.65% | 4.30% | 55.22% | 2.29% |
| fresh_backbone | duplicate_free | pooled_ledoit_wolf | 69.43% | 0.00% | 53.13% | 0.00% |
| fresh_backbone | duplicate_free | fractional_0.5 | 70.41% | 0.98% | 53.81% | 0.68% |
| fresh_backbone | duplicate_free | fractional_0.75 | 69.46% | 0.04% | 53.04% | -0.09% |
| fresh_backbone | duplicate_free | centered_angular_0.5 | 74.23% | 4.80% | 55.92% | 2.79% |
| fresh_backbone | duplicate_free | centered_angular_0.75 | 73.66% | 4.23% | 55.35% | 2.22% |
| fresh_projection | all_rows | pooled_ledoit_wolf | 66.18% | 0.00% | 49.13% | 0.00% |
| fresh_projection | all_rows | fractional_0.5 | 66.85% | 0.67% | 49.61% | 0.48% |
| fresh_projection | all_rows | fractional_0.75 | 66.63% | 0.45% | 49.52% | 0.39% |
| fresh_projection | all_rows | centered_angular_0.5 | 69.92% | 3.74% | 51.40% | 2.27% |
| fresh_projection | all_rows | centered_angular_0.75 | 69.66% | 3.48% | 51.47% | 2.33% |
| fresh_projection | duplicate_free | pooled_ledoit_wolf | 66.36% | 0.00% | 49.40% | 0.00% |
| fresh_projection | duplicate_free | fractional_0.5 | 66.93% | 0.57% | 49.81% | 0.41% |
| fresh_projection | duplicate_free | fractional_0.75 | 66.74% | 0.38% | 49.75% | 0.35% |
| fresh_projection | duplicate_free | centered_angular_0.5 | 69.99% | 3.63% | 51.60% | 2.20% |
| fresh_projection | duplicate_free | centered_angular_0.75 | 69.73% | 3.36% | 51.66% | 2.27% |

## Legacy 3-label results

| condition | evaluation_view | method | accuracy | delta_accuracy | macro_f1 | delta_macro_f1 |
|---|---|---|---|---|---|---|
| legacy_backbone | all_rows | pooled_ledoit_wolf | 38.99% | 0.00% | 31.16% | 0.00% |
| legacy_backbone | all_rows | fractional_0.5 | 36.85% | -2.14% | 30.47% | -0.68% |
| legacy_backbone | all_rows | fractional_0.75 | 38.07% | -0.92% | 30.58% | -0.57% |
| legacy_backbone | all_rows | centered_angular_0.5 | 35.47% | -3.52% | 29.27% | -1.89% |
| legacy_backbone | all_rows | centered_angular_0.75 | 37.46% | -1.53% | 29.92% | -1.24% |
| legacy_backbone | duplicate_free | pooled_ledoit_wolf | 38.99% | 0.00% | 31.16% | 0.00% |
| legacy_backbone | duplicate_free | fractional_0.5 | 36.85% | -2.14% | 30.47% | -0.68% |
| legacy_backbone | duplicate_free | fractional_0.75 | 38.07% | -0.92% | 30.58% | -0.57% |
| legacy_backbone | duplicate_free | centered_angular_0.5 | 35.47% | -3.52% | 29.27% | -1.89% |
| legacy_backbone | duplicate_free | centered_angular_0.75 | 37.46% | -1.53% | 29.92% | -1.24% |
| legacy_projection | all_rows | pooled_ledoit_wolf | 39.30% | 0.00% | 31.21% | 0.00% |
| legacy_projection | all_rows | fractional_0.5 | 35.17% | -4.13% | 28.06% | -3.15% |
| legacy_projection | all_rows | fractional_0.75 | 37.16% | -2.14% | 29.09% | -2.12% |
| legacy_projection | all_rows | centered_angular_0.5 | 34.10% | -5.20% | 27.59% | -3.62% |
| legacy_projection | all_rows | centered_angular_0.75 | 37.77% | -1.53% | 29.61% | -1.60% |
| legacy_projection | duplicate_free | pooled_ledoit_wolf | 39.30% | 0.00% | 31.21% | 0.00% |
| legacy_projection | duplicate_free | fractional_0.5 | 35.17% | -4.13% | 28.06% | -3.15% |
| legacy_projection | duplicate_free | fractional_0.75 | 37.16% | -2.14% | 29.09% | -2.12% |
| legacy_projection | duplicate_free | centered_angular_0.5 | 34.10% | -5.20% | 27.59% | -3.62% |
| legacy_projection | duplicate_free | centered_angular_0.75 | 37.77% | -1.53% | 29.61% | -1.60% |

## Strict 5-label 5-shot results

| condition | evaluation_view | method | accuracy | delta_accuracy | macro_f1 | delta_macro_f1 |
|---|---|---|---|---|---|---|
| strict_5label_5shot | all_rows | pooled_ledoit_wolf | 67.98% | 0.00% | 63.30% | 0.00% |
| strict_5label_5shot | all_rows | fractional_0.5 | 66.27% | -1.71% | 61.95% | -1.35% |
| strict_5label_5shot | all_rows | fractional_0.75 | 67.59% | -0.39% | 62.99% | -0.31% |
| strict_5label_5shot | all_rows | centered_angular_0.5 | 66.67% | -1.31% | 62.61% | -0.69% |
| strict_5label_5shot | all_rows | centered_angular_0.75 | 66.40% | -1.57% | 61.90% | -1.39% |

## 67.06% statement

The captured source wording is: "the best angular candidate reaches 67.06% accuracy." The source does not define its aggregation formula. For this validation only, the operational interpretation is the larger of the two per-method six-seed mean strict accuracies. The observed value is 66.67%, which differs from 67.06% by -0.39 percentage points. This is not the mean of per-seed maxima.

## Reproducibility boundary

All rows come from the production scorer factory with experimental enumeration explicitly enabled. Existing callers keep experimental enumeration disabled. The machine-readable evidence records cache, protocol, manifest, membership, source, and artifact hashes. A new locked dataset or nested development protocol is still required for confirmatory selection.
