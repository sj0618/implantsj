# Unseen 7→3 Mahalanobis 개선 실험

## 결과 요약

Encoder는 다시 학습하지 않았다. 거리 모델은 target-train으로만 적합했고, 방법은 target-valid Macro-F1으로 골랐다. Test는 선택에 사용하지 않았다. 값은 `Accuracy / Macro-F1 / Balanced Accuracy`(%)다.

| 조건 | 기존 pooled Mahalanobis | Validation 선택 개선법 | 개선 후 test | Macro-F1 변화 |
|---|---:|---|---:|---:|
| Fresh SupCon backbone 384-D, test 446 | 98.21 / 92.94 / 91.25 | RDA, class covariance 75% + logdet | **99.55 / 98.21 / 97.53** | **+5.27%p** |
| Fresh SupCon projection 128-D, test 446 | 93.27 / 81.46 / 86.46 | Class-specific covariance + logdet | **97.76 / 93.30 / 92.81** | **+11.84%p** |
| Legacy SupCon backbone 384-D, test 109 | 63.30 / 48.23 / 50.77 | Class-balanced PCA 256-D + pooled LW | **64.22 / 51.57 / 58.89** | **+3.34%p** |
| Legacy SupCon projection 128-D, test 109 | 60.55 / 44.77 / 47.65 | RDA, class covariance 75% + logdet | **71.56 / 54.31 / 56.53** | **+9.53%p** |

Fresh split의 exact train-test 중복 5장을 제외해도 backbone 99.55 / 98.20 / 97.53, projection 97.73 / 93.30 / 92.81로 유지됐다. Legacy split에는 exact 중복이 없다.

## 20%대 과거 조건과의 연결

| Embedding | Cosine | 기존 pooled Mahalanobis | 개선 Mahalanobis |
|---|---:|---:|---:|
| Legacy backbone | 27.52 / 26.87 | 63.30 / 48.23 | **64.22 / 51.57** |
| Legacy projection | 31.19 / 29.75 | 60.55 / 44.77 | **71.56 / 54.31** |

과거 20%대 unseen 결과도 같은 checkpoint에서 개선됐다. Projection은 기존 Mahalanobis보다 Accuracy +11.01%p, Macro-F1 +9.53%p, Balanced Accuracy +8.88%p 높았다.

## 개선 방식

### Regularized Discriminant Mahalanobis (RDA)

기존 방식은 모든 class가 하나의 pooled covariance를 공유했다. 개선 방식은 target-train에서 class별 covariance를 추정하고 pooled covariance와 혼합한다.

`Σ_k = (1-λ)Σ_pooled + λΣ_class,k`

판정 점수는 class별 Mahalanobis 제곱거리와 covariance 부피 보정값을 합친 `d_k² + log|Σ_k|`다. Class prior는 넣지 않아 Bicon 다수 비율을 그대로 강화하지 않았다. `λ`는 target validation에서만 선택했다.

### Class-balanced PCA + pooled Mahalanobis

Legacy backbone에서는 Bicon이 PCA 방향을 지배하지 않도록 target-train 각 class에서 같은 수의 표본을 뽑아 PCA를 적합했다. 384-D를 256-D로 줄인 뒤 기존 pooled Ledoit-Wolf Mahalanobis를 적용했다. Test Balanced Accuracy는 50.77에서 58.89%로 올랐다.

## Fresh backbone 클래스별 결과

| Class | 기존 Recall / F1 | 개선 Recall / F1 |
|---|---:|---:|
| Bego | 88.57 / 92.54 | **100.00 / 98.59** |
| Bicon | 100.00 / 99.48 | **100.00 / 99.87** |
| ITI | 85.19 / 86.79 | **92.59 / 96.15** |

개선 후 test 446장 중 오분류는 2장이다: ITI 1장이 Bego, ITI 1장이 Bicon으로 분류됐다.

## 실험 계약과 한계

- Encoder 학습: source 7-label만 사용.
- Distance fit: target-train label과 embedding만 사용.
- Method selection: target-validation Macro-F1, 이후 Balanced Accuracy와 Accuracy로 tie-break.
- Target test는 선택과 적합에 사용하지 않음.
- 후보 수: backbone 28개, projection 24개.
- 이 결과는 unseen-label target-gallery transfer이며 target label을 사용하므로 strict zero-shot이 아니다.
- Validation 한 split에서 여러 후보를 비교했으므로 별도 외부/group-unseen test에서 재검증하기 전에는 99.55%를 일반 성능으로 단정하지 않는다.
- Exact duplicate 제거는 수행했지만 patient/source-group 및 near-duplicate 독립성까지 증명하지는 않는다.

## 산출물

- `fresh_backbone/`, `fresh_projection/`: 최신 cropped 446-test 결과.
- `legacy_backbone/`, `legacy_projection/`: 과거 20%대 protocol의 109-test 재현 결과.
- 각 폴더의 `validation_candidates.csv`: validation 후보 순위.
- `test_selected_vs_baseline.csv`: test에서 baseline과 validation-selected 방법만 비교.
- `test_per_class.csv`, `confusion_test_selected.csv`, `test_predictions_selected.csv`.
- `protocol.json`: checkpoint·manifest hash, 선택 계약, 중복 제외 수.

재실행 도구: `tools/improve_unseen_mahalanobis.py`.
