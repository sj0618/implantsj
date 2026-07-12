# Metric Learning Diagnosis — Current Results Report

Snapshot: 2026-07-12 13:10 KST

Scope: 현재 생성된 학습 기록과 곡선만 정리한 중간 보고서

Status: **최종 metric-learning 진단 결과가 아님**

## 1. Executive summary

- **ViT-ClosedFT**는 지정한 10 epoch 학습을 완료했다. 최종 test accuracy는 **0.9543**, test macro-F1은 **0.9267**이다.
- **ViT-ClosedFT-SupCon**은 기록상 3/10 epoch까지만 완료됐다. 최신 test accuracy는 **0.8765**, test macro-F1은 **0.7930**이다. 10 epoch 결과가 아니므로 ClosedFT와의 성능 차이를 SupCon의 효과로 해석하면 안 된다.
- 재사용한 **ViT-FullFT** 기록은 54 epoch이며 per-epoch test metric이 없다. 최고 validation macro-F1은 epoch 42의 **0.9106**, 최고 validation accuracy는 epoch 54의 **0.9514**이다.
- 현재 기록만으로는 open/closed representation, SupCon의 최종 효과, Euclidean 대 cosine, GT/Mean/SS prototype 우열을 판단할 수 없다. 해당 항목은 공통 평가가 끝난 뒤 보고한다.

## 2. Training status

| Model | Labels | Training status | Best/latest epoch | Validation accuracy | Validation macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---|---:|---:|---:|---:|---:|
| ViT Base | 42 | pretrained reference; curve 없음 | — | — | — | — | — |
| ViT-ClosedFT | 27 closed | **complete 10/10** | 10 | **0.9494** | **0.9104** | **0.9543** | **0.9267** |
| ViT-ClosedFT-SupCon | 27 closed | **partial 3/10 recorded** | 3 | 0.8734 | 0.7919 | 0.8765 | 0.7930 |
| ViT-FullFT | 42 full | existing checkpoint/history reused | 42 F1 / 54 acc | **0.9514** | **0.9106** | not recorded | not recorded |
| ViT-FullFT-SupCon | 42 full | 결과 없음 | — | — | — | — | — |

## 3. Learning curves

각 그림의 loss, accuracy, macro-F1 패널은 서로 다른 metric 범위가 잘 보이도록 **독립적인 y축**을 사용한다.

### ViT-ClosedFT — complete 10 epochs

![ViT-ClosedFT learning curves](../../../outputs/metricdiag-viz/vit-closedft.png)

| Epoch | Train loss | Valid acc | Valid macro-F1 | Test acc | Test macro-F1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0761 | 0.8980 | 0.8257 | 0.8918 | 0.8279 |
| 2 | 0.8836 | 0.9084 | 0.8433 | 0.9120 | 0.8480 |
| 3 | 0.8388 | 0.9098 | 0.8487 | 0.9109 | 0.8483 |
| 4 | 0.8039 | 0.9230 | 0.8742 | 0.9308 | 0.8834 |
| 5 | 0.7743 | 0.9318 | 0.8841 | 0.9350 | 0.8896 |
| 6 | 0.7448 | 0.9291 | 0.8805 | 0.9369 | 0.8951 |
| 7 | 0.7179 | 0.9427 | 0.9018 | 0.9459 | 0.9095 |
| 8 | 0.6942 | 0.9447 | 0.9043 | 0.9512 | 0.9161 |
| 9 | 0.6756 | 0.9485 | 0.9083 | 0.9531 | 0.9213 |
| 10 | **0.6656** | **0.9494** | **0.9104** | **0.9543** | **0.9267** |

Epoch 1→10 변화:

- train loss: 1.0761 → 0.6656 (**−0.4105**)
- validation accuracy: 0.8980 → 0.9494 (**+5.14%p**)
- validation macro-F1: 0.8257 → 0.9104 (**+8.48%p**)
- test accuracy: 0.8918 → 0.9543 (**+6.25%p**)
- test macro-F1: 0.8279 → 0.9267 (**+9.89%p**)

### ViT-ClosedFT-SupCon — partial 3/10 epochs

![ViT-ClosedFT-SupCon partial learning curves](../../../outputs/metricdiag-viz/vit-closedft-supcon.png)

| Epoch | Train loss | Valid acc | Valid macro-F1 | Test acc | Test macro-F1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.4605 | 0.8510 | 0.7568 | 0.8513 | 0.7595 |
| 2 | 1.3177 | 0.8732 | 0.7795 | **0.8791** | 0.7918 |
| 3 | **1.2811** | **0.8734** | **0.7919** | 0.8765 | **0.7930** |

Epoch 1→3 변화:

- train loss: 1.4605 → 1.2811 (**−0.1794**)
- validation accuracy: 0.8510 → 0.8734 (**+2.24%p**)
- validation macro-F1: 0.7568 → 0.7919 (**+3.50%p**)
- test accuracy: 0.8513 → 0.8765 (**+2.53%p**)
- test macro-F1: 0.7595 → 0.7930 (**+3.35%p**)

### ViT-FullFT — reused existing 54-epoch run

![ViT-FullFT validation learning curves](../../../outputs/metricdiag-viz/vit-fullft.png)

| Selection | Epoch | Train loss | Valid accuracy | Valid macro-F1 |
|---|---:|---:|---:|---:|
| Initial | 1 | 1.1785 | 0.8899 | 0.7105 |
| Best validation macro-F1 | 42 | 0.7116 | 0.9444 | **0.9106** |
| Best/final validation accuracy | 54 | 0.6948 | **0.9514** | 0.8995 |

이 run의 `history.csv`에는 test accuracy와 test macro-F1 열이 없다. 따라서 FullFT의 test 성능이나 open/closed 분리 성능은 공통 embedding 평가 전에는 보고하지 않는다.

## 4. Current comparisons

### ClosedFT final vs ClosedFT-SupCon partial

| Metric | ClosedFT epoch 10 | SupCon epoch 3 | Difference (SupCon − ClosedFT) |
|---|---:|---:|---:|
| Validation accuracy | 0.9494 | 0.8734 | −7.60%p |
| Validation macro-F1 | 0.9104 | 0.7919 | −11.86%p |
| Test accuracy | 0.9543 | 0.8765 | −7.77%p |
| Test macro-F1 | 0.9267 | 0.7930 | −13.38%p |

이 표는 **10 epoch 대 3 epoch의 불균형 비교**다. SupCon이 나쁘다는 결론의 근거로 사용하지 않는다.

### ClosedFT vs FullFT validation-only reference

| Metric | ClosedFT best | FullFT best | Difference (FullFT − ClosedFT) |
|---|---:|---:|---:|
| Validation accuracy | 0.9494 | 0.9514 | +0.19%p |
| Validation macro-F1 | 0.9104 | 0.9106 | +0.02%p |

label 수(27 vs 42), epoch 수(10 vs 54), split 및 per-epoch test 기록이 다르므로 이 수치도 representation 우열의 최종 근거가 아니다.

## 5. Experiment questions — current answerability

| Question | Current status | Required evidence |
|---|---|---|
| Base 대비 ClosedFT가 representation을 저하시키는가? | **미판정** | 동일 manifest·transform의 Base/ClosedFT embedding 평가 |
| SupCon이 representation에 도움이 되는가? | **미판정** | ClosedFT-SupCon 및 FullFT-SupCon 10 epoch 완료 후 backbone/projector 동시 비교 |
| Full-set 학습이 도움이 되는가? | **미판정** | ClosedFT/FullFT의 open·closed test 분리 평가 |
| Euclidean과 cosine 중 무엇이 좋은가? | **미판정** | validation에서 distance 선택 후 held-out test 보고 |
| GT/Mean/SS prototype 중 무엇이 좋은가? | **미판정** | 동일 embedding에서 3 prototype 정책의 validation/test 결과 |

## 6. Missing report artifacts

- ViT-FullFT-SupCon 10-epoch checkpoint와 curve
- ViT-ClosedFT-SupCon의 나머지 epoch 기록
- 5개 모델 × 3 prototype × 2 distance × 2 partition × valid/test의 120-row 결과
- 공통 backbone 및 실제 deployed representation t-SNE
- open/closed split별 최종 accuracy, macro-F1, weighted-F1
- GT prototype 전체 이미지의 English-name mapping 및 content leakage audit

누락된 값과 이미지는 임의로 채우지 않는다.

## 7. Artifact and provenance index

| Artifact | Remote path | Notes |
|---|---|---|
| ClosedFT history | `/home/ubuntu/workspace/outputs/20260711_202711_metricdiag_fresh_closedft27_e10/metrics/history.csv` | 10 complete epochs; validation/test columns present |
| ClosedFT curve | `/home/ubuntu/workspace/outputs/metricdiag-viz/vit-closedft.png` | regenerated from current history |
| ClosedFT-SupCon history | `/home/ubuntu/workspace/outputs/20260712_043644_metricdiag_fresh_closedft_supcon27_e10/metrics/history.csv` | 3 completed epochs at snapshot |
| ClosedFT-SupCon curve | `/home/ubuntu/workspace/outputs/metricdiag-viz/vit-closedft-supcon.png` | partial status explicitly shown |
| FullFT history | `/home/ubuntu/workspace/outputs/20260701_060811_models_vit_42class/metrics/history.csv` | existing 54-epoch validation-only history |
| FullFT curve | `/home/ubuntu/workspace/outputs/metricdiag-viz/vit-fullft.png` | regenerated from current history |

새로 저장되는 checkpoint에는 학습 manifest의 SHA-256을 기록하고, 최종 평가 결과에는 evaluator/source/runtime/checkpoint/manifest/GT zip의 SHA-256 fingerprint를 저장하도록 코드가 준비되어 있다. 기존 checkpoint에 학습 당시 hash가 없으면 이를 소급해 생성하지 않고 `training-time hash unavailable`로 표시한다.

## 8. Reporting rule

이 문서는 현재 확인 가능한 학습 기록의 **snapshot**이다. 최종 결론은 모든 필수 checkpoint가 준비되고 공통 evaluator가 120/120 deployed rows, 120/120 common-backbone rows, 모든 t-SNE와 provenance audit를 통과한 뒤 별도 최종 보고서에 기록한다.
