# 치과 임플란트 분류 실험 결과 요약

## 1. 요약

직접 분류는 ViT image-only가 가장 안정적이었다. 그러나 source label에서 unseen target으로 옮기는 metric transfer에서는 fine-tuning보다 학습하지 않은 Base가 더 잘 전이됐다.

| 대표 수치 | 결과 |
|---|---:|
| 10-label plain ViT 직접 분류 최고 Test Accuracy | **95.54%** |
| 27-label ClosedFT 최신 Test Macro-F1 | **92.67%** |
| uncropped 7→3 metric transfer 관측 최고 Accuracy | **82.76%** |

실험 범위: 3/7/10-label supervised, SupCon/Triplet, two-stage, VLM feature, 7→3 transfer, 27 closed + 15 open metric diagnosis.

## 2. 실험별 정량 결과

표기는 `Test Accuracy / Macro-F1` 순서다. 서로 다른 split·protocol은 절대값보다 각 실험 축의 방향성으로 해석한다.

| 실험 축 | 대표 결과 | 해석 |
|---|---|---|
| Direct classifier: plain ViT (3/7/10-label) | 3-label **94.50 / 77.00**<br>7-label **94.40 / 88.29**<br>10-label **95.54 / 88.12** | 직접 분류의 안정적 기준선 |
| Latest direct classifier: 27-label ClosedFT | **95.43 / 92.67** | 최신 closed-set 결과 |
| SupCon: 7-label | **95.87 / 75.12** | Accuracy는 높지만 minority class F1이 낮음 |
| Two-stage: 10-label | **95.09 / 87.86** | Main baseline 95.54%보다 개선 없음 |
| Fine-tune: 10→3 label | **91.74 / 71.73** | Validation gain이 test로 유지되지 않음 |
| Metric transfer: 7→3 legacy | **37.61 / 34.75 → 68.81 / 54.92** | Two-stage는 개선하지만 dedicated 3-label보다 낮음 |
| Metric transfer: 7→3 uncropped | Base **82.76 / 64.16**<br>CE **72.41 / 50.14**<br>Triplet **68.10 / 55.48**<br>SupCon **53.45 / 39.88** | 학습하지 않은 Base가 fine-tuned 표현보다 높음 |
| Metric diagnosis: 42-label GT27, cosine | Base **51.92 / 52.80** > Full **50.00 / 50.59**<br>> Closed **44.23 / 42.46** > Sup **40.38 / 43.08** | Fine-tuning이 generic metric 표현을 약화 |
| VLM structural feature-only | **32~45% Accuracy** vs image-only **94%+** | 최종 classifier 대체 불가; 보조·ablation 용도 |

## 3. 현재 판단과 다음 우선순위

### 유지할 것

- 10-label plain ViT를 실용 baseline으로 유지한다.
- 3-label 작업은 dedicated plain ViT를 사용한다.
- Direct classification에서는 plain ViT가 가장 안정적이다.

### Metric learning

- SupCon/Triplet은 source 내부 분리는 가능했다.
- 하지만 unseen 3-label transfer는 Base보다 낮았다.
- 따라서 현재는 최종 classifier보다 분석용 embedding으로 활용한다.

### 보정 실험

- Two-stage: **95.54% → 95.09%**로 전체 성능이 하락했다.
- 10→3 fine-tune: Test Accuracy **91.74%**.
- 두 방식 모두 기본 파이프라인으로 채택하지 않는다.

### 다음 우선순위

- VLM feature-only는 32~45% 수준으로 최종 classifier를 대체하지 못한다.
- Loss·거리식 튜닝보다 ROI crop을 우선한다.
- 384/512 multi-crop을 비교한다.
- Source-group balanced prototype을 사용한다.
- Accuracy만 보지 말고 Macro-F1 중심으로 평가한다.
- 최소 3개 seed 또는 반복 split으로 재검증한다.

## 정리

직접 분류 성능은 충분히 높다. 일반화가 막히는 주된 이유는 source와 target의 목적이 다르고, uncropped 영상에서 ROI가 작으며, prototype이 전체 분포를 잘 대표하지 못하기 때문이다.

### 근거 문서

- `docs/analysis/EXPERIMENT_SUMMARY.md`
- `docs/reports/METRIC_DIAGNOSIS_FINAL_REPORT.md`
- `docs/reports/full_uncropped_7to3_metric/report/REPORT.md`
- `docs/analysis/PROJECT_ANALYSIS.md`
