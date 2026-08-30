# 임플란트 분류 실험 결과 통합표

> 표기: `Test Accuracy / Macro-F1 (%)`\
> 주의: direct classification, legacy transfer, uncropped transfer, GT27 diagnosis는 서로 다른 split·protocol이므로 절대 순위를 단순 비교하지 않는다.

| 구분 | 라벨/대상 | 방법 | Test Accuracy | Macro-F1 | 해석 |
|---|---|---|---:|---:|---|
| Direct classifier | 3-label | Plain ViT | 94.50 | 77.00 | 전용 3-label baseline |
| Direct classifier | 7-label | Plain ViT | 94.40 | 88.29 | plain 모델 중 균형이 좋음 |
| Direct classifier | 10-label | Plain ViT | **95.54** | 88.12 | 직접 분류 정확도 최고 |
| Latest classifier | 27 closed-label | ClosedFT CE | 95.43 | **92.67** | 최신 closed-set 결과 |
| Metric classifier | 7-label | SupCon + prototype | 95.87 | 75.12 | Accuracy는 높지만 minority F1이 낮음 |
| Error correction | 10-label | Main ViT + two-stage | 95.09 | 87.86 | baseline보다 개선 없음 |
| Fine-tuning transfer | 10 → 3-label | Checkpoint fine-tune | 91.74 | 71.73 | validation 개선이 test로 유지되지 않음 |
| Legacy metric transfer | 7 → 3-label | Direct prototype | 37.61 | 34.75 | 직접 전이 성능 약함 |
| Legacy metric transfer | 7 → 3-label | Metric two-stage | 68.81 | 54.92 | 개선되지만 dedicated 3-label보다 낮음 |
| Uncropped transfer | 7 → 3-label | ViT-Base | **82.76** | **64.16** | 학습하지 않은 Base가 최고 |
| Uncropped transfer | 7 → 3-label | CE backbone | 72.41 | 50.14 | fine-tuning 후 전이 하락 |
| Uncropped transfer | 7 → 3-label | Triplet backbone | 68.10 | 55.48 | Base보다 낮음 |
| Uncropped transfer | 7 → 3-label | SupCon backbone | 53.45 | 39.88 | source 분리와 target 전이 불일치 |
| GT27 metric diagnosis | 27-label GT, cosine | ViT-Base | 51.92 | 52.80 | generic representation 기준선 |
| GT27 metric diagnosis | 27-label GT, cosine | ViT-FullFT | 50.00 | 50.59 | Base보다 소폭 낮음 |
| GT27 metric diagnosis | 27-label GT, cosine | ViT-ClosedFT | 44.23 | 42.46 | fine-tuning 후 generic metric 표현 약화 |
| GT27 metric diagnosis | 27-label GT, cosine | ViT-FullFT-SupCon | 40.38 | 43.08 | 현재 조건에서 가장 낮은 accuracy |
| VLM feature-only | 7-label | Structural feature ML | 32~45 | - | image-only ViT를 대체하지 못함 |

## 요약

| 질문 | 결론 |
|---|---|
| 직접 분류 baseline은? | 10-label plain ViT, Test Accuracy **95.54%** |
| Macro-F1이 가장 높은 최신 결과는? | 27-label ClosedFT, **92.67%** |
| Metric transfer에서 가장 안정적인 표현은? | Uncropped 7→3 조건의 ViT-Base, Accuracy **82.76%**, Macro-F1 **64.16%** |
| Two-stage는 도움이 됐나? | Submodel 자체는 좋았지만 전체 성능 개선에는 실패 |
| VLM feature가 ViT를 대체하나? | 대체하지 못함. 보조 signal·ablation 용도 |
| 다음 개선 우선순위는? | ROI crop, 384/512 multi-crop, group-balanced prototype, Macro-F1 중심 평가, 3개 이상 seed |

## 근거 문서

- `docs/analysis/EXPERIMENT_SUMMARY.md`
- `docs/reports/METRIC_DIAGNOSIS_FINAL_REPORT.md`
- `docs/reports/full_uncropped_7to3_metric/report/REPORT.md`
- `docs/analysis/PROJECT_ANALYSIS.md`
