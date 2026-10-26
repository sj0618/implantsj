# 마하라노비스 거리 실험 결과 (2026-08-16)

## 최종 선택

이번 비교에서는 `rda_0.75_logdet_1`을 최종 방법으로 골랐다. fresh와 legacy 데이터의 backbone, projection 조합 네 가지에서 공통으로 실행할 수 있는 24개 후보를 비교했다. 순위는 검증 데이터의 Macro-F1 평균으로 정했고, 테스트 결과는 방법을 고른 뒤 확인했다.

`rda_0.75_logdet_1`의 검증 Macro-F1은 평균 71.96%였다. `pooled_ledoit_wolf`보다 2.33%p 높다. 테스트에서도 평균 Macro-F1이 7.66%p 올랐으며, 네 조건에서 모두 pooled baseline보다 높았다.

## 검증 성능 순위

| method | macro_f1 | balanced_accuracy | accuracy |
|---|---|---|---|
| rda_0.75_logdet_1 | 71.96 | 76.62 | 80.13 |
| multi_prototype_2 | 70.29 | 74.38 | 80.02 |
| multi_prototype_5 | 70.19 | 72.86 | 79.36 |
| multi_prototype_3 | 70.05 | 74.28 | 79.95 |
| equal_class_covariance_0.25 | 69.85 | 74.56 | 79.18 |
| pooled_oas | 69.74 | 73.03 | 79.71 |
| pooled_ledoit_wolf | 69.64 | 72.92 | 79.59 |
| equal_class_covariance_0.5 | 68.96 | 74.17 | 78.68 |
| equal_class_covariance_0.75 | 68.53 | 74.37 | 78.58 |
| rda_0.5_logdet_1 | 68.16 | 76.58 | 76.45 |

## 테스트 결과

| condition | method | accuracy | macro_f1 | balanced_accuracy |
|---|---|---|---|---|
| fresh_backbone | pooled_ledoit_wolf | 98.21 | 92.94 | 91.25 |
| fresh_backbone | rda_0.75_logdet_1 | 99.55 | 98.21 | 97.53 |
| fresh_projection | pooled_ledoit_wolf | 93.27 | 81.46 | 86.46 |
| fresh_projection | rda_0.75_logdet_1 | 94.84 | 87.91 | 95.99 |
| legacy_backbone | pooled_ledoit_wolf | 63.30 | 48.23 | 50.77 |
| legacy_backbone | rda_0.75_logdet_1 | 71.56 | 57.60 | 63.26 |
| legacy_projection | pooled_ledoit_wolf | 60.55 | 44.77 | 47.65 |
| legacy_projection | rda_0.75_logdet_1 | 71.56 | 54.31 | 56.53 |

## Strict 5-label 5-shot 확인

지원 표본이 적을 때는 양상이 달랐다. 클래스마다 5장만 사용해 프로토타입과 공분산을 만들고 여섯 개 seed로 반복했을 때, `rda_0.75_logdet_1`의 Macro-F1은 pooled 기준보다 4.31%p 낮았다. 이 결과를 보면 RDA는 공분산을 추정할 target-train 표본이 충분한 현재 3-label 전이 설정에 적합하다. 클래스별 지원 표본이 5장뿐인 조건에서는 pooled Ledoit-Wolf가 더 안정적이었다.

| method | macro_f1 | balanced_accuracy | accuracy |
|---|---|---|---|
| pooled_ledoit_wolf | 63.30 | 64.49 | 67.98 |
| class_balanced_ledoit_wolf | 63.30 | 64.49 | 67.98 |
| equal_class_covariance_0.25 | 63.30 | 64.49 | 67.98 |
| multi_prototype_2 | 63.28 | 64.47 | 67.98 |
| multi_prototype_3 | 63.25 | 64.38 | 67.85 |
| pooled_oas | 63.19 | 64.38 | 67.85 |
| equal_class_covariance_0.5 | 63.18 | 64.39 | 67.85 |
| equal_class_covariance_0.75 | 63.18 | 64.39 | 67.85 |
| equal_class_covariance_1 | 63.18 | 64.39 | 67.85 |
| multi_prototype_4 | 63.09 | 64.27 | 67.72 |

저장된 strict 결과와 다시 계산한 값의 최대 절대 오차는 `0`이었다.

## 적용 기준

현재 3-label 전이 평가에는 클래스별 공분산을 75% 반영하고 로그 행렬식 보정값을 더한 RDA를 사용한다. 공분산을 클래스당 5장 정도로만 추정해야 한다면 pooled Ledoit-Wolf를 사용한다.
