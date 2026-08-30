# 동일한 prototype과 covariance를 사용한 10-shot Mahalanobis 평가

## 결과 요약

클래스당 동일한 10장만 사용해 prototype과 covariance를 모두 만들고, target validation을 사용하지 않은 고정 Mahalanobis로 test 446장을 평가했다. 값은 `Accuracy / Macro-F1 / Balanced Accuracy`다.

| 방법 | Seed 42 | 6-seed 평균 ± 표준편차 | 6-seed 범위 |
|---|---:|---:|---:|
| 고정 Pooled Ledoit-Wolf | 69.06 / 53.32 / 76.77 | 69.36 ± 7.12 / 52.93 ± 6.87 / 69.92 ± 7.18 | 60.31~78.48 / 44.84~63.45 / 60.76~78.14 |
| 고정 RDA λ=0.75 + logdet | 69.06 / 56.08 / 74.17 | 70.40 ± 6.74 / 54.40 ± 7.95 / 68.02 ± 8.34 | 61.21~81.39 / 44.67~65.76 / 55.65~78.12 |

Target-validation 튜닝을 사용하지 않은 주 결과는 첫 행의 pooled Ledoit-Wolf다. RDA는 이전 실험 설정을 그대로 고정해 covariance 표본 수의 영향만 비교한 참고 결과다.

## Seed별 결과

| Seed | Pooled Acc. | Pooled F1 | Pooled BAcc. | RDA Acc. | RDA F1 | RDA BAcc. |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 76.46 | 57.82 | 73.05 | 81.39 | 65.76 | 78.12 |
| 1 | 60.31 | 44.84 | 60.76 | 61.21 | 44.67 | 55.65 |
| 2 | 78.48 | 63.45 | 78.14 | 73.32 | 60.67 | 72.38 |
| 3 | 68.61 | 50.80 | 67.45 | 67.04 | 48.20 | 64.83 |
| 4 | 63.23 | 47.34 | 63.34 | 70.40 | 51.01 | 62.95 |
| 42 | 69.06 | 53.32 | 76.77 | 69.06 | 56.08 | 74.17 |

## 이전 10-shot prototype 실험과 비교

| 조건 | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|
| Prototype 10장, covariance target-train 전체, validation-selected RDA, seed 42 | 98.88 | 96.33 | 97.27 |
| Prototype 10장, covariance 동일 10장, 고정 RDA, seed 42 | 69.06 | 56.08 | 74.17 |
| 차이 | **-29.82%p** | **-40.25%p** | **-23.10%p** |

98.88%는 10장 prototype 자체보다 target-train 3,572장으로 추정한 covariance와 target-validation 방법 선택의 영향이 컸다. Covariance까지 같은 10장으로 제한하면 성능과 seed 안정성이 크게 낮아진다.

## 실험 조건과 제한

- Encoder: source 7-label SupCon ViT-Small backbone 384-D 고정.
- Target label: Bego, Bicon, ITI. Encoder 학습에는 포함되지 않음.
- Support: target-train에서 클래스당 10장, 총 30장 무작위 비복원 추출.
- Prototype: 각 클래스의 동일 10개 embedding 평균.
- Covariance: prototype에 사용한 동일 30개 embedding만 사용.
- Method selection: 이번 실행에서 target-valid를 사용하지 않고 방법을 고정.
- Query: target-test 446장만 사용.
- 선택된 support와 test 사이 exact duplicate: 6개 seed 모두 0장.
- 현재 cropped 데이터는 patient/source-group 독립 split이 입증되지 않았다. 따라서 이 결과는 `30-image support 제한`을 충족하지만 완전한 patient-unseen 결과는 아니다.

## 산출물

- `backbone_seed*/`: 고정 Pooled Ledoit-Wolf 결과.
- `rda_seed*/`: 고정 RDA와 동일 seed의 pooled baseline 결과.
- 각 폴더의 `prototype_membership.csv`: 실제 사용한 클래스당 10장.
- 각 폴더의 `protocol.json`: covariance와 validation 사용 조건.
