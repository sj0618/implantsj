# Unseen 5-label 전체 데이터 pool strict 5-shot 평가

## 결과 요약

Unseen 5개 클래스의 기존 train·valid·test 152장을 하나로 합쳤다. 매 seed에서 클래스당 5장씩 총 25장을 support로 뽑아 prototype과 covariance를 모두 만들고, support를 제외한 127장 전체를 query로 평가했다.

대표값은 6개 support seed의 산술평균이다. 고정 pooled Ledoit-Wolf Mahalanobis는 Accuracy 67.98%, Macro-F1 63.30%, Balanced Accuracy 64.49%를 기록했다.

## 평가 구성

| 클래스 | 전체 | Support | Query |
|---|---:|---:|---:|
| BEGO | 35 | 5 | 30 |
| BILIM | 30 | 5 | 25 |
| IMPLANCE | 34 | 5 | 29 |
| ITI | 38 | 5 | 33 |
| NUCLEOSS | 15 | 5 | 10 |
| **합계** | **152** | **25** | **127** |

- Encoder는 `ADIN, BICON, DENTIUM, MIS, NORIS, OSSTEM` 6개만 학습했다.
- Prototype과 covariance는 동일한 support 5장/클래스만 사용했다.
- Target validation을 이용한 거리 선택이나 hyperparameter 튜닝은 하지 않았다.
- 각 seed 안에서 support와 query의 path 및 exact image는 겹치지 않는다.
- Support seed: 0, 1, 2, 3, 4, 42.

## 거리별 대표 결과

모든 값은 동일한 6개 support seed의 산술평균이다.

| 거리 | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|
| Euclidean | 55.64 | 52.50 | 54.05 |
| Manhattan | 55.91 | 52.92 | 54.40 |
| Cosine | 56.56 | 53.62 | 55.63 |
| Pooled Ledoit-Wolf Mahalanobis | 67.98 | 63.30 | 64.49 |
| 고정 RDA λ=0.75 + logdet | 63.25 | 58.98 | 61.25 |

## 클래스별 Pooled Mahalanobis 평균

| 클래스 | Precision | Recall | F1 |
|---|---:|---:|---:|
| BEGO | 78.81 | 73.89 | 75.39 |
| BILIM | 80.32 | 60.67 | 66.67 |
| IMPLANCE | 68.79 | 75.86 | 71.66 |
| ITI | 73.67 | 68.69 | 70.64 |
| NUCLEOSS | 29.18 | 43.33 | **32.12** |

## 해석

기존 고정 test 17장에서는 한 장이 Accuracy 5.88%p를 움직였지만, 이번 query 127장에서는 한 장이 약 0.79%p를 움직인다. 따라서 이번 결과가 17장 결과보다 안정적이다. 다만 NUCLEOSS는 전체가 15장뿐이라 support 5장을 제외하면 query가 10장이고 성능도 가장 낮다.

이번 평가는 전체 target 이미지를 활용하는 반복 episodic 평가다. 각 seed 내부의 support/query는 독립이지만, 다른 seed 사이에서는 같은 이미지가 서로 다른 역할로 재사용된다. 또한 patient/source-group 독립성은 입증되지 않았으므로 patient-unseen 결과로 단정하지 않는다.

## 산출물

- `all_seed_results.csv`: 모든 seed와 거리 결과.
- `summary.csv`: 거리별 평균·표준편차·범위.
- `per_class_all_seeds.csv`: 클래스별 결과.
- `seed*/support_membership.csv`: 실제 support 5장/클래스.
- `seed*/predictions.csv`: query 127장 예측.
- `protocol.json`: 평가 조건.

## Closed set 추가 평가

같은 checkpoint로 source 6-label closed-set 평가도 수행했다. closed는 train 전체 676장을 support로 사용하고 test 84장을 query로 평가했다. 결과는 [closed-set train-to-test 평가 보고서](../episodic_source_train_to_test/REPORT.md)에서 확인할 수 있다. Open과 동일하게 class당 5장만 사용하는 보조 비교 결과는 [5-shot closed 보조 보고서](../episodic_all_source_6shot/REPORT.md)에 남겨두었다.
