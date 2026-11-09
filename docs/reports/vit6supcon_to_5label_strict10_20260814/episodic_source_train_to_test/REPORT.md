# Source 6-label closed-set train-to-test 평가

## 평가 목적

closed set에서는 open set과 달리 support를 5장으로 제한하지 않았다. source train 전체로 클래스 prototype과 covariance를 만들고, 학습에 사용하지 않은 source test만 query로 평가했다.

- Checkpoint: 기존 6-label SupCon best checkpoint
- Support: source train 676장 전체
- Query: source test 84장 전체
- 클래스: ADIN, BICON, DENTIUM, MIS, NORIS, OSSTEM
- Target validation을 이용한 방법 선택이나 hyperparameter 튜닝은 하지 않았다.

## 데이터 구성

| 클래스 | Train support | Test query |
|---|---:|---:|
| ADIN | 209 | 26 |
| BICON | 83 | 10 |
| DENTIUM | 60 | 8 |
| MIS | 25 | 3 |
| NORIS | 160 | 20 |
| OSSTEM | 139 | 17 |
| **합계** | **676** | **84** |

## Backbone 결과

모든 train support를 사용한 결과이며, 단위는 %다.

| 거리 | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|
| Euclidean | 53.57 | 51.89 | 59.93 |
| Manhattan | 52.38 | 51.40 | 59.14 |
| Cosine | 54.76 | 52.86 | 60.57 |
| Pooled Ledoit-Wolf Mahalanobis | 66.67 | **70.99** | 68.77 |
| **고정 RDA λ=0.75 + logdet** | **72.62** | 70.52 | **70.85** |

## Projection head 결과

| 거리 | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|
| Euclidean | 54.76 | 54.22 | 61.25 |
| Manhattan | 52.38 | 49.85 | 54.18 |
| Cosine | 54.76 | 54.22 | 61.25 |
| Pooled Ledoit-Wolf Mahalanobis | 64.29 | 62.68 | 62.04 |
| **고정 RDA λ=0.75 + logdet** | **71.43** | **69.32** | **68.77** |

## 해석

이제 closed는 open보다 낮지 않다. Backbone 기준 closed RDA는 Accuracy 72.62%, Macro-F1 70.52%, Balanced Accuracy 70.85%다. open에서 같은 backbone의 최고 결과는 Pooled Ledoit-Wolf Mahalanobis 67.98%, 63.30%, 64.49%였다.

따라서 앞서 계산한 `episodic_all_source_6shot`의 44.17%는 closed의 정식 결과가 아니라, open과 동일하게 source support를 5장으로 제한한 비교용 보조 실험이다. closed의 본 결과는 이 보고서처럼 train 전체를 사용하는 train-to-test 평가로 해석한다.

다만 source test가 84장뿐이고 MIS test는 3장이므로 한 장이 전체 Accuracy 약 1.19%p를 움직인다. open과 closed의 클래스 수와 query 구성도 다르므로 점수 차이를 그대로 일반화 성능 차이로 해석하지 않는다.

## 산출물

- `backbone/summary.csv`: backbone 거리별 결과.
- `projection/summary.csv`: projection 거리별 결과.
- `backbone/per_class.csv`, `projection/per_class.csv`: 클래스별 지표.
- `backbone/confusion_*.csv`, `projection/confusion_*.csv`: confusion matrix.
- `protocol.json`: support/query split과 계산 조건.
