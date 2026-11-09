# Source 6-label closed-set 전체 데이터 pool strict 5-shot 보조 평가

> 이 폴더는 open과 동일한 5-shot 조건을 맞춘 보조 비교용이다. closed의 정식 train-to-test 결과는 [이 보고서](../episodic_source_train_to_test/REPORT.md)를 기준으로 한다.

## 결과 요약

기존 open 평가와 같은 checkpoint, 거리 계산, support seed 6개를 유지하고 source 6개 클래스의 전체 848장을 평가했다. 각 seed에서 클래스당 5장을 support로 뽑고, 나머지 818장을 query로 사용했다.

이 결과는 일반적인 supervised closed-set 분류기 정확도가 아니라, 학습된 embedding에서 5-shot prototype과 covariance만으로 source 클래스가 다시 구분되는지 확인한 결과다.

## 평가 구성

| 클래스 | 전체 | Support | Query |
|---|---:|---:|---:|
| ADIN | 262 | 5 | 257 |
| BICON | 104 | 5 | 99 |
| DENTIUM | 76 | 5 | 71 |
| MIS | 32 | 5 | 27 |
| NORIS | 200 | 5 | 195 |
| OSSTEM | 174 | 5 | 169 |
| **합계** | **848** | **30** | **818** |

- Support seed는 `0, 1, 2, 3, 4, 42`다.
- Prototype과 covariance는 같은 support 5장/클래스만 사용했다.
- 각 seed 안에서 support와 query의 path는 겹치지 않는다.
- Target validation을 이용한 거리 선택이나 hyperparameter 튜닝은 하지 않았다.

## Backbone 결과

모든 값은 6개 support seed의 산술평균이며, 단위는 %다.

| 거리 | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|
| Euclidean | 39.30 | 39.05 | 45.64 |
| Manhattan | 40.18 | 39.83 | 46.20 |
| Cosine | 39.47 | 39.34 | 46.32 |
| **Pooled Ledoit-Wolf Mahalanobis** | **44.17** | **43.80** | **49.73** |
| 고정 RDA λ=0.75 + logdet | 42.14 | 41.61 | 47.92 |

## Projection head 결과

같은 protocol을 128차원 projection embedding에도 적용했다.

| 거리 | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|
| Euclidean | 39.73 | 38.76 | 44.58 |
| Manhattan | 39.00 | 37.79 | 43.32 |
| Cosine | 39.75 | 38.73 | 44.67 |
| **Pooled Ledoit-Wolf Mahalanobis** | **40.53** | **39.82** | **46.58** |
| 고정 RDA λ=0.75 + logdet | 39.47 | 38.99 | 45.29 |

## 해석

이 closed-set 5-shot 평가에서도 backbone에서는 Pooled Ledoit-Wolf Mahalanobis가 가장 높았다. 다만 open 결과의 67.98%와 closed 결과의 44.17%는 query 수, 클래스 수, 클래스 구성과 embedding 조건이 다르므로 난이도를 직접 비교하는 점수로 해석하면 안 된다.

closed 결과가 낮다는 것은 현재 embedding이 source 클래스 전체를 5장 prototype만으로 안정적으로 다시 나누지는 못한다는 뜻이다. 따라서 Mahalanobis가 open에서 잘 나온 이유를 “source와 target 모두 완벽하게 분리한다”로 확대 해석할 수 없고, target episode의 support 표본과 공분산 구조에 특히 잘 맞았는지 추가 확인이 필요하다.

## 산출물

- `backbone/summary.csv`: backbone 거리별 평균 결과.
- `projection/summary.csv`: projection 거리별 평균 결과.
- `backbone/per_class.csv`, `projection/per_class.csv`: seed별 클래스 지표.
- `protocol.json`: checkpoint, manifest, seed, support/query 조건.
