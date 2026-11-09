# Unseen 3-label 클래스당 10장 mean prototype 평가

## 조건

- Encoder: source 7-label로 SupCon 학습한 ViT-Small, backbone 384-D 고정.
- Target: encoder 학습에 없던 `Bego`, `Bicon`, `ITI`.
- 이 실험에서는 prototype만 제한했다. Target-train에서 클래스별 10장을 무작위 비복원 추출하고, 10개 embedding 평균으로 클래스당 prototype 1개를 만들었다.
- Prototype 사용량: Bego 10 + Bicon 10 + ITI 10 = 총 30장.
- Covariance: 기존 조건대로 target-train 전체(Bego 282, Bicon 3,071, ITI 219)의 embedding으로 추정.
- 방법 선택: target-valid Macro-F1만 사용. Test는 적합·선택에 사용하지 않음.
- 평가: target-test 446장. Seed 42를 주 결과로 쓰고 seed 0, 1, 2, 3, 4를 추가해 표본 분산 확인.

## Seed 42 결과

| Prototype | Distance | Test Accuracy | Test Macro-F1 | Balanced Accuracy |
|---|---|---:|---:|---:|
| 클래스당 10장 Mean | Pooled Ledoit-Wolf Mahalanobis | 93.05% | 77.88% | 80.04% |
| 클래스당 10장 Mean | Validation-selected RDA Mahalanobis | **98.88%** | **96.33%** | **97.27%** |
| 전체 train Mean | 동일 RDA Mahalanobis | 99.55% | 98.21% | 97.53% |

Prototype을 전체 평균에서 10장 평균으로 줄이자 개선형 RDA 기준 Accuracy는 `-0.67%p`, Macro-F1은 `-1.88%p` 낮아졌다. 그래도 10장만으로 높은 성능을 유지했다.

## 6개 prototype seed 반복

| Seed | Test Accuracy | Macro-F1 | Balanced Accuracy | Validation 선택법 |
|---:|---:|---:|---:|---|
| 0 | 99.33% | 97.56% | 97.44% | RDA λ=0.75 + logdet |
| 1 | 99.10% | 96.93% | 97.36% | RDA λ=0.75 + logdet |
| 2 | 98.65% | 95.83% | 97.18% | RDA λ=0.75 + logdet |
| 3 | 99.33% | 97.56% | 97.44% | RDA λ=0.75 + logdet |
| 4 | 99.33% | 97.56% | 97.44% | RDA λ=0.75 + logdet |
| 42 | 98.88% | 96.33% | 97.27% | RDA λ=0.75 + logdet |
| 평균 ± 표준편차 | 99.10 ± 0.28% | 96.96 ± 0.74% | 97.36 ± 0.11% | 6/6 동일 |
| 최솟값~최댓값 | 98.65~99.33% | 95.83~97.56% | 97.18~97.44% | - |

전체 train Mean prototype의 99.55 / 98.21과 비교하면, 10-shot 6-seed 평균은 Accuracy `-0.45%p`, Macro-F1 `-1.25%p`다.

## 클래스별 6-seed 평균

| Class | Recall 평균 | F1 평균 | F1 범위 |
|---|---:|---:|---:|
| Bego | 100.00% | 98.36% | 97.22~98.59% |
| Bicon | 99.48% | 99.61% | 99.35~99.74% |
| ITI | 92.59% | 92.90% | 90.91~94.34% |

## 중복 제거 민감도

Target-train과 exact hash가 같은 test 5장을 제외한 441장에서도 6-seed 평균은 다음과 같다.

- Accuracy: 99.09 ± 0.29%
- Macro-F1: 96.96 ± 0.74%
- Balanced Accuracy: 97.36 ± 0.11%

원 결과와 사실상 동일하다.

## 해석

이 실험은 prototype 생성 이미지를 클래스당 10장으로 줄여도 RDA Mahalanobis 성능이 유지되는지 확인했다. Prototype에는 30장만 사용했지만 covariance는 target-train 전체에서 추정했다. 따라서 전체 target 정보를 클래스당 10장으로 제한한 strict 10-shot 실험은 아니며, target-valid로 거리 설정을 골랐기 때문에 strict zero-shot도 아니다.

## 산출물

- `backbone_seed0/` … `backbone_seed42/`: seed별 결과와 실제 10장 membership.
- `backbone_seed_results.csv`: baseline·개선법 seed별 test 결과.
- `backbone_selected_per_class_all_seeds.csv`: 클래스별 결과.
- `backbone_duplicate_free_all_seeds.csv`: exact duplicate 제거 결과.
- 각 seed의 `prototype_membership.csv`: prototype 생성에 사용된 클래스별 10장 경로.
