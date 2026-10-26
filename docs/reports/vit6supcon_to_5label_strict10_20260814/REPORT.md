# 11-label 데이터: 6-label SupCon → unseen 5-label strict 10-shot

## 결과 요약

`ADIN, BICON, DENTIUM, MIS, NORIS, OSSTEM` 6개만으로 ViT-Small SupCon을 30 epoch 학습하고, 학습에 없던 `BEGO, BILIM, IMPLANCE, ITI, NUCLEOSS` 5개를 평가했다. 클래스당 동일한 10장으로 prototype과 covariance를 모두 만들었으며 target validation은 방법 선택에 사용하지 않았다.

고정 pooled Ledoit-Wolf Mahalanobis의 6-seed 평균은 Accuracy 69.61 ± 6.88%, Macro-F1 59.60 ± 7.17%, Balanced Accuracy 61.94 ± 6.45%다.

## 데이터 분리

| 용도 | 클래스 | Train | Valid | Test |
|---|---|---:|---:|---:|
| Source 학습 | adin, bicon, dentium, mis, noris, osstem | 676 | 88 | 84 |
| Target unseen | bego, bilim, implance, iti, nucleoss | 121 | 14 | 17 |

- Source와 target의 label은 완전히 분리했다.
- Source와 target 사이, target train과 valid 및 test 사이의 exact SHA 중복은 모두 0장이다.
- Target support는 각 target-train에서 10장씩 총 50장만 사용했다.

## 학습 조건

- Model: `vit_small_patch16_224.augreg_in21k_ft_in1k`, backbone 384-D.
- Loss: SupCon, projection 128-D, temperature 0.07.
- Training: 30 epoch, batch 16, learning rate 3e-5, seed 42.
- Checkpoint selection: source 6-label validation Macro-F1만 사용.
- Best checkpoint: epoch 20.
- Source validation: Accuracy 55.68%, Macro-F1 55.29%.
- Source test: Accuracy 54.76%, Macro-F1 54.22%.

## Unseen 5-label strict 10-shot 결과

값은 6개 support seed의 `평균 ± 표준편차`이며, backbone 384-D를 사용했다.

| 거리 | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|
| Euclidean | 47.06 ± 14.88 | 41.79 ± 12.70 | 42.78 ± 13.97 |
| Manhattan | 45.10 ± 14.73 | 39.21 ± 13.71 | 40.83 ± 13.57 |
| Cosine | 48.04 ± 16.81 | 41.84 ± 14.36 | 43.61 ± 15.51 |
| Pooled Ledoit-Wolf Mahalanobis | 69.61 ± 6.88 | 59.60 ± 7.17 | 61.94 ± 6.45 |
| 고정 RDA λ=0.75 + logdet | 65.69 ± 4.43 | 57.28 ± 3.93 | 60.00 ± 3.65 |

주 결과는 target validation으로 방법을 고르지 않은 고정 pooled Ledoit-Wolf다. RDA는 이전 설정을 그대로 고정한 참고 결과다.

## Pooled Mahalanobis seed별 결과

| Seed | Accuracy | Macro-F1 | Balanced Accuracy |
|---:|---:|---:|---:|
| 0 | 64.71 | 50.56 | 55.00 |
| 1 | 70.59 | 61.48 | 63.33 |
| 2 | 70.59 | 61.48 | 63.33 |
| 3 | 64.71 | 57.54 | 58.33 |
| 4 | 82.35 | 71.56 | 73.33 |
| 42 | 64.71 | 55.00 | 58.33 |

## 클래스별 결과

Pooled Mahalanobis 6-seed 평균이다.

| Target class | Precision | Recall | F1 |
|---|---:|---:|---:|
| BEGO | 89.17 | 87.50 | 87.37 |
| BILIM | 58.33 | 55.56 | 56.19 |
| IMPLANCE | 78.33 | 79.17 | 76.85 |
| ITI | 70.56 | 87.50 | 77.59 |
| NUCLEOSS | **0.00** | **0.00** | **0.00** |

## 해석 제한

- Test가 총 17장뿐이어서 한 장이 전체 Accuracy를 5.88%p 움직인다.
- NUCLEOSS test는 2장뿐이며 6개 seed에서 모두 오분류됐다.
- 현재 split의 patient/source-group 독립성은 입증되지 않았다.
- 따라서 결과는 `label-unseen + 50-image support 제한`은 충족하지만 patient-unseen 일반화 성능으로 단정할 수 없다.

## 산출물

- `manifests/source_6.csv`, `manifests/target_5.csv`: 실제 source/target 분리.
- `training/20260814_004412_vit_small_supcon6_e30_seed42/checkpoints/best.pt`: 학습된 best checkpoint.
- `strict10/backbone_seed*/`: 고정 Pooled Mahalanobis 6-seed 결과.
- `strict10/rda_seed*/`: 고정 RDA 참고 결과.
- 각 평가 폴더의 `prototype_membership.csv`: 실제 사용한 클래스당 10장.
