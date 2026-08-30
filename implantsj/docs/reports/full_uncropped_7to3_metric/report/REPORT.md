# FULL 7-label to uncropped 3-label metric transfer report

2026-07-23 완료. `FULL_7label`만으로 backbone을 학습하고, crop하지 않은 `FULL_Bego_Bicon_ITI_3label`을 target gallery/query로 사용했다. 3개 학습 loss, 6개 표현, 3개 prototype, 2개 거리로 구성된 primary test 36/36 설정을 모두 실행했다.

## 결론

- 이 조건에서는 학습하지 않은 ViT-Base가 가장 높았다. validation-GT+cosine에서 accuracy 82.76%, macro-F1 64.16%를 기록했다.
- 7라벨 fine-tuning은 source 내부 성능은 높였지만 unseen 3라벨 전이는 모두 악화시켰다. 관측 최고 accuracy는 CE 72.41%, Triplet backbone 68.10%, SupCon backbone 53.45%, SupCon projection 44.83%다.
- 낮은 metric 성능은 거리식보다 source 7라벨에 특화된 표현과 unseen target 3라벨의 목적 및 domain 차이, uncropped 이미지에서 implant가 작게 토큰화되는 문제와 더 관련이 있다.
- Target test 116장 중 Bicon이 95장이라 Bicon만 예측해도 accuracy가 81.90%다. 따라서 macro-F1, macro-recall, 클래스별 recall을 함께 봐야 한다.

## 실험 프로토콜

- Backbone: `vit_base_patch16_224`, ImageNet pretrained, 입력 224×224, seed 2026, 15 epochs.
- Source 학습 라벨: `ADIN`, `DIONAVI`, `Dentium`, `MIS`, `NORIS`, `nobel`, `osstem`.
- Target 평가 라벨: `Bego`, `Bicon`, `ITI`.
- 학습: CE, SupCon(temperature 0.07, K=2), batch-hard Triplet(margin 0.2, K=2).
- 표현: Base 768d, CE backbone 768d, SupCon backbone 768d/projection 128d, Triplet backbone 768d/projection 128d.
- Prototype:
  - `validation-GT`: target validation 전체를 라벨별 평균.
  - `Mean`: target train/support 전체를 라벨별 평균.
  - `SS`: target train/support에서 seed로 고정한 라벨당 5장 평균.
- 거리: Euclidean, cosine.
- Backbone 학습과 checkpoint 선택에는 3라벨 이미지가 전혀 들어가지 않았다. Target train/validation은 backbone 고정 후 prototype을 만드는 데만 사용했고, target test 116장은 query로만 사용했다.

> 이 보고서의 `validation-GT`는 독립 외부 42-label GT가 아니다. `FULL_Bego_Bicon_ITI_3label`의 labeled validation split으로 만든 target prototype이다. 따라서 이 실험은 zero-shot이 아니라 supervised target-gallery metric transfer다.

## 데이터 및 누수 감사

- 원본 usable 이미지 6,058장, 모든 이미지 640×640. 빈 YOLO label 18개는 제외했다.
- 원본 split에서 Roboflow source 계열 233개가 split을 가로질렀다. Immutable test와 같은 `group_id`를 가진 개발 이미지 363장(Bego 49, Bicon 314)을 제거한 뒤 train/validation을 `group_id` 단위로 재분할했다.
- 최종 train-validation-test path 겹침 0, source-group 겹침 0, 원본 split 간 exact SHA 중복 0.
- Source 7라벨: train 2,348 / validation 593 / 내부 test 339.
- Target 3라벨: support 1,839장(고유 source 332), validation-GT 460장(77 source), test 116장(106 source).
- Target test: Bego 14, Bicon 95, ITI 7. ITI 1장의 정오답은 ITI recall을 14.29%p 움직인다.
- Support/validation-GT에는 같은 원본의 증강본이 그룹당 최대 9장 있어 image-weighted prototype이 특정 source를 과대가중할 수 있다.

## 학습 결과

| 모델 | loss | best epoch | source validation | 동 epoch source test | 선택 기준 |
| --- | --- | ---: | ---: | ---: | --- |
| ViT-ClosedFT | CE | 12 | 95.11% | 97.94% | valid_loss |
| ViT-ClosedFT-SupCon | SUPCON | 12 | 94.44% | 97.35% | valid_accuracy |
| ViT-ClosedFT-Triplet | TRIPLET | 15 | 85.67% | - | valid_accuracy |

CE와 SupCon의 source test는 기록만 했고 선택에 사용하지 않았다. Triplet은 source test의 `nobel`이 1장뿐이라 batch-hard 평가용 positive pair를 만들 수 없어 source test loss/accuracy 기록을 생략했다. Triplet checkpoint 선택은 source validation mean-prototype accuracy만 사용했다.

![Training curves](../figures/training_curves.png)

## Primary 3-label test 결과

아래의 "최고"는 36개 test 조합을 계산한 뒤 관측한 값이다. 별도 target validation query로 선택한 운영점이 아니므로 배포 모델 선택 수치로 사용할 수 없다.

| 표현 | 최고 accuracy (설정) | 그 설정 macro-F1 | 최고 macro-F1 (설정) | 그 설정 accuracy |
| --- | ---: | ---: | ---: | ---: |
| ViT-Base | 82.76% (GT+Cosine) | 64.16% | 64.16% (GT+Cosine) | 82.76% |
| CE backbone | 72.41% (Mean+Euclidean) | 50.14% | 50.33% (GT+Euclidean) | 70.69% |
| SupCon backbone | 53.45% (GT+Euclidean) | 39.88% | 39.88% (GT+Euclidean) | 53.45% |
| SupCon projection | 44.83% (Mean+Cosine) | 35.93% | 35.93% (Mean+Cosine) | 44.83% |
| Triplet backbone | 68.10% (Mean+Euclidean) | 55.48% | 55.48% (Mean+Euclidean) | 68.10% |
| Triplet projection | 62.93% (GT+Euclidean) | 47.03% | 47.03% (GT+Euclidean) | 62.93% |

전체 6개 표현 중 Base가 accuracy와 macro-F1 모두 1위다. Base 최고 설정의 클래스별 결과는 Bego 11/14 (78.57% recall), Bicon 82/95 (86.32%), ITI 3/7 (42.86%)다. Accuracy는 majority baseline보다 0.86%p만 높지만, 세 라벨을 모두 일부 구분해 macro-F1은 majority-only 분류기보다 훨씬 낫다.

### Prototype 평균

| Prototype | accuracy | macro-recall | macro-F1 |
| --- | ---: | ---: | ---: |
| Mean | 62.93% | 61.96% | 48.26% |
| GT | 62.93% | 56.28% | 45.10% |
| SS | 44.83% | 53.69% | 36.77% |

Mean이 전체 평균 macro-F1에서 가장 안정적이고, SS는 라벨당 5장과 증강 source 중복 때문에 크게 흔들렸다. Validation-GT는 Base에서 최고였지만 모델 전체 평균에서는 Mean보다 낮았다.

### 거리 평균

| 거리 | accuracy | macro-recall | macro-F1 |
| --- | ---: | ---: | ---: |
| Cosine | 57.04% | 57.55% | 43.47% |
| Euclidean | 56.75% | 57.07% | 43.29% |

Cosine과 Euclidean의 평균 macro-F1 차이는 0.18%p뿐이다. 거리 선택은 부차적이며 표현 자체의 target 분리가 병목이다.

## Source-group 균형 sensitivity

같은 원본의 증강본을 먼저 source-group 안에서 평균하고, 각 source-group을 동일 가중치로 prototype에 반영했다. Query는 기존 116장 그대로 둔 비교다.

| 표현 | Primary 최고 macro-F1 | Group-balanced 최고 macro-F1 | 변화 | Group-balanced macro-recall |
| --- | ---: | ---: | ---: | ---: |
| ViT-Base | 64.16% | 65.95% | +1.79%p | 75.69% |
| CE backbone | 50.33% | 51.24% | +0.91%p | 60.98% |
| SupCon backbone | 39.88% | 40.75% | +0.87%p | 58.65% |
| SupCon projection | 35.93% | 37.00% | +1.08%p | 56.19% |
| Triplet backbone | 55.48% | 55.99% | +0.51%p | 67.69% |
| Triplet projection | 47.03% | 46.28% | -0.75%p | 63.48% |

대부분 소폭 개선됐지만 순위는 바뀌지 않았다. Base macro-F1은 +1.79%p, CE는 +0.91%p, SupCon backbone은 +0.87%p였다. 증강본 가중치 편향은 실제 문제지만 전체 성능 저하의 주원인은 아니다.

## 낮은 metric 정확도의 원인

1. 학습 목적과 평가 목적이 다르다. CE/SupCon/Triplet은 source 7라벨 사이의 분리를 최적화했다. Source validation은 CE 95.11%, SupCon 94.44%, Triplet 85.67%였지만 target 3라벨에서는 Base보다 모두 낮았다. 특히 closed-only SupCon은 source class 응집을 강화하는 대신 unseen 제조사에 필요한 일반 표현을 가장 많이 잃었다.
2. Uncropped 상태에서 implant가 너무 작다. Target 2,778장의 가장 큰 annotation 외접박스 면적 중앙값은 전체 화면의 5.53%이고 81.35%는 10% 미만이다. 224 resize 후 implant 짧은 변 중앙값은 37.6px, 32.65%는 ViT-B/16 patch 한 칸인 16px보다 작고 p10은 8.05px다. 나사산·neck·connection 같은 세부 구조가 tokenization 전에 사라질 수 있다.
3. 한 클래스가 여러 촬영 양식/배경 모드를 포함한다. Mean prototype geometry에서 모든 표현의 `inter-prototype / intra-class` separation ratio가 1보다 작았다(Base 0.45, CE 0.53, SupCon backbone 0.52, SupCon projection 0.65, Triplet backbone 0.43, Triplet projection 0.39). 즉 같은 라벨 내부 변이가 prototype 사이 거리보다 크므로 한 centroid로 대표하기 어렵다.
4. 불균형과 작은 test가 accuracy를 왜곡한다. Bicon이 81.90%를 차지하고 ITI는 7장뿐이다. SupCon backbone 최고 설정은 Bego를 14/14 맞췄지만 Bicon recall 49.47%, ITI 14.29%로 무너졌다.
5. Projection head가 source objective에 더 특화됐다. SupCon projection은 backbone보다 최고 accuracy가 8.62%p 낮고, Triplet projection도 backbone보다 5.17%p 낮았다. Source metric loss가 직접 작동한 128d 공간이 unseen target에는 오히려 덜 전이됐다.
6. Prototype sampling 분산이 크다. SS 평균 macro-F1은 36.77%로 Mean 48.26%보다 11.49%p 낮다. 라벨당 5장만 사용하면서 동일 source 증강본이 섞여 representative prototype을 만들지 못했다.
7. 거리식을 바꿔도 representation 문제는 해결되지 않는다. Cosine/Euclidean의 전체 평균 차이가 거의 없고, 여러 Triplet 설정은 두 거리의 예측이 완전히 같았다.

## 권장 사항

- 이 7-label to 3-label uncropped transfer에서는 ViT-Base를 유지한다.
- Target source-group 정보를 사용할 수 있다면 Base + group-balanced Mean + cosine이 sensitivity에서 accuracy 82.76%, macro-F1 65.95%, macro-recall 75.69%로 가장 균형적이었다. 단, 이 설정도 test 관측값이므로 별도 validation query에서 선택해야 한다.
- 다음 실험은 loss나 거리 튜닝보다 implant ROI crop, detector/segmentation 기반 ROI pooling, 고해상도 multi-crop, patch보다 작은 implant의 제거 또는 보정을 우선한다.
- Target prototype은 image 평균보다 source-group 평균을 사용하고, SS는 여러 seed 또는 최소 source-group 수를 늘려야 한다.
- Test를 클래스별로 확충하고 최소 3개 seed 평균·신뢰구간을 보고해야 한다. 특히 ITI 7장으로는 결론의 분산이 너무 크다.
- Fine-tuning을 계속하려면 source 7라벨만 강하게 분리하는 SupCon보다 target과 촬영 분포가 다양한 episodic/domain-generalization 학습을 우선한다.

## t-SNE

### ViT-Base


### CE backbone

![CE backbone t-SNE](../figures/ViT-ClosedFT/tsne.png)

### SupCon backbone

![SupCon backbone t-SNE](../figures/ViT-ClosedFT-SupCon-backbone/tsne.png)

### SupCon projection

![SupCon projection t-SNE](../figures/ViT-ClosedFT-SupCon-projection/tsne.png)

### Triplet backbone

![Triplet backbone t-SNE](../figures/ViT-ClosedFT-Triplet-backbone/tsne.png)

### Triplet projection

![Triplet projection t-SNE](../figures/ViT-ClosedFT-Triplet-projection/tsne.png)

t-SNE는 정성 시각화이며 모델 판정은 원공간 prototype 지표를 우선한다. 이 실험은 target 3라벨만 시각화하므로 가운데 closed panel이 비어 있는 것이 정상이다.

## Checkpoint

| 모델 | validation-selected 평가 checkpoint |
| --- | --- |
| CE | `runs/ViT-ClosedFT/best_weights.safetensors` (epoch 12, 원본 `.pt`와 tensor 완전 일치 검증) |
| SupCon | `runs/ViT-ClosedFT-SupCon/best_weights.safetensors` (epoch 12, 원본 `.pt`와 tensor 완전 일치 검증) |
| Triplet | `runs/ViT-ClosedFT-Triplet/checkpoints/best.safetensors` (epoch 15, `last`와 SHA-256 동일) |

CE/SupCon의 optimizer 포함 `.pt`는 학습 재개용으로 그대로 보존했다. 평가에는 검증된 weights-only safetensors를 사용했다.

## 재현성 및 한계

- Seed 2026과 `CUBLAS_WORKSPACE_CONFIG=:4096:8`을 고정했다. PyTorch memory-efficient attention은 일부 CUDA kernel이 비결정적이라는 경고를 냈으므로 재실행 시 미세한 차이가 날 수 있다.
- 한 seed 결과이며 prototype/distance의 "최고"는 test를 본 사후 비교다.
- CE/SupCon은 기존 보고서와 동일한 배포본 `metricdiag.training`, Triplet과 최종 embedding/evaluation은 현재 `implantsj` snapshot을 사용했다. 공통 loss/optimizer/sampler 설정은 동일하지만 완전히 하나의 실행 모듈로 통일한 재실험은 아니다. 코드 hash는 `validation.json`에 기록했다.
- Group-clean 전에 중단한 누수 split run은 공식 결과에서 완전히 제외했다.

## 산출물

- `all_primary_metrics.csv`: 6표현 × 3 prototype × 2 distance = 36행.
- `best_observed_accuracy_by_model.csv`, `best_observed_macro_f1_by_model.csv`.
- `all_per_class_metrics.csv`, `per_class_at_best_macro_f1.csv`.
- `all_geometry.csv`, `prototype_method_averages.csv`, `distance_method_averages.csv`.
- `all_sensitivity_metrics.csv`, `sensitivity_best_macro_f1.csv`.
- `training_summary.csv`, `validation.json`.
- `../figures/training_curves.png`, `../figures/<model>/tsne.png`.
- Split/ROI/group audit: `../input/final_split_audit.json`, `../input/object_occupancy_audit.json`, `../input/group_replication_audit.json`.
