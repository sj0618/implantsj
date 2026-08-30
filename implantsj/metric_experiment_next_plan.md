# 다음 실험 계획: Metric Learning 기반 전이 성능 개선

## 1. 실험 방향

Plain ViT의 직접 분류 성능은 높았지만, 새로운 라벨로 전이하는 metric 실험에서는 성능과 클래스별 균형이 충분하지 않았다. 다음 실험은 직접 분류 정확도를 더 올리기보다 새로운 target 라벨에서도 잘 분리되는 representation과 안정적인 prototype 분류 방법을 찾는 데 초점을 둔다.

현재 비교 기준은 다음과 같다.

| 기준 | 조건 | Test Accuracy | Macro-F1 | 용도 |
|---|---|---:|---:|---|
| 직접 분류 기준 | 3-label Plain ViT | 94.50% | 77.00% | 전용 분류 모델의 참고값 |
| metric transfer 기준 | uncropped 7→3 ViT-Base | 82.76% | 64.16% | 현재 metric 전이의 best 기준 |
| 최신 closed-set 기준 | 27-label ClosedFT | 95.43% | 92.67% | 최신 분류 성능 참고값 |

※ 위 결과는 라벨 수, 학습 방식, 평가 protocol이 다르므로 동일 조건의 우열을 의미하지 않는다.

## 2. 단계별 실험 계획

| 단계 | 우선순위 | 실험 내용 | 고정할 조건 | 주요 평가 지표 | 판단 기준 |
|---|---:|---|---|---|---|
| 1. 입력 개선 | 1 | uncropped와 ROI crop 비교; 224와 384/512 해상도 비교; single-crop과 multi-crop 비교 | ViT-Base, 동일 loss, 동일 prototype 방식 | Accuracy, Macro-F1, 클래스별 F1 | ROI crop 또는 고해상도에서 Macro-F1이 개선되는지 확인 |
| 2. Representation 비교 | 2 | CE, SupCon, Triplet representation 비교; encoder frozen, partial fine-tuning 비교; backbone과 projection head 비교 | 1단계에서 가장 좋은 입력 조건 | Macro-F1, minority F1, intra/inter-class distance | 학습하지 않은 Base representation이 계속 우세한지 확인 |
| 3. Sampling·loss 개선 | 3 | group-balanced sampler, class-aware batch, hard positive/negative mining 적용; SupCon temperature와 Triplet margin 소규모 ablation | 최적 입력과 representation 고정 | Macro-F1, minority F1, per-class recall | Accuracy만 오르지 않고 소수 클래스 성능도 함께 오르는지 확인 |
| 4. Prototype 개선 | 4 | 전체 평균 prototype, group-balanced prototype, medoid prototype 비교; support sample 수 변화; cosine과 Euclidean 비교 | encoder와 target split 고정 | Macro-F1, prototype별 분산, confusion matrix | prototype 선택에 따른 성능 변동이 줄어드는지 확인 |
| 5. Target adaptation | 5 | frozen zero-shot, 소량 target fine-tuning, episodic metric training 비교; 1/5/10-shot 조건 검토 | 최적 입력·loss·prototype 조합 | Accuracy, Macro-F1, seed별 평균±표준편차 | 적은 target 샘플에서도 안정적으로 개선되는지 확인 |
| 6. 최종 재현성 검증 | 6 | 동일 설정을 3개 이상 seed로 반복; 고정 test set과 동일한 평가 코드 사용 | 최종 후보 2~3개 | 평균±표준편차, per-class F1, confusion matrix | 단일 seed 우연이 아닌 일관된 개선인지 확인 |

## 3. 권장 실행 순서

| 실험 라운드 | 실행할 조합 | 목적 |
|---|---|---|
| 1차 | uncropped vs ROI crop × ViT-Base × 기존 prototype | 현재 best representation에서 입력 문제의 영향을 먼저 확인 |
| 2차 | 최적 입력 × CE/SupCon/Triplet × frozen/partial fine-tuning | metric representation과 fine-tuning 전략 비교 |
| 3차 | 최적 representation × balanced sampling × prototype ablation | 낮은 minority F1과 prototype 변동 문제 개선 |
| 4차 | 최종 후보 × 3개 이상 seed × 고정 target test set | 최종 성능과 재현성 검증 |

## 4. 평가 및 성공 기준

| 항목 | 적용 기준 |
|---|---|
| 1차 목표 | 현재 metric transfer 기준인 Macro-F1 64.16%를 안정적으로 초과 |
| 2차 목표 | Accuracy 82.76% 수준을 유지하면서 Macro-F1과 minority F1을 함께 개선 |
| 주 지표 | Macro-F1을 1순위로 사용하고 Accuracy는 보조 지표로 보고 |
| 필수 분석 | 클래스별 F1/recall, confusion matrix, prototype별 sample 수와 변동성 기록 |
| 재현성 | 최소 3개 seed의 평균±표준편차 보고 |
| 비교 원칙 | 입력, split, prototype 계산 방식, seed를 고정하고 한 번에 한 요소만 변경 |

## 5. 예상 결론과 다음 선택

| 관찰 결과 | 다음 선택 |
|---|---|
| ROI crop·고해상도에서 성능이 개선됨 | metric 실험의 기본 입력을 ROI 기반으로 고정하고 loss/sampling 실험 진행 |
| ViT-Base가 fine-tuning보다 계속 우수함 | encoder는 frozen으로 두고 prototype 및 sampling 개선에 집중 |
| Accuracy만 높고 Macro-F1이 낮음 | class-balanced sampling, hard negative, prototype 구성 개선을 우선 적용 |
| prototype 종류에 따라 결과 변동이 큼 | group-balanced prototype과 support 수 고정 후 반복 검증 |
| 모든 metric 설정이 직접 분류보다 낮음 | metric learning은 unseen label 전이와 보조 분석 목적에 한정하고, 직접 분류를 최종 baseline으로 유지 |

## 실행 요약

먼저 ROI crop과 고해상도 입력에서 implant 정보가 충분히 보이는지 확인한다. 그다음 CE, SupCon, Triplet representation을 같은 조건에서 비교하고 group-balanced sampling과 prototype 설계를 검증한다. 마지막으로 target adaptation과 다중 seed 반복 실험을 거쳐 새로운 라벨 전이에서 Macro-F1이 안정적으로 좋아지는 metric pipeline을 고른다.
