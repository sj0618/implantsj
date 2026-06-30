# implant_python_only_final 발표 요약

## 1. 제목
**치과 임플란트 ROI 이미지 분류 실험 결과 요약**

- 대상 폴더: `/workspace/implant_python_only_final`
- 핵심 모델: ViT 기반 supervised classification, SupCon metric learning, two-stage refinement
- 목적: 임플란트 브랜드/종류 분류 성능을 비교하고, 가장 안정적인 파이프라인을 찾는 것

---

## 2. 프로젝트 한 줄 요약
이 프로젝트는 **치과 임플란트 방사선 ROI 이미지를 여러 라벨 체계(3/7/10-label)로 분류**하는 실험 플랫폼입니다.

주요 실험 축은 다음과 같습니다.

1. **Plain supervised ViT**
2. **SupCon / metric learning ViT**
3. **Two-stage 재분류 파이프라인**
4. **3-label fine-tuning from 10-label checkpoint**

---

## 3. 실험 구성

### 데이터 구성
- `data/3label`
- `data/7label`
- `data/AI Hub Data`
- label별 manifest 기반 학습/검증/테스트 분리

### 주요 산출물
- 학습 결과: `outputs/`
- 요약 문서: `EXPERIMENT_SUMMARY.md`
- 세부 분석: `outputs/20260526_123524_3label_ft_from_10label_vit_base/RESULT_ANALYSIS.md`
- two-stage 결과: `outputs/two_stage_summary*.json`, `outputs/two_stage_metrics.json`

---

## 4. 어떤 시도를 했는가: 과정 중심 정리

이번 프로젝트는 단순히 하나의 모델을 학습한 것이 아니라, **임플란트 분류 문제를 여러 각도에서 검증**하는 방향으로 진행되었습니다.

### 4-1. 라벨 체계 자체를 바꿔가며 비교
- **3-label / 7-label / 10-label**로 문제를 나눠서 학습했습니다.
- 라벨 수를 줄이면 문제는 단순해지지만, 구분 정보가 줄어드는지 확인했습니다.
- 라벨 수를 늘리면 세부 분류는 가능하지만, 클래스 경계가 더 어려워지는지 확인했습니다.

### 4-2. 기본 supervised ViT를 여러 설정으로 반복 실험
- `vit_base_patch16_224`를 중심으로 **plain supervised classification**을 먼저 확립했습니다.
- 학습률, augmentation, class weight 여부를 바꿔가며 안정적인 baseline을 찾았습니다.
- 이 과정에서 “어떤 설정이 test에서 가장 안정적인가”를 먼저 확인했습니다.

### 4-3. class imbalance를 완화하려는 시도
- 소수 클래스 성능을 끌어올리기 위해 **class weights**를 적용했습니다.
- strong augmentation을 넣어 일반화 능력을 높이려 했습니다.
- 다만 validation에서는 좋아 보여도, test에서는 오히려 흔들리는 경우가 있어 과적합 가능성도 같이 점검했습니다.

### 4-4. embedding 기반 분리를 시도
- 단순 분류가 아니라 **SupCon / metric learning**으로 표현 공간을 더 잘 나누는지 실험했습니다.
- 목표는 “분류기 하나”보다 “클래스 간 거리를 잘 만드는 representation”이 실제로 도움이 되는지 확인하는 것이었습니다.
- 결과적으로 accuracy는 높아질 수 있었지만, minority class는 여전히 어려웠습니다.

### 4-5. 혼동되는 클래스만 다시 보는 two-stage 방식 시도
- 모든 샘플을 다시 분류하지 않고, **혼동이 잦은 그룹만 second-stage model로 재분류**하는 방식을 적용했습니다.
- 예: `NORIS/osstem`, `ADIN/MIS`
- 이 접근은 “전체 모델을 바꾸기보다, 헷갈리는 구간만 보정할 수 있는가?”를 보기 위한 실험이었습니다.

### 4-6. 상위 라벨 학습을 하위 라벨로 전이하는 시도
- `10-label` checkpoint를 `3-label`로 fine-tune해서 **transfer learning 효과**를 확인했습니다.
- validation에서는 좋아졌지만 test에서는 유지되지 않았습니다.
- 즉, “학습 데이터에 맞는 개선”과 “실제 일반화 성능”이 다를 수 있다는 점을 확인했습니다.

### 4-7. VLM feature 및 fusion 가능성도 함께 검토
- Azure VLM으로 구조적 feature를 추출하고,
- feature-only 분류, image+VLM fusion, prompt 기반 feature 생성도 실험했습니다.
- 즉, 이미지 단독 모델이 한계가 있는지, 외부 구조 정보를 보조 신호로 쓸 수 있는지도 함께 살펴봤습니다.

---

## 5. 핵심 결과 한눈에 보기

| 실험 | Test Accuracy | Test Macro F1 | 해석 |
|---|---:|---:|---|
| 3-label plain ViT | 0.9450 | 0.7700 | 3-label 기준 대표 baseline |
| 7-label plain ViT | 0.9440 | 0.8829 | plain 모델 중 7-label 대표 성능 |
| 7-label SupCon | 0.9587 | 0.7512 | 정확도는 높지만 minority class가 약함 |
| 10-label plain ViT | **0.9554** | 0.8812 | 전체 supervised baseline 중 가장 안정적 |
| 10-label + NORIS/osstem two-stage | 0.9509 | 0.8786 | 적용은 됐지만 성능 이득은 없음 |
| 10-label → 3-label fine-tune | 0.9174 | 0.7173 | validation은 좋아졌지만 test 일반화 실패 |

---

## 6. 결과 해석

### 6-1. 가장 좋은 supervised baseline
- **10-label plain ViT**가 현재 가장 강한 supervised 기준선입니다.
- test accuracy 0.9554로 최고 수준입니다.
- macro F1도 0.8812로 균형이 좋습니다.

### 6-2. 7-label plain model
- 7-label plain ViT도 충분히 좋은 성능을 보입니다.
- 특히 macro F1 0.8829로 class balance 측면이 좋습니다.

### 6-3. SupCon / metric learning
- SupCon은 **accuracy는 높지만 macro F1이 낮아지는 경향**이 있습니다.
- 즉, 전체 맞춤 수는 늘지만 소수 class의 경계는 아직 어려운 편입니다.

### 6-4. Two-stage 재분류
- NORIS/osstem second-stage는 실제로 적용되었지만,
- current test split에서는 **baseline보다 성능을 개선하지 못했습니다**.
- 따라서 기본 전략으로는 비추천입니다.

### 6-5. 10-label → 3-label fine-tuning
- validation macro F1은 크게 개선됐습니다.
- 하지만 test에서는 오히려 baseline보다 떨어졌습니다.
- 결론적으로 현재 split 기준으로는 **대체 모델로 쓰기 어렵습니다**.

---

## 7. 추가 실험 포인트

### K-fold 요약
- 10-label plain ViT k-fold validation 평균:
  - accuracy ≈ 0.8713
  - macro F1 ≈ 0.8356
- 10-label SupCon k-fold validation 평균:
  - accuracy ≈ 0.9602
  - macro F1 ≈ 0.9054

> 참고: 위 수치는 validation 기준이며, test 결과와는 다를 수 있습니다.

---

## 8. 발표용 결론

### 최종 결론
1. **현재 가장 추천하는 baseline은 10-label plain ViT**입니다.
2. **7-label plain ViT도 안정적 대안**입니다.
3. **SupCon은 분석용 가치가 높지만, 곧바로 대체 모델로 쓰기엔 macro F1이 아쉽습니다.**
4. **Two-stage와 fine-tuning은 현재 split에서는 성능 이득이 제한적**입니다.

### 실무적 추천
- production baseline: `20260507_134640_plain_vit_10label`
- plain 7-label baseline: `20260507_095713_plain_vit_7label`
- embedding/prototype 분석: `20260507_101426_supcon_vit_7label`
- two-stage는 confidence threshold가 추가될 때만 재검토

---

## 9. 발표 마무리 멘트 예시
> 이번 실험에서는 ViT 기반 분류가 가장 안정적이었고, 10-label plain 모델이 현재 기준 최고의 baseline으로 확인되었습니다. Metric learning과 two-stage refinement는 일부 가능성을 보였지만, 현재 test split에서는 baseline을 넘지 못했습니다. 따라서 다음 단계는 baseline 유지와 함께, 소수 클래스 경계 개선을 위한 추가 실험으로 보는 것이 적절합니다.
