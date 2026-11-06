# Euclidean · Cosine · Mahalanobis 비교

동일한 uncropped 7→3 평가에서 거리 함수만 비교했다. 값은 `Accuracy / Macro-F1`(%)이다.

| 모델 | Prototype | Euclidean Acc./F1 | Cosine Acc./F1 | Mahalanobis Acc./F1 | 판정 |
|---|---|---:|---:|---:|---|
| ViT-Base | GT | 80.17 / 56.26 | 82.76 / 64.16 | **90.52 / 70.57** | Mahalanobis 우세 |
| ViT-Base | Mean | 81.90 / 64.12 | 81.90 / 63.99 | **93.10 / 79.69** | Mahalanobis 우세 |
| ViT-Base | SS | 47.41 / 37.75 | 46.55 / 37.11 | **84.48 / 67.46** | Mahalanobis 크게 우세 |
| 7-label CE | GT | 70.69 / 50.33 | 70.69 / 45.44 | 계산 대기 | 원 checkpoint 필요 |
| 7-label CE | Mean | 72.41 / 50.14 | 72.41 / 49.90 | 계산 대기 | 원 checkpoint 필요 |
| 7-label CE | SS | 43.10 / 31.79 | 43.10 / 31.57 | 계산 대기 | 원 checkpoint 필요 |

## 7→3 unseen-label Mahalanobis 검증 (cropped, 별도 실험)

7개 제조사로 학습한 CE backbone을 고정하고, 학습에 없던 `Bego·Bicon·ITI` 3개 제조사를 Mean prototype으로 분류했다.

| Source 모델 | Target | Prototype | Euclidean Acc./F1 | Cosine Acc./F1 | Mahalanobis Acc./F1 | 판정 |
|---|---|---|---:|---:|---:|---|
| 7-label CE, 30 epoch (best epoch 11) | unseen 3-label test, n=446 | Mean | 73.09 / 48.74 | 73.32 / 48.83 | **95.07 / 84.19** | Mahalanobis 우세 |
| 동일 모델, exact duplicate 5장 제외 | unseen 3-label test, n=441 | Mean | 72.79 / 48.66 | 73.02 / 48.76 | **95.01 / 84.18** | 결과 유지 |

- Cosine 대비 Mahalanobis는 Accuracy `+21.75%p`, Macro-F1 `+35.36%p` 개선됐다.
- exact duplicate 5장을 제외해도 Accuracy `95.01%`, Macro-F1 `84.18%`로 사실상 동일했다.
- 7-label 학습에 없던 target label에서도 Mahalanobis metric 평가가 잘 작동했다.
- 단, target train label로 Mean prototype과 covariance를 만들었으므로 strict zero-shot 결과는 아니다. 또한 이 모델은 CE로 학습하고 metric 방식으로 평가한 모델이지, metric loss로 학습한 모델은 아니다.

### 발표에서 사용할 문장

> 7개 제조사 학습에 포함되지 않은 3개 제조사로 전이했을 때, Euclidean/Cosine은 Accuracy 약 73%, Macro-F1 약 49%였으나 Mahalanobis는 95.07% / 84.19%를 기록했다. 중복 영상을 제거해도 성능이 유지되어 unseen target-label 환경에서 거리 metric 선택의 효과를 확인했다. 다만 target train label을 사용한 gallery 기반 평가이므로 zero-shot으로 해석하지 않는다.

## 계산 조건

- 기존 표의 query, GT/Mean/SS prototype 및 seed 2026을 그대로 유지했다.
- 모델별 target support 1,839장의 class-mean residual만으로 pooled covariance를 추정했다.
- Ledoit-Wolf shrinkage를 적용했으며 target test 116장은 covariance와 prototype 학습에 사용하지 않았다.
- ViT-Base의 기존 Euclidean/Cosine 6개 값은 원표와 소수점 둘째 자리까지 재현한 후 Mahalanobis를 계산했다.
- SciPy 직접 계산과 최대 절대오차는 `4.97e-14`였다.

## 현재 상태

ViT-Base 3개 Mahalanobis 셀과 별도 cropped 7→3 unseen-label 결과는 확정값이다. 첫 표의 7-label CE 원본 epoch-12 checkpoint는 중지된 RunPod 작업공간에만 있어, 서버 접근 전까지 동일 uncropped 프로토콜의 Mahalanobis 값은 확정할 수 없다. 별도 cropped 실험값을 첫 표에 섞지 않고 독립 표로 제시했다.
