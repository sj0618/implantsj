# unseen15 metric 결과 해석

## 관측된 결과

예시 결과:

- accuracy = 0.1917
- macro_f1 = 0.1466
- weighted_f1 = 0.2266

27개 seen label로 학습한 metric space를 15개 unseen label에 그대로 적용하면 이 정도로 낮은 값이 나올 수 있다.

## 왜 낮은 것이 정상에 가까운가

### 1. unseen 15라벨은 학습 목표에 없었다

SupCon과 Triplet 기반 metric learning은 학습에 사용한 seen class가 잘 분리되도록 embedding space를 만든다.

27개 seen class는 잘 분리되도록 학습됐지만, 학습에 등장하지 않은 15개 unseen class를 각각 구분하도록 학습된 것은 아니다.

따라서 unseen 15라벨 분류 성능이 낮아도 이상하지 않다.

### 2. 이 모델은 진짜 zero-shot 모델이 아니다

현재 ViT + SupCon / Triplet 구조는 CLIP처럼 text embedding을 이용해 class name 자체로 zero-shot 분류를 수행하는 구조가 아니다.

이 구조는 `stella`, `iu`, `Simpleline` 같은 이름을 보고 분류하지 않으며, 학습되지 않은 새 class를 의미적으로 자동 분리하지도 않는다.

그래서 이 실험을 "진짜 zero-shot classification"으로 해석하면 안 된다.

### 3. prototype이 없으면 unseen class를 제대로 맞추기 어렵다

Metric learning 분류는 class prototype이나 support embedding이 있어야 안정적이다.

그런데 unseen 15 class는 학습 중 prototype이 형성되지 않았기 때문에, 평가 시점에 다음 문제가 발생한다.

- unseen 이미지를 seen class embedding space 안으로 강제로 끌어다 붙이게 됨
- 또는 소수의 support sample로 만든 prototype 품질이 매우 불안정함

따라서 정확도가 낮게 나온다.

### 4. unseen class는 표본 수가 적고 불균형할 가능성이 크다

표본 수가 적은 라벨은 다음 문제가 크다.

- 이미지 다양성 부족
- 대표 특징 불안정
- prototype 품질 저하
- 노이즈 영향 증가

특히 샘플 수가 적은 label은 몇 장 차이만으로도 metric 분포가 크게 흔들릴 수 있다.

### 5. 시각적으로 비슷한 class는 seen space에서 흡수되기 쉽다

임플란트 구조가 유사한 class들은 unseen일수록 기존 seen class 중 가장 가까운 cluster로 흡수되기 쉽다.

낮은 정확도는 단순 실패라기보다 각 unseen class가 어떤 seen class 근처에 놓이는지 보여주는 분석 결과일 가능성이 높다.

## 가장 가능성 높은 원인 우선순위

| 원인 | 가능성 |
|---|---|
| 15라벨을 학습하지 않았기 때문에 class 분리 불가 | 매우 높음 |
| 구조 자체가 진짜 zero-shot 모델이 아님 | 매우 높음 |
| unseen prototype 품질 부족 | 높음 |
| 데이터 수 부족 / 클래스 불균형 | 높음 |
| 27 seen과 15 unseen 사이 형태적 유사성 | 높음 |
| 코드 오류 | 가능성은 있으나 우선순위 낮음 |

## 이 결과를 어떻게 해석해야 하는가

이 실험은 정확한 unseen class 분류 성능 평가보다,

> unseen sample이 기존 seen class embedding space에서 어디로 매핑되는지 확인하는 분석 실험

으로 해석하는 것이 더 타당하다.

Accuracy 하나보다 아래 항목을 함께 봐야 한다.

- true unseen label별 nearest seen label 분포
- cosine similarity 평균
- 특정 unseen label이 어떤 seen label에 집중적으로 매핑되는지
- confusion matrix에서 구조적으로 비슷한 군집이 있는지

## 평가 방식별 해석 포인트

### 경우 A. unseen 15 이미지를 seen 27 prototype에 붙인 경우

이 경우 낮은 accuracy는 거의 예상 가능한 결과다.

이 설정에서는 accuracy보다 아래 분석이 더 중요하다.

- unseen label → nearest seen label 매핑표
- unseen label별 similarity score 분포
- embedding 시각화(t-SNE / UMAP)

### 경우 B. unseen 15 내부에서 few-shot prototype을 만든 경우

이 경우는 zero-shot보다 few-shot metric classification에 가깝다.

그래도 낮다면 아래를 본다.

- prototype 생성 샘플 수 부족
- embedding normalize 문제
- margin / temperature 설정 문제
- crop 품질 문제
- class 간 시각적 특징 차이 부족

## 보고서용 문장

아래 문장은 그대로 보고서에 넣을 수 있다.

> 27개 seen class로 학습한 metric learning 모델을 이용해 15개 unseen class에 대해 zero-shot에 준하는 평가를 수행한 결과, unseen class 분류 성능은 낮게 나타났다. 이는 SupCon 및 Triplet 기반 metric learning 모델이 학습된 seen class 간 embedding 분리를 목표로 하며, 학습되지 않은 class의 독립적인 decision boundary나 안정적인 prototype을 형성하지 못하기 때문이다. 따라서 본 실험은 unseen class의 정확한 분류 성능 평가보다는, unseen sample이 seen class embedding space에서 어떤 class와 유사하게 매핑되는지 분석하는 용도로 해석하는 것이 타당하다.

## 실무 결론

1. 현재 수치는 비정상이라기보다 실험 설정상 자연스러운 결과에 가깝다.
2. 이 결과만으로 metric model이 고장났다고 결론내리면 안 된다.
3. Unseen15 평가는 분류 accuracy보다 mapping 분석으로 해석하는 것이 맞다.
4. 현재 우선 작업은 unseen15 학습이 아니라 전체 데이터 13feature 추가다.
