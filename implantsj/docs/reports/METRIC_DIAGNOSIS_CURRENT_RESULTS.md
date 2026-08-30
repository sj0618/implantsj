# Metric Learning 진단 실험: 현재 결과

기준 시점: 2026-07-13 09:00 KST

범위: 저장된 학습 기록, 체크포인트, 결과 파일, t-SNE 이미지, 원격 화면 캡처

저장되지 않은 체크포인트와 평가 수치는 추정하지 않았다.

## 1. 요약

- ViT-ClosedFT는 closed-set 27라벨로 10 epoch 학습을 완료했다. 최종 test accuracy는 0.9543, test macro-F1은 0.9267이다.
- ViT-ClosedFT-SupCon은 확인된 학습 기록이 3/10 epoch까지다. 최신 test accuracy는 0.8765, test macro-F1은 0.7930이지만, 10 epoch ClosedFT와 직접 비교해 SupCon의 효과로 해석할 수 없다.
- 재사용한 ViT-FullFT는 42라벨 학습 기록을 가진다. 최고 validation accuracy는 0.9514, 최고 validation macro-F1은 0.9106이다. per-epoch test metric은 기록되지 않았다.
- 프로토타입 t-SNE는 ClosedFT, ClosedFT-SupCon, FullFT 체크포인트에 대해 생성됐다. Base와 FullFT-SupCon은 체크포인트가 없어 PASS 처리됐다.
- 각 표현 공간에서 저장된 프로토타입 좌표는 110점이다: GT 26점 + Mean 42점 + SS 42점.
- GT-prototype은 영문명 매핑 기준으로 closed 20/27, open 6/15만 매핑됐다. Mean과 SS는 closed 27/27, open 15/15 전체를 포함한다.
- 현재 결과만으로는 Base 대비 fine-tuning 저하 여부, SupCon의 최종 효과, Euclidean 대 cosine 우열, GT/Mean/SS의 성능 우열을 확정할 수 없다. 시각화는 확보됐지만 동일 조건의 분류 성능 표가 아직 없다.

## 2. 실험 구성

### 데이터 분할

- closed-set: 27라벨
- open-set: 15라벨
- closed manifest: `/home/ubuntu/workspace/data/manifests/metric-closed27-corrected.csv` - 141,265행
- open manifest: `/home/ubuntu/workspace/data/manifests/metric-open15-corrected.csv` - 7,865행
- full manifest: `/home/ubuntu/workspace/data/manifests/models_vit.csv`
- GT manifest: `/home/ubuntu/workspace/data/gt.csv` - 241행, 27개 GT 폴더
- SS 설정: 라벨당 5개 학습 샘플(`ss_per_label=5`), seed 42

open-set 15라벨은 다음과 같다.

`Simpleline`, `exfeel internal`, `55 III`, `Biohorizon external`, `TSV`, `5 clean tapered`, `GS III`, `screw vent`, `atlas internal`, `stella`, `3i`, `ankylos`, `iu`, `replace select`, `core1`

### 모델 및 체크포인트

| Model | Checkpoint | 현재 처리 |
|---|---|---|
| ViT Base | 없음 | PASS |
| ViT-ClosedFT | `/home/ubuntu/workspace/outputs/20260711_202711_metricdiag_fresh_closedft27_e10/checkpoints/best.pt` | 학습·곡선·프로토타입 t-SNE 확인 |
| ViT-ClosedFT-SupCon | `/home/ubuntu/workspace/outputs/20260712_043644_metricdiag_fresh_closedft_supcon27_e10/checkpoints/best.pt` | 부분 학습 기록·곡선·프로토타입 t-SNE 확인 |
| ViT-FullFT | `/home/ubuntu/workspace/outputs/20260701_060811_models_vit_42class/checkpoints/best.pt` | 기존 체크포인트·곡선·프로토타입 t-SNE 확인 |
| ViT-FullFT-SupCon | 없음 | PASS |

### 프로토타입 정의

- GT-prototype: Yonsei/AI Hub 기준 이미지를 영문 라벨명으로 매핑해 사용
- Mean-prototype: 라벨별 전체 training embedding 평균
- SS-prototype: 라벨별 training embedding 일부를 선택해 평균; 현재 설정은 라벨당 5개

## 3. 학습 결과

| Model | Labels | 기록 상태 | 기준 epoch | Validation accuracy | Validation macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---|---:|---:|---:|---:|---:|
| ViT Base | 42 | 체크포인트 없음 | - | - | - | - | - |
| ViT-ClosedFT | 27 closed | **10/10 완료** | 10 | **0.9494** | **0.9104** | **0.9543** | **0.9267** |
| ViT-ClosedFT-SupCon | 27 closed | **3/10까지만 기록 확인** | 3 | 0.8734 | 0.7919 | 0.8765 | 0.7930 |
| ViT-FullFT | 42 full | 기존 54-epoch 기록 재사용 | 42 F1 / 54 acc | **0.9514** | **0.9106** | 미기록 | 미기록 |
| ViT-FullFT-SupCon | 42 full | 체크포인트 없음 | - | - | - | - | - |

### 3.1 ViT-ClosedFT: 10 epoch 완료

![ViT-ClosedFT learning curves](images/vit-closedft-curves.png)

각 패널은 metric 변화가 독립적으로 보이도록 서로 다른 y축 범위를 사용한다.

| Epoch | Train loss | Valid acc | Valid macro-F1 | Test acc | Test macro-F1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0761 | 0.8980 | 0.8257 | 0.8918 | 0.8279 |
| 2 | 0.8836 | 0.9084 | 0.8433 | 0.9120 | 0.8480 |
| 3 | 0.8388 | 0.9098 | 0.8487 | 0.9109 | 0.8483 |
| 4 | 0.8039 | 0.9230 | 0.8742 | 0.9308 | 0.8834 |
| 5 | 0.7743 | 0.9318 | 0.8841 | 0.9350 | 0.8896 |
| 6 | 0.7448 | 0.9291 | 0.8805 | 0.9369 | 0.8951 |
| 7 | 0.7179 | 0.9427 | 0.9018 | 0.9459 | 0.9095 |
| 8 | 0.6942 | 0.9447 | 0.9043 | 0.9512 | 0.9161 |
| 9 | 0.6756 | 0.9485 | 0.9083 | 0.9531 | 0.9213 |
| 10 | **0.6656** | **0.9494** | **0.9104** | **0.9543** | **0.9267** |

Epoch 1→10에서 train loss는 0.4105 감소, validation accuracy는 5.14%p 증가, test macro-F1은 9.88%p 증가했다.

### 3.2 ViT-ClosedFT-SupCon: 확인된 3 epoch

![ViT-ClosedFT-SupCon partial learning curves](images/vit-closedft-supcon-partial-curves.png)

| Epoch | Train loss | Valid acc | Valid macro-F1 | Test acc | Test macro-F1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.4605 | 0.8510 | 0.7568 | 0.8513 | 0.7595 |
| 2 | 1.3177 | 0.8732 | 0.7795 | **0.8791** | 0.7918 |
| 3 | **1.2811** | **0.8734** | **0.7919** | 0.8765 | **0.7930** |

현재 수치는 3 epoch 결과다. ClosedFT의 10 epoch 결과보다 낮지만, 학습 길이가 다르므로 SupCon 자체가 성능을 저하시킨다는 결론으로 사용하지 않는다.

### 3.3 ViT-FullFT: 기존 54-epoch 기록

![ViT-FullFT selected validation epochs](images/vit-fullft-selected-epochs.png)

| Selection | Epoch | Train loss | Valid accuracy | Valid macro-F1 |
|---|---:|---:|---:|---:|
| Initial | 1 | 1.1785 | 0.8899 | 0.7105 |
| Best validation macro-F1 | 42 | 0.7116 | 0.9444 | **0.9106** |
| Best/final validation accuracy | 54 | 0.6948 | **0.9514** | 0.8995 |

FullFT의 `history.csv`에는 per-epoch test accuracy와 test macro-F1이 없다. 따라서 이 기록만으로 open/closed test 성능을 보고하지 않는다.

## 4. t-SNE 및 프로토타입 결과

### 4.1 일반 classifier t-SNE: 42라벨

![Classifier ViT t-SNE, 42 labels](images/classifier-tsne-42-labels-screen.png)

화면에서는 42라벨 점들이 몇 개의 영역으로 나뉘지만 각 영역 안에서 여러 라벨이 섞인다. 캡처에 체크포인트와 manifest 경로가 표시되지 않아 특정 모델의 정량 근거로는 사용할 수 없다.

### 4.2 GT 기반 metric embedding t-SNE

![Metric ViT GT cosine-trained embedding t-SNE](images/metric-gt-cosine-screen.png)

GT/cosine-trained embedding 화면에서는 일부 라벨이 국소적으로 모이지만 여러 영역에서 라벨이 섞인다. 체크포인트 경로와 Euclidean 대응 결과가 없어 이 그림만으로 cosine의 우위를 판단할 수 없다.

### 4.3 체크포인트별 프로토타입 t-SNE 생성 상태

| Model | Representation | 상태 | 저장 좌표 수 | GT coverage | Mean coverage | SS coverage |
|---|---|---|---:|---:|---:|---:|
| ViT-ClosedFT | common backbone | ok | **110** | closed 20/27, open 6/15 | closed 27/27, open 15/15 | closed 27/27, open 15/15 |
| ViT-ClosedFT-SupCon | common backbone | ok | **110** | closed 20/27, open 6/15 | closed 27/27, open 15/15 | closed 27/27, open 15/15 |
| ViT-ClosedFT-SupCon | deployed/projected | ok | **110** | closed 20/27, open 6/15 | closed 27/27, open 15/15 | closed 27/27, open 15/15 |
| ViT-FullFT | common backbone | ok | **110** | closed 20/27, open 6/15 | closed 27/27, open 15/15 | closed 27/27, open 15/15 |
| ViT Base | - | PASS | - | - | - | - |
| ViT-FullFT-SupCon | - | PASS | - | - | - | - |

좌표 수 110은 `GT 26 + Mean 42 + SS 42`로 구성된다. GT coverage가 26/42인 이유는 현재 영문명 매핑으로 연결된 GT 라벨이 closed 20개와 open 6개이기 때문이다. Mean과 SS는 train manifest에서 직접 생성되므로 42라벨 전체를 포함한다.

ClosedFT의 SS-prototype 결과 화면은 다음과 같다.

![ClosedFT SS prototype t-SNE screen](images/closedft-ss-prototype-screen.jpeg)

왼쪽은 closed-set, 오른쪽은 open-set 프로토타입 좌표다. SS는 라벨당 5개 샘플의 평균을 사용한다. 두 패널은 같은 축 범위를 사용한다. 로컬 보고서에 직접 포함된 prototype 캡처는 이 ClosedFT SS 화면뿐이다. 나머지 완료 상태는 `progress.json`의 상태, 좌표 수, coverage로 기록했다.

### 4.4 원격 결과 파일 확인 화면

ClosedFT 결과 파일은 `status=ok`이며 해당 체크포인트 경로가 기록돼 있다.

![ClosedFT progress screen](images/closedft-progress-screen.jpeg)

FullFT 결과 파일도 `status=ok`이며 기존 42-class `best.pt`를 사용했다.

![FullFT progress screen](images/fullft-progress-screen.jpeg)

## 5. 연구 질문별 판단

| 질문 | 현재 답변 | 근거와 한계 |
|---|---|---|
| Base 대비 fine-tuning이 representation을 저하시키는가? | **미판정** | ViT Base 체크포인트가 없어 동일 manifest 조건의 Base embedding을 만들지 못했다. |
| SupCon이 feature representativeness에 도움이 되는가? | **미판정** | ClosedFT-SupCon의 backbone/projected t-SNE는 생성됐지만, 확인된 학습 기록은 3 epoch이고 동일 조건의 최종 accuracy/F1 비교가 없다. |
| full-set 학습이 도움이 되는가? | **제한적 수치만 확인** | 표시된 값 기준 FullFT의 최고 validation accuracy는 ClosedFT보다 **+0.20%p**, 최고 macro-F1은 **+0.02%p** 높다. 그러나 42라벨·54 epoch와 27라벨·10 epoch 비교이므로 "FullFT가 최고"라고 확정할 수 없다. |
| Euclidean과 cosine 중 무엇이 좋은가? | **미판정** | cosine-trained t-SNE 화면은 있으나 동일 prototype·split에서 두 거리의 accuracy/F1 결과표가 없다. |
| GT, Mean, SS 중 무엇이 좋은가? | **성능 우열 미판정** | 시각화와 coverage는 확보됐다. coverage 기준으로 Mean/SS는 42/42, GT는 26/42지만, 이는 라벨 가용성 차이이며 분류 성능 우열이 아니다. |

### FullFT와 ClosedFT의 validation 참고 비교

| Metric | ClosedFT best | FullFT best | Difference (FullFT − ClosedFT) |
|---|---:|---:|---:|
| Validation accuracy | 0.9494 | 0.9514 | **+0.20%p** |
| Validation macro-F1 | 0.9104 | 0.9106 | **+0.02%p** |

## 6. 실험 단계 진행표

| 단계 | 현재 상태 |
|---|---|
| ClosedFT 10-epoch 학습곡선 | 완료 |
| ClosedFT-SupCon 10-epoch 학습곡선 | 부분: 3 epoch 기록만 확인 |
| FullFT 학습곡선 | 기존 54-epoch 기록 재사용 |
| FullFT-SupCon 학습곡선 | PASS: 체크포인트 없음 |
| 일반 classifier/metric t-SNE | 일부 결과 화면 확보 |
| GT/Mean/SS prototype t-SNE | ClosedFT, ClosedFT-SupCon, FullFT 완료 |
| open/closed별 Euclidean 평가 수치 | 현재 결과 없음 |
| open/closed별 cosine 평가 수치 | 현재 결과 없음 |
| prototype별 최종 accuracy/F1 순위 | 현재 결과 없음 |

## 7. 결과 파일 위치

| Artifact | Remote path |
|---|---|
| ClosedFT history | `/home/ubuntu/workspace/outputs/20260711_202711_metricdiag_fresh_closedft27_e10/metrics/history.csv` |
| ClosedFT-SupCon history | `/home/ubuntu/workspace/outputs/20260712_043644_metricdiag_fresh_closedft_supcon27_e10/metrics/history.csv` |
| FullFT history | `/home/ubuntu/workspace/outputs/20260701_060811_models_vit_42class/metrics/history.csv` |
| ClosedFT prototype t-SNE | `/home/ubuntu/workspace/outputs/prototype-tsne-closedft/prototype_tsne/` |
| ClosedFT prototype progress | `/home/ubuntu/workspace/outputs/prototype-tsne-closedft/progress.json` |
| ClosedFT-SupCon prototype t-SNE | `/home/ubuntu/workspace/outputs/prototype-tsne-closedft-supcon/prototype_tsne/` |
| ClosedFT-SupCon prototype progress | `/home/ubuntu/workspace/outputs/prototype-tsne-closedft-supcon/progress.json` |
| FullFT prototype t-SNE | `/home/ubuntu/workspace/outputs/prototype-tsne-fullft/prototype_tsne/` |
| FullFT prototype progress | `/home/ubuntu/workspace/outputs/prototype-tsne-fullft/progress.json` |

## 8. 결론

확인된 수치 중에서는 ClosedFT 10 epoch의 test accuracy 0.9543과 macro-F1 0.9267이 가장 높다. FullFT의 validation 최고값은 ClosedFT보다 조금 높지만 데이터 범위와 epoch가 달라 우열을 정할 수 없다. 프로토타입 t-SNE는 세 체크포인트에서 생성됐다. Mean과 SS는 42라벨 전체를 포함했고, GT는 영문명 매핑이 된 26라벨만 포함했다. 원인 진단을 마치려면 거리, prototype, open/closed 조건을 맞춘 정량 평가가 필요하다.
