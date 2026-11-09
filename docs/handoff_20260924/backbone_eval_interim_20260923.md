# 파인튜닝 백본 헤드 제거 평가 — 중간 결과 (2026-09-23 14:49 KST 기준)

## 평가 조건

| 항목 | 값 |
|---|---|
| 프로토콜 | closed27_open15 (closed 27클래스 학습, open 15클래스는 학습에 미사용) |
| 모델 | DINOv2 파인튜닝 백본 `checkpoints/backbones/20260918-170637-6ea9b483/dinov2_full_closed27_open15_c001.pt` |
| 임베딩 | 분류 헤드 제거, 백본 출력 사용 (`embedding_source=backbone`, `embedding_loss=none`) |
| 프로토타입 | mean (클래스 평균) |
| 분할 | test |
| 출처 | `aihub_run_new/outputs/metric/closed27_open15/backbone_eval/closed27_open15_dinov2_none/eval/metrics_test_mean_*.json` |

## 결과 (test, prototype=mean)

| 거리 | 전체 Acc | 전체 Macro-F1 | 전체 Weighted-F1 | Closed Acc | Closed Macro-F1 | Closed Weighted-F1 | Open Acc | Open Macro-F1 | Open Weighted-F1 |
|---|---|---|---|---|---|---|---|---|---|
| cosine | 0.8790 | 0.6107 | 0.8707 | 0.9294 | 0.5535 | 0.9347 | 0.2457 | 0.0913 | 0.3710 |
| cosine (spherical mean) | 0.8787 | 0.6113 | 0.8710 | 0.9287 | 0.5536 | 0.9346 | 0.2498 | 0.0922 | 0.3737 |
| euclidean | 0.8771 | 0.5969 | 0.8688 | 0.9289 | 0.5513 | 0.9347 | 0.2260 | 0.0769 | 0.3450 |
| mahalanobis / class_balanced | 0.8775 | 0.5974 | 0.8674 | 0.9309 | 0.5535 | 0.9360 | 0.2056 | 0.0760 | 0.3193 |
| mahalanobis / diagonal | 0.8763 | 0.5903 | 0.8667 | 0.9299 | 0.5521 | 0.9352 | 0.2031 | 0.0682 | 0.3160 |
| **mahalanobis / ledoit_wolf** | **0.8856** | **0.6163** | **0.8763** | **0.9347** | 0.5545 | **0.9382** | **0.2686** | 0.0940 | **0.3987** |
| mahalanobis / oas | **0.8856** | 0.6162 | **0.8763** | **0.9347** | **0.5546** | **0.9382** | **0.2686** | **0.0941** | 0.3986 |

참고(valid, mahalanobis / ledoit_wolf): Closed Acc 0.9285 / Macro-F1 0.5472 / Weighted-F1 0.9325, Open Acc 0.2703 / Macro-F1 0.0959 / Weighted-F1 0.3974.

## 요약

- 가장 좋은 조합은 Mahalanobis(ledoit_wolf ≈ oas)입니다. 전체 88.56%, Closed 93.47%, Open 26.86%이고, 두 공분산 추정 방식의 정확도가 같습니다.
- Closed 정확도는 거리 방식과 상관없이 92.9~93.5%로 차이가 0.6%p 안쪽입니다.
- Open 정확도는 20.3~26.9%로 거리 방식에 따라 차이가 더 큽니다. 학습에 없던 15클래스는 백본 임베딩만으로는 거의 구분하지 못합니다.
- Closed의 Macro-F1(약 0.55)이 정확도(약 0.93)보다 훨씬 낮습니다. 일부 클래스가 크게 틀리고 있거나, Macro-F1을 계산할 때 open 클래스까지 포함했을 수 있습니다. 어느 쪽인지는 아직 확인하지 않았습니다.

## 아직 없는 결과

- DINOv2의 나머지 조합: 거리 fractional·centered_angular, 프로토타입 ss·medoid·kmeans_multi.
- vit, band-vit, vit_supcon 백본: 14:40 확인 시점에 결과 폴더가 없었습니다.
- 클래스별 정답·오답 수: metrics JSON에는 없고 `confusion_*.csv`, `predictions_*.csv`에 있습니다.
