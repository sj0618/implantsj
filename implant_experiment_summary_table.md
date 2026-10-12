# `aihub_run_new` 모델 평가 결과표

- 원격 결과 기준 경로: `/home/ubuntu/aihub_run_new/outputs/backbone`
- 집계 기준: 각 실험 폴더의 `summary.json` 내 최종 테스트 지표
- 단위: `%`

## 기존 27개 클래스

| 모델 | Accuracy | Macro-F1 | Weighted-F1 | 결과 폴더 |
|---|---:|---:|---:|---|
| Band-DINOv2 | **95.2375** | **90.2448** | **95.2209** | `20260918-170637-6ea9b483/band-dinov2_full_closed27_open15_c001` |
| ViT | 95.1463 | 90.1225 | 95.1546 | `20260918-170637-6ea9b483/vit_full_closed27_open15_c001` |
| Band-ViT | 94.9443 | 89.8067 | 94.8947 | `20260918-170637-6ea9b483/band-vit_full_closed27_open15_c001` |
| DINOv2 | 93.7716 | 87.5357 | 93.7487 | `20260918-170637-6ea9b483/dinov2_full_closed27_open15_c001` |

## 전체 42개 클래스

| 모델 | Accuracy | Macro-F1 | Weighted-F1 | 결과 폴더 |
|---|---:|---:|---:|---|
| Band-ViT | **95.8857** | 91.9726 | **95.7909** | `20260919-113246-b2be95d6/band-vit_full_full42_c001` |
| Band-DINOv2 | 95.6488 | 91.8831 | 95.6527 | `20260919-113246-b2be95d6/band-dinov2_full_full42_c001` |
| ViT | 95.6301 | 91.9435 | 95.6199 | `20260919-113246-b2be95d6/vit_full_full42_c001` |
| ViT-Large | 95.4375 | **91.9985** | 95.4615 | `20260920-111629-fe6fc822/vit-large_full_full42_c001` |
| DINOv2 | 93.6934 | 89.0007 | 93.6765 | `20260919-113246-b2be95d6/dinov2_full_full42_c001` |

## 미완료 실험

| 학습 범위 | 모델 | 상태 | 결과 폴더 |
|---|---|---|---|
| 전체 42개 클래스 | Band-ViT-Large | `summary.json` 없음 — 최종 테스트 지표 미집계 | `20260920-111629-fe6fc822/band-vit-large_full_full42_c001` |

## 핵심 결과

| 구분 | 최고 모델 | 결과 |
|---|---|---:|
| 기존 27개 클래스 Accuracy | Band-DINOv2 | 95.2375% |
| 기존 27개 클래스 Macro-F1 | Band-DINOv2 | 90.2448% |
| 전체 42개 클래스 Accuracy | Band-ViT | 95.8857% |
| 전체 42개 클래스 Macro-F1 | ViT-Large | 91.9985% |
| 전체 42개 클래스 Weighted-F1 | Band-ViT | 95.7909% |

> 이 표는 새 디렉토리에서 완료가 확인된 백본 분류 실험만 집계한다. 기존 `aihub_run`의 거리 기반 평가 및 이전 실험 결과는 포함하지 않는다.
