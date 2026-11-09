# closed27_open15 metric 평가 결과 정리 (2026-09-24 12:05 KST 기준)

출처: `aihub_run_new/outputs/metric/closed27_open15/*/*/eval/metrics_test_*.json` 167개. test split만 정리했습니다. 수치는 JSON의 `overall/closed/open.accuracy`를 소수 4자리로 반올림한 값입니다.
조합 이름은 `{prototype}_{distance}[_{covariance}]` 형식입니다.

## 실험별 조건

| 실험 (결과 폴더) | 백본 시작점 | 학습 | 공분산 추정 데이터 |
|---|---|---|---|
| `backbone_eval/closed27_open15_dinov2_none` | 파인튜닝 체크포인트 `20260918-170637-6ea9b483/dinov2_full_closed27_open15_c001.pt` | 없음 (분류 헤드만 제거) | closed27 train |
| `dinov2_full_finetune_5ep/*` (supcon, arcface, triplet) | timm pretrained (`checkpoint=null`) | metric loss, 백본 포함 전체 학습, 5 epoch | closed27 train |
| `head_only/closed27_open15_vit_supcon` | 기록 미확인 | head만 학습 (supcon) | closed27 train |

- `dinov2_full_finetune_5ep`는 2026-09-23 12:45 UTC에 시작된 실행입니다. 설정은 `experiment_records/20260923-1655-dinov2-full-four-loss-5ep/launch.sh`에 있습니다.
- 이 실행에서 loss 4종 중 **covaware_supcon은 test metrics 파일이 없습니다**. 학습 폴더는 있지만 평가 결과가 없어서, 중간에 멈췄거나 실패한 것으로 보입니다. 원인은 확인하지 않았습니다.
- `full_finetune`과 `full_finetune_40e_p5` 폴더에는 `metrics_test_*.json`이 없어서 표에서 뺐습니다.
- `backbone_eval`은 mean 프로토타입 10개와 ss 프로토타입 5개만 있습니다. medoid와 kmeans_multi는 없고, vit·band-vit·vit_supcon 백본 결과도 없습니다. 실행 컨테이너가 중간에 사라진 것으로 보입니다.

## 실험별 최고 조합

| 실험 | 전체 Acc 최고 조합 | 전체 | Closed | Open | Open Acc 최고 조합 | 전체 | Closed | Open |
|---|---|---|---|---|---|---|---|---|
| DINOv2 파인튜닝 백본, 헤드 제거 (backbone_eval) | mean_mahalanobis_ledoit_wolf | 0.8856 | 0.9347 | 0.2686 | mean_mahalanobis_ledoit_wolf | 0.8856 | 0.9347 | 0.2686 |
| DINOv2 timm→metric 5ep supcon | mean_mahalanobis_ledoit_wolf | 0.8838 | 0.9113 | 0.5389 | kmeans_multi_mahalanobis_ledoit_wolf | 0.8439 | 0.8516 | 0.7469 |
| DINOv2 timm→metric 5ep arcface | mean_mahalanobis_oas | 0.8604 | 0.8810 | 0.6011 | kmeans_multi_mahalanobis_class_balanced | 0.7564 | 0.7625 | 0.6798 |
| DINOv2 timm→metric 5ep triplet | mean_mahalanobis_ledoit_wolf | 0.1244 | 0.1150 | 0.2432 | mean_mahalanobis_rda | 0.0319 | 0.0000 | 0.4333 |
| ViT head-only supcon | mean_mahalanobis_ledoit_wolf | 0.5566 | 0.5580 | 0.5389 | kmeans_multi_mahalanobis_ledoit_wolf | 0.5438 | 0.5361 | 0.6396 |

## 주요 조합 비교

| 실험 | 조합 | 전체 Acc | Closed Acc | Open Acc |
|---|---|---|---|---|
| DINOv2 파인튜닝 백본, 헤드 제거 (backbone_eval) | mean_cosine | 0.8790 | 0.9294 | 0.2457 |
| DINOv2 파인튜닝 백본, 헤드 제거 (backbone_eval) | mean_euclidean | 0.8771 | 0.9289 | 0.2260 |
| DINOv2 파인튜닝 백본, 헤드 제거 (backbone_eval) | mean_mahalanobis_ledoit_wolf | 0.8856 | 0.9347 | 0.2686 |
| DINOv2 파인튜닝 백본, 헤드 제거 (backbone_eval) | mean_mahalanobis_class_balanced | 0.8775 | 0.9309 | 0.2056 |
| DINOv2 파인튜닝 백본, 헤드 제거 (backbone_eval) | ss_mahalanobis_ledoit_wolf | 0.8740 | 0.9334 | 0.1278 |
| DINOv2 timm→metric 5ep supcon | mean_cosine | 0.8652 | 0.9076 | 0.3325 |
| DINOv2 timm→metric 5ep supcon | mean_euclidean | 0.8652 | 0.9080 | 0.3268 |
| DINOv2 timm→metric 5ep supcon | mean_mahalanobis_ledoit_wolf | 0.8838 | 0.9113 | 0.5389 |
| DINOv2 timm→metric 5ep supcon | mean_mahalanobis_class_balanced | 0.8683 | 0.9085 | 0.3628 |
| DINOv2 timm→metric 5ep supcon | kmeans_multi_mahalanobis_ledoit_wolf | 0.8439 | 0.8516 | 0.7469 |
| DINOv2 timm→metric 5ep supcon | ss_mahalanobis_ledoit_wolf | 0.8750 | 0.9084 | 0.4554 |
| DINOv2 timm→metric 5ep arcface | mean_cosine | 0.8355 | 0.8699 | 0.4029 |
| DINOv2 timm→metric 5ep arcface | mean_euclidean | 0.8361 | 0.8709 | 0.3997 |
| DINOv2 timm→metric 5ep arcface | mean_mahalanobis_ledoit_wolf | 0.8603 | 0.8809 | 0.6011 |
| DINOv2 timm→metric 5ep arcface | mean_mahalanobis_class_balanced | 0.8513 | 0.8735 | 0.5717 |
| DINOv2 timm→metric 5ep arcface | kmeans_multi_mahalanobis_ledoit_wolf | 0.7642 | 0.7713 | 0.6757 |
| DINOv2 timm→metric 5ep arcface | ss_mahalanobis_ledoit_wolf | 0.8489 | 0.8782 | 0.4808 |
| DINOv2 timm→metric 5ep triplet | mean_cosine | 0.0139 | 0.0102 | 0.0598 |
| DINOv2 timm→metric 5ep triplet | mean_euclidean | 0.0139 | 0.0102 | 0.0598 |
| DINOv2 timm→metric 5ep triplet | mean_mahalanobis_ledoit_wolf | 0.1244 | 0.1150 | 0.2432 |
| DINOv2 timm→metric 5ep triplet | mean_mahalanobis_class_balanced | 0.0894 | 0.0816 | 0.1876 |
| DINOv2 timm→metric 5ep triplet | kmeans_multi_mahalanobis_ledoit_wolf | 0.1133 | 0.1060 | 0.2048 |
| DINOv2 timm→metric 5ep triplet | ss_mahalanobis_ledoit_wolf | 0.0509 | 0.0513 | 0.0459 |
| ViT head-only supcon | mean_cosine | 0.4800 | 0.4889 | 0.3677 |
| ViT head-only supcon | mean_euclidean | 0.4781 | 0.4871 | 0.3645 |
| ViT head-only supcon | mean_mahalanobis_ledoit_wolf | 0.5566 | 0.5580 | 0.5389 |
| ViT head-only supcon | mean_mahalanobis_class_balanced | 0.5323 | 0.5328 | 0.5258 |
| ViT head-only supcon | kmeans_multi_mahalanobis_ledoit_wolf | 0.5438 | 0.5361 | 0.6396 |
| ViT head-only supcon | ss_mahalanobis_ledoit_wolf | 0.3043 | 0.3065 | 0.2776 |

## 관찰

1. **Mahalanobis(ledoit_wolf ≈ oas)가 대부분의 실험에서 가장 좋습니다.** DINOv2 supcon 5ep 기준으로 mean_cosine과 비교하면 Open Acc가 0.3325에서 0.5389로 20.6%p 오르고, Closed Acc는 0.9076에서 0.9113으로 거의 그대로입니다.
2. **Open 정확도는 metric 학습 쪽이 백본만 쓴 경우보다 높습니다.** 백본만 쓴 경우(backbone_eval) mean_maha_LW의 Open Acc는 0.2686입니다. timm 시작 supcon 5ep는 0.5389, arcface 5ep는 0.6011입니다. 반면 Closed Acc는 백본만 쓴 쪽이 0.9347로 가장 높습니다.
3. **kmeans_multi(클래스당 prototype 5개) + Mahalanobis가 Open Acc 최고치입니다.** supcon에서 0.7469입니다. 대신 Closed Acc가 0.8516으로 약 6%p 내려갑니다.
4. **triplet은 학습이 무너졌습니다.** 전체 Acc가 0.01~0.12 수준입니다. 5 epoch 설정에서 학습에 실패한 것으로 보이고, 원인은 확인하지 않았습니다.
5. **rda 공분산은 모든 실험에서 비정상입니다.** Open Acc가 0.0이거나 Closed Acc가 0.0이고, 전체 Acc는 0.006~0.52입니다. 수치가 정상 범위를 벗어나서 구현 문제로 의심됩니다. 확인하지 않았습니다.
6. **ViT head-only supcon은 전체 정확도가 0.56 이하입니다.** 백본을 얼리고 head만 학습하면 DINOv2 전체 학습보다 크게 낮습니다.

## 진행 중 (2026-09-24 11:58 KST 시작)

- 공분산을 **프로토타입(gallery) 이미지 = closed train + open train**으로 추정하도록 바꾼 평가 코드로 DINOv2 loss 4종 × 5 epoch를 다시 돌리고 있습니다.
  - A: timm 시작 → `outputs/metric/closed27_open15/dinov2_galcov_timm_5ep`. 실행 중이고 컨테이너는 `mgal-timm`입니다.
  - B: 파인튜닝 체크포인트 시작 → `outputs/metric/closed27_open15/dinov2_galcov_ft_5ep`. A가 끝나면 자동으로 시작하고, 컨테이너는 `mgal-ft`입니다.
  - 코드와 스크립트는 `outputs/experiment_records/20260924-galcov-5ep/`에 있습니다. `evaluate.py` 129번째 줄에 `fit_z, fit_y = gallery_z, gallery_y`를 추가했고, 실행 스크립트는 `a.sh`, `b.sh`, `queue.log`입니다.
  - 시간을 줄이려고 t-SNE 시각화는 껐습니다 (`eval.visualize.enabled=false`).
  - 이 방식은 open15 train 이미지를 공분산 추정에 쓰므로 **target-supervised gallery 평가**입니다. 기존 closed-only 공분산 결과와 같은 선상에서 비교하면 안 됩니다.
- 2번 실험(ViT를 1번의 최고 loss로 학습)은 A와 B 결과가 나온 뒤 진행합니다.

## 부록: 전체 조합 수치 (test)

### DINOv2 파인튜닝 백본, 헤드 제거 (backbone_eval)

| 조합 | 전체 Acc | Closed Acc | Open Acc |
|---|---|---|---|
| mean_centered_angular | 0.8789 | 0.9337 | 0.1892 |
| mean_cosine | 0.879 | 0.9294 | 0.2457 |
| mean_cosine_spherical | 0.8787 | 0.9287 | 0.2498 |
| mean_euclidean | 0.8771 | 0.9289 | 0.226 |
| mean_fractional | 0.8778 | 0.9341 | 0.1695 |
| mean_mahalanobis_class_balanced | 0.8775 | 0.9309 | 0.2056 |
| mean_mahalanobis_diagonal | 0.8763 | 0.9299 | 0.2031 |
| mean_mahalanobis_ledoit_wolf | 0.8856 | 0.9347 | 0.2686 |
| mean_mahalanobis_oas | 0.8856 | 0.9347 | 0.2686 |
| mean_mahalanobis_rda | 0.0844 | 0.0911 | 0.0 |
| ss_cosine | 0.8722 | 0.9318 | 0.1229 |
| ss_cosine_spherical | 0.8728 | 0.9318 | 0.131 |
| ss_euclidean | 0.8704 | 0.9292 | 0.131 |
| ss_mahalanobis_ledoit_wolf | 0.874 | 0.9334 | 0.1278 |
| ss_mahalanobis_oas | 0.874 | 0.9334 | 0.1269 |

### DINOv2 timm→metric 5ep supcon

| 조합 | 전체 Acc | Closed Acc | Open Acc |
|---|---|---|---|
| kmeans_multi_centered_angular | 0.8363 | 0.8535 | 0.6192 |
| kmeans_multi_cosine | 0.8142 | 0.8383 | 0.5111 |
| kmeans_multi_euclidean | 0.8161 | 0.8399 | 0.5168 |
| kmeans_multi_fractional | 0.8345 | 0.8511 | 0.6257 |
| kmeans_multi_mahalanobis_class_balanced | 0.8233 | 0.8388 | 0.6282 |
| kmeans_multi_mahalanobis_diagonal | 0.8156 | 0.8393 | 0.5176 |
| kmeans_multi_mahalanobis_ledoit_wolf | 0.8439 | 0.8516 | 0.7469 |
| kmeans_multi_mahalanobis_oas | 0.8441 | 0.8518 | 0.7461 |
| kmeans_multi_mahalanobis_rda | 0.4607 | 0.4973 | 0.0 |
| mean_centered_angular | 0.8674 | 0.909 | 0.344 |
| mean_cosine | 0.8652 | 0.9076 | 0.3325 |
| mean_cosine_spherical | 0.8652 | 0.9076 | 0.3325 |
| mean_euclidean | 0.8652 | 0.908 | 0.3268 |
| mean_fractional | 0.8698 | 0.9093 | 0.3735 |
| mean_mahalanobis_class_balanced | 0.8683 | 0.9085 | 0.3628 |
| mean_mahalanobis_diagonal | 0.8654 | 0.9078 | 0.3325 |
| mean_mahalanobis_ledoit_wolf | 0.8838 | 0.9113 | 0.5389 |
| mean_mahalanobis_oas | 0.8837 | 0.911 | 0.5405 |
| mean_mahalanobis_rda | 0.5193 | 0.5606 | 0.0 |
| medoid_centered_angular | 0.8701 | 0.9057 | 0.4218 |
| medoid_cosine | 0.8683 | 0.9008 | 0.4586 |
| medoid_euclidean | 0.8683 | 0.9008 | 0.4586 |
| medoid_fractional | 0.8735 | 0.9062 | 0.4627 |
| medoid_mahalanobis_class_balanced | 0.8722 | 0.9061 | 0.4464 |
| medoid_mahalanobis_diagonal | 0.8684 | 0.9014 | 0.4529 |
| medoid_mahalanobis_ledoit_wolf | 0.8675 | 0.9046 | 0.4005 |
| medoid_mahalanobis_oas | 0.8674 | 0.9046 | 0.3997 |
| medoid_mahalanobis_rda | 0.5033 | 0.5433 | 0.0 |
| ss_centered_angular | 0.8669 | 0.9068 | 0.3653 |
| ss_cosine | 0.8667 | 0.9059 | 0.3751 |
| ss_cosine_spherical | 0.8667 | 0.9059 | 0.3751 |
| ss_euclidean | 0.8669 | 0.9058 | 0.3784 |
| ss_fractional | 0.8686 | 0.9063 | 0.3939 |
| ss_mahalanobis_class_balanced | 0.8685 | 0.9085 | 0.3661 |
| ss_mahalanobis_diagonal | 0.8663 | 0.9057 | 0.3718 |
| ss_mahalanobis_ledoit_wolf | 0.875 | 0.9084 | 0.4554 |
| ss_mahalanobis_oas | 0.8749 | 0.9083 | 0.4545 |
| ss_mahalanobis_rda | 0.4875 | 0.5263 | 0.0 |

### DINOv2 timm→metric 5ep arcface

| 조합 | 전체 Acc | Closed Acc | Open Acc |
|---|---|---|---|
| kmeans_multi_centered_angular | 0.76 | 0.7709 | 0.6233 |
| kmeans_multi_cosine | 0.7547 | 0.7694 | 0.5692 |
| kmeans_multi_euclidean | 0.7614 | 0.7758 | 0.5807 |
| kmeans_multi_fractional | 0.7608 | 0.7722 | 0.6175 |
| kmeans_multi_mahalanobis_class_balanced | 0.7564 | 0.7625 | 0.6798 |
| kmeans_multi_mahalanobis_diagonal | 0.7599 | 0.7742 | 0.5799 |
| kmeans_multi_mahalanobis_ledoit_wolf | 0.7642 | 0.7713 | 0.6757 |
| kmeans_multi_mahalanobis_oas | 0.7643 | 0.7714 | 0.6757 |
| kmeans_multi_mahalanobis_rda | 0.2672 | 0.2885 | 0.0 |
| mean_centered_angular | 0.8448 | 0.8733 | 0.4857 |
| mean_cosine | 0.8355 | 0.8699 | 0.4029 |
| mean_cosine_spherical | 0.8355 | 0.8699 | 0.4029 |
| mean_euclidean | 0.8361 | 0.8709 | 0.3997 |
| mean_fractional | 0.845 | 0.8737 | 0.484 |
| mean_mahalanobis_class_balanced | 0.8513 | 0.8735 | 0.5717 |
| mean_mahalanobis_diagonal | 0.8368 | 0.8714 | 0.4021 |
| mean_mahalanobis_ledoit_wolf | 0.8603 | 0.8809 | 0.6011 |
| mean_mahalanobis_oas | 0.8604 | 0.881 | 0.6011 |
| mean_mahalanobis_rda | 0.3466 | 0.3742 | 0.0 |
| medoid_centered_angular | 0.8314 | 0.8606 | 0.4652 |
| medoid_cosine | 0.8196 | 0.8515 | 0.4177 |
| medoid_euclidean | 0.8196 | 0.8515 | 0.4177 |
| medoid_fractional | 0.8307 | 0.8617 | 0.4406 |
| medoid_mahalanobis_class_balanced | 0.8369 | 0.8694 | 0.4283 |
| medoid_mahalanobis_diagonal | 0.8199 | 0.8513 | 0.4259 |
| medoid_mahalanobis_ledoit_wolf | 0.8432 | 0.8788 | 0.3964 |
| medoid_mahalanobis_oas | 0.8431 | 0.8788 | 0.3948 |
| medoid_mahalanobis_rda | 0.3447 | 0.3721 | 0.0 |
| ss_centered_angular | 0.8541 | 0.8868 | 0.4439 |
| ss_cosine | 0.8141 | 0.8462 | 0.4103 |
| ss_cosine_spherical | 0.8141 | 0.8462 | 0.4103 |
| ss_euclidean | 0.8185 | 0.8501 | 0.4218 |
| ss_fractional | 0.8527 | 0.8866 | 0.4275 |
| ss_mahalanobis_class_balanced | 0.8483 | 0.8779 | 0.4758 |
| ss_mahalanobis_diagonal | 0.8508 | 0.8844 | 0.4283 |
| ss_mahalanobis_ledoit_wolf | 0.8489 | 0.8782 | 0.4808 |
| ss_mahalanobis_oas | 0.8488 | 0.878 | 0.4808 |
| ss_mahalanobis_rda | 0.3392 | 0.3662 | 0.0 |

### DINOv2 timm→metric 5ep triplet

| 조합 | 전체 Acc | Closed Acc | Open Acc |
|---|---|---|---|
| kmeans_multi_centered_angular | 0.0566 | 0.0536 | 0.0942 |
| kmeans_multi_cosine | 0.0287 | 0.0265 | 0.0565 |
| kmeans_multi_euclidean | 0.0287 | 0.0265 | 0.0565 |
| kmeans_multi_fractional | 0.0489 | 0.0457 | 0.0901 |
| kmeans_multi_mahalanobis_class_balanced | 0.0849 | 0.0791 | 0.1572 |
| kmeans_multi_mahalanobis_diagonal | 0.0291 | 0.027 | 0.0557 |
| kmeans_multi_mahalanobis_ledoit_wolf | 0.1133 | 0.106 | 0.2048 |
| kmeans_multi_mahalanobis_oas | 0.1131 | 0.1059 | 0.2031 |
| kmeans_multi_mahalanobis_rda | 0.0298 | 0.0 | 0.4046 |
| mean_centered_angular | 0.0488 | 0.0453 | 0.0925 |
| mean_cosine | 0.0139 | 0.0102 | 0.0598 |
| mean_cosine_spherical | 0.0139 | 0.0102 | 0.0598 |
| mean_euclidean | 0.0139 | 0.0102 | 0.0598 |
| mean_fractional | 0.0252 | 0.0203 | 0.0868 |
| mean_mahalanobis_class_balanced | 0.0894 | 0.0816 | 0.1876 |
| mean_mahalanobis_diagonal | 0.0148 | 0.0111 | 0.0614 |
| mean_mahalanobis_ledoit_wolf | 0.1244 | 0.115 | 0.2432 |
| mean_mahalanobis_oas | 0.1243 | 0.1149 | 0.2424 |
| mean_mahalanobis_rda | 0.0319 | 0.0 | 0.4333 |
| medoid_centered_angular | 0.024 | 0.023 | 0.0369 |
| medoid_cosine | 0.022 | 0.0212 | 0.0319 |
| medoid_euclidean | 0.022 | 0.0212 | 0.0319 |
| medoid_fractional | 0.0292 | 0.0278 | 0.0467 |
| medoid_mahalanobis_class_balanced | 0.0362 | 0.0351 | 0.0508 |
| medoid_mahalanobis_diagonal | 0.0244 | 0.0234 | 0.0377 |
| medoid_mahalanobis_ledoit_wolf | 0.0237 | 0.0228 | 0.0352 |
| medoid_mahalanobis_oas | 0.0237 | 0.0227 | 0.0352 |
| medoid_mahalanobis_rda | 0.0071 | 0.0 | 0.0966 |
| ss_centered_angular | 0.0358 | 0.0371 | 0.0197 |
| ss_cosine | 0.0315 | 0.0325 | 0.0188 |
| ss_cosine_spherical | 0.0315 | 0.0325 | 0.0188 |
| ss_euclidean | 0.0315 | 0.0325 | 0.0188 |
| ss_fractional | 0.0346 | 0.0356 | 0.0213 |
| ss_mahalanobis_class_balanced | 0.0436 | 0.0444 | 0.0336 |
| ss_mahalanobis_diagonal | 0.0317 | 0.0324 | 0.0229 |
| ss_mahalanobis_ledoit_wolf | 0.0509 | 0.0513 | 0.0459 |
| ss_mahalanobis_oas | 0.051 | 0.0514 | 0.0459 |
| ss_mahalanobis_rda | 0.0113 | 0.0 | 0.1532 |

### ViT head-only supcon

| 조합 | 전체 Acc | Closed Acc | Open Acc |
|---|---|---|---|
| kmeans_multi_centered_angular | 0.4946 | 0.497 | 0.4636 |
| kmeans_multi_cosine | 0.5028 | 0.5049 | 0.4767 |
| kmeans_multi_euclidean | 0.5024 | 0.5041 | 0.4808 |
| kmeans_multi_fractional | 0.5377 | 0.5391 | 0.5201 |
| kmeans_multi_mahalanobis_class_balanced | 0.5365 | 0.5295 | 0.6241 |
| kmeans_multi_mahalanobis_diagonal | 0.5039 | 0.5058 | 0.4799 |
| kmeans_multi_mahalanobis_ledoit_wolf | 0.5438 | 0.5361 | 0.6396 |
| kmeans_multi_mahalanobis_oas | 0.5438 | 0.5361 | 0.6396 |
| kmeans_multi_mahalanobis_rda | 0.0064 | 0.0069 | 0.0 |
| mean_centered_angular | 0.4352 | 0.4497 | 0.2539 |
| mean_cosine | 0.48 | 0.4889 | 0.3677 |
| mean_cosine_spherical | 0.48 | 0.4889 | 0.3677 |
| mean_euclidean | 0.4781 | 0.4871 | 0.3645 |
| mean_fractional | 0.4987 | 0.5092 | 0.3661 |
| mean_mahalanobis_class_balanced | 0.5323 | 0.5328 | 0.5258 |
| mean_mahalanobis_diagonal | 0.481 | 0.4907 | 0.3595 |
| mean_mahalanobis_ledoit_wolf | 0.5566 | 0.558 | 0.5389 |
| mean_mahalanobis_oas | 0.5566 | 0.558 | 0.5389 |
| mean_mahalanobis_rda | 0.0064 | 0.0069 | 0.0 |
| medoid_centered_angular | 0.3498 | 0.3636 | 0.1761 |
| medoid_cosine | 0.4604 | 0.468 | 0.3636 |
| medoid_euclidean | 0.4604 | 0.468 | 0.3636 |
| medoid_fractional | 0.4123 | 0.4252 | 0.2498 |
| medoid_mahalanobis_class_balanced | 0.268 | 0.2814 | 0.0991 |
| medoid_mahalanobis_diagonal | 0.4586 | 0.4656 | 0.371 |
| medoid_mahalanobis_ledoit_wolf | 0.2474 | 0.2594 | 0.0975 |
| medoid_mahalanobis_oas | 0.2474 | 0.2594 | 0.0975 |
| medoid_mahalanobis_rda | 0.0064 | 0.0069 | 0.0 |
| ss_centered_angular | 0.3083 | 0.3158 | 0.2138 |
| ss_cosine | 0.3485 | 0.3499 | 0.3309 |
| ss_cosine_spherical | 0.3485 | 0.3499 | 0.3309 |
| ss_euclidean | 0.3495 | 0.351 | 0.3309 |
| ss_fractional | 0.356 | 0.3599 | 0.3071 |
| ss_mahalanobis_class_balanced | 0.316 | 0.3183 | 0.2867 |
| ss_mahalanobis_diagonal | 0.3502 | 0.3517 | 0.3301 |
| ss_mahalanobis_ledoit_wolf | 0.3043 | 0.3065 | 0.2776 |
| ss_mahalanobis_oas | 0.3043 | 0.3065 | 0.2776 |
| ss_mahalanobis_rda | 0.0064 | 0.0069 | 0.0 |