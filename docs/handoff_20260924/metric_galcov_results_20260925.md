# galcov(gallery 공분산) 재평가 최종 결과 (2026-09-25 01:50 KST 기준)

`metric_results_summary_20260924.md`(closed-only 공분산 결과)의 후속 문서입니다.
두 세션의 확인 결과를 합쳤습니다.
- 2026-09-24 17:23 KST: happy-shaw 세션이 서버 JSON 95개를 확인했습니다. 원래 파일은 `happy-shaw` worktree의 `docs/metric_galcov_results_20260924.md`였고, 이 문서로 옮기면서 지웠습니다.
- 2026-09-25 01:45 KST: 이 세션이 서버의 galcov JSON 143개(백본 29, arcface·supcon·triplet 각 38)를 모두 다시 확인했습니다.

표기: 각 칸은 **전체 / Closed(seen) / Open(unseen)** 정확도입니다. H는 seen과 unseen의 조화평균입니다. 모두 test split이고, seed 1개, 분할 1개, 신뢰구간 없음입니다.

## 0. galcov란

- 평가에서 Mahalanobis 거리에 쓰는 **공분산 행렬 Σ를 어떤 이미지로 추정하느냐**만 바꾼 설정입니다.
- 기존 방식(closed-only)은 Σ를 **closed27 train 임베딩**만으로 추정했습니다.
- galcov는 Σ를 **gallery 이미지**로 추정합니다. gallery는 프로토타입을 만드는 데 쓴 이미지로, closed27 train과 open15 train을 합친 것입니다.
- 사용자 지시는 "test 때 train 공분산 말고 test 공분산 사용"이었습니다. 이를 "프로토타입(gallery) 이미지의 공분산"으로 구현했습니다.
- 수정은 한 줄입니다. 위치는 `outputs/experiment_records/20260924-galcov-5ep/evaluate.py` 129행입니다.
  ```python
  fit_z, fit_y = gallery_z, gallery_y
  ```
  원본 `implant_pipeline_new/src/metric/evaluate.py`는 수정하지 않았습니다. 기록 폴더의 사본을 컨테이너에 읽기 전용으로 덮어 마운트했습니다.
- **해석할 때 주의할 점:**
  - open15의 **라벨 있는 train 이미지**가 Σ 추정에 들어갑니다. 그래서 open 클래스를 전혀 모르는 zero-shot 설정이 아닙니다. 등록된 gallery를 쓰는 inductive 설정입니다.
  - test query 이미지는 쓰지 않습니다.
  - 논문에서는 closed-only 결과와 **반드시 구분해서** 보고해야 합니다.
- 대조군: cosine과 euclidean은 Σ를 쓰지 않습니다. 이 둘의 수치가 기존 결과와 소수 4자리까지 같아서, 패치가 Σ 경로만 바꿨다는 것을 확인했습니다.

## 1. 실험 3개의 최종 상태 [확인: `docker ps -a`, `docker logs`]

| 컨테이너 | 내용 | 종료 | 남은 결과 |
|---|---|---|---|
| `mgal-timm` | timm DINOv2 → metric loss 4종 × 5 epoch | Exit 1: **CUDA OOM** (다른 프로세스가 29.17GB 사용 중) | supcon, triplet, arcface 완료(각 38개 조합). **covaware_supcon만 결과 없음** |
| `mgal-ft` | 파인튜닝 DINOv2 → metric 5 epoch | Exit 1: **AssertionError** | 없음 |
| `mgal-bb` | 백본만 쓴 평가(헤드 제거): dinov2, vit, band-vit | Exit 143: 외부 SIGTERM. 로그 마지막 10:27 | dinov2 29개 조합. vit·band-vit는 시작 전에 종료 |

- `mgal-timm`의 OOM은 `mgal-bb`를 같은 GPU에 동시에 띄운 탓입니다. 29.17GB를 쓰던 PID가 bb인지는 확인하지 않았지만 [추정] 그렇습니다.
- `mgal-ft`는 `src/run_metric.py:39`의 `assert kwargs['backbone_registry']['dinov2']['checkpoint'] is None` 때문에 실패했습니다. **현재 코드는 파인튜닝 체크포인트에서 metric 학습을 시작하는 경로를 막고 있습니다.** 다시 돌리려면 이 assert를 고쳐야 합니다(코드 수정이므로 사용자 확인 필요).
- `mgal-bb`에 누가 SIGTERM을 보냈는지는 확인하지 못했습니다.

## 2. 백본만 사용 (DINOv2 파인튜닝 체크포인트, 헤드 제거)

체크포인트: `checkpoints/backbones/20260918-170637-6ea9b483/dinov2_full_closed27_open15_c001.pt`
원자료: 서버 `~/g.txt` (`backbone_eval_galcov/closed27_open15_dinov2_none/eval/metrics_test_*.json` 29개)

| 조합 | closed-only Σ (seen / unseen / H) | galcov (전체 / seen / unseen / H) |
|---|---|---|
| **kmeans_multi + Maha LW** | 없음 | 0.9070 / 0.920 / **0.749** / **0.825** |
| kmeans_multi + euclidean | 없음 | 0.8907 / 0.916 / 0.568 / 0.702 |
| kmeans_multi + cosine | 없음 | 0.8908 / 0.917 / 0.565 / 0.699 |
| **mean + Maha LW** | 0.935 / 0.269 / 0.417 | 0.9092 / 0.926 / **0.702** / **0.798** |
| mean + Maha OAS | 0.935 / 0.269 / 0.417 | 0.9093 / 0.926 / 0.702 / 0.798 |
| mean + Maha class_balanced | 0.931 / 0.206 / 0.337 | 0.8899 / 0.919 / 0.520 / 0.664 |
| mean + Maha diagonal | 0.930 / 0.203 / 0.333 | 0.8765 / 0.930 / 0.207 / 0.339 |
| mean + centered_angular | 0.934 / 0.189 / 0.315 | 0.8842 / 0.933 / 0.274 / 0.423 |
| mean + fractional | 0.934 / 0.170 / 0.287 | 0.8812 / 0.935 / 0.208 / 0.340 |
| mean + cosine (Σ 없음) | 0.929 / 0.246 / 0.389 | 0.8790 / 0.929 / 0.246 / 0.389 (같음) |
| mean + euclidean (Σ 없음) | 0.929 / 0.226 / 0.364 | 0.8771 / 0.929 / 0.226 / 0.364 (같음) |
| medoid + Maha class_balanced | 없음 | 0.8663 / 0.905 / 0.386 / 0.541 |
| medoid + centered_angular | 없음 | 0.8741 / 0.915 / 0.359 / 0.515 |
| medoid + Maha LW | 없음 | 0.8704 / 0.923 / 0.205 / 0.335 |
| ss + Maha LW | 0.933 / 0.128 / 0.225 | 0.8732 / 0.932 / 0.133 / 0.232 |
| ss + Maha class_balanced | 없음 | 0.8758 / 0.931 / 0.187 / 0.311 |
| ss + fractional | 없음 | 0.8704 / 0.936 / 0.044 / 0.084 |

나머지 조합(medoid cosine·euclidean·fractional·diagonal·oas, ss cosine·spherical·euclidean·diagonal·oas·centered_angular)은 모두 unseen이 0.11–0.36입니다.

## 3. timm DINOv2 → metric 학습 5 epoch

원자료: 서버 `~/g2.txt`(143행, 2026-09-25 01:45 확인). 로컬 사본은 같은 폴더의 `metrics_test_galcov.txt`입니다(백본 29개 + arcface 38개 + supcon 38개, triplet은 요약 한 줄).

### 3-1. supcon

| 조합 | closed-only Σ (전체 / seen / unseen) | galcov (전체 / seen / unseen) | galcov H |
|---|---|---|---|
| mean + cosine | 0.8652 / 0.9076 / 0.3325 | 0.8663 / 0.9092 / 0.3268 | 0.481 |
| mean + Maha LW | 0.8838 / 0.9113 / 0.5389 | 0.8976 / 0.9109 / **0.7305** | 0.811 |
| mean + Maha OAS | 0.8837 / 0.9110 / 0.5405 | 0.8979 / 0.9109 / 0.7338 | 0.813 |
| **kmeans_multi + Maha LW** | 0.8439 / 0.8516 / 0.7469 | 0.8542 / 0.8569 / **0.8206** | **0.838** |
| ss + Maha LW | 0.8750 / 0.9084 / 0.4554 | 0.8799 / 0.9065 / 0.5455 | 0.681 |
| medoid + Maha LW | 0.8675 / 0.9046 / 0.4005 | 0.8634 / 0.8956 / 0.4595 | 0.607 |

- 두 실행은 따로 학습한 것입니다. 그래도 Σ를 쓰지 않는 cosine이 0.8652 → 0.8663으로 거의 같아서, 학습 편차는 작습니다.

### 3-2. arcface

| 조합 | closed-only Σ (전체 / seen / unseen) | galcov (전체 / seen / unseen) | galcov H |
|---|---|---|---|
| **mean + Maha LW / OAS** | OAS: 0.8604 / 0.8810 / 0.6011 | 0.8683 / 0.8832 / **0.6814** | **0.769** |
| kmeans_multi + Maha OAS | 기록 없음 | 0.7868 / 0.7904 / 0.7420 | 0.765 |
| mean + Maha class_balanced | 기록 없음 | 0.8534 / 0.8778 / 0.5471 | 0.674 |
| mean + cosine | 기록 없음 | 0.8446 / 0.8778 / 0.4275 | 0.575 |
| ss + Maha LW | 기록 없음 | 0.8582 / 0.8852 / 0.5184 | 0.654 |

- arcface는 supcon보다 seen이 약 3%p, unseen이 약 5%p 낮습니다. **5 epoch 기준으로 loss는 supcon이 가장 좋습니다(H 0.838 대 0.769).**

### 3-3. triplet, rda, covaware

- triplet은 이번에도 붕괴했습니다. 38개 조합 모두 전체 0.015–0.105이고, 가장 좋은 mean + Maha LW도 0.1048 / 0.0959 / 0.2170입니다.
- rda는 여전히 비정상입니다. open 0.0, 전체 0.18(arcface)–0.36(supcon)입니다.
- covaware_supcon은 OOM으로 결과가 없습니다.

## 4. 관찰

1. **Σ를 gallery로 추정하면 unseen 정확도가 크게 오르고, seen은 거의 그대로입니다.**
   - 백본만: mean + Maha LW의 unseen이 0.27 → 0.70, H가 0.417 → 0.798입니다.
   - supcon 5 epoch: 같은 조합의 unseen이 0.54 → 0.73입니다.
   - 2026-09-12 로컬 기록에서 Mahalanobis의 open이 60–70%였던 것은 Σ를 open train까지 포함해 추정했기 때문이라는 추정과 맞습니다.
2. **H 최고는 supcon + kmeans_multi + Maha LW(0.838)입니다.** 백본만 쓴 kmeans_multi + Maha LW(0.825)와 차이는 0.013이고, seed 1개라서 차이가 유의한지는 알 수 없습니다.
3. **전체 정확도 최고는 metric 학습 없이 백본만 쓴 mean + Maha OAS(0.9093)입니다.**
4. Σ 효과는 mean·kmeans_multi 프로토타입과 LW·OAS·class_balanced 공분산에서 컸습니다. ss·medoid 프로토타입과 diagonal 공분산에서는 작았습니다. [추정] ss와 medoid는 Σ를 다른 경로로 계산하거나 프로토타입 위치가 Σ 효과를 상쇄하는 것으로 보입니다. 코드는 확인하지 않았습니다.

## 5. 남은 일

1. 파인튜닝 백본으로 metric 학습(`mgal-ft`): `run_metric.py:39`의 assert를 고칠지 결정이 필요합니다.
2. covaware_supcon을 GPU를 혼자 쓰는 상태에서 다시 돌립니다(arcface는 완료).
3. 원래 요청 2번(ViT를 가장 좋은 loss로 5 epoch)은 **supcon**으로 진행하면 됩니다. H 기준입니다.
4. `mgal-bb`의 vit와 band-vit를 다시 돌립니다. ss 조합 하나에 약 30분이 걸리므로 ss를 빼면 빨라집니다.
5. seed와 분할을 반복하고 신뢰구간을 붙입니다(논문 P0).
