# 원격 실험 확인 기록 — 2026-09-30

## 확인 범위

- 확인 시각: 2026-09-30 약 12:28–12:46 KST. Windows App의 `localhost:3389` 세션 안에서 `user07-gpu` 터미널을 읽었다.
- 확인 방법: 원격 CSV·JSON·실행 코드·로그·Docker 종료 상태를 화면으로 확인하고 집계 수치를 전사했다. 원격 파일 다운로드나 데이터 이동은 하지 않았다.
- 아래 값은 새로 계산한 평가 결과가 아니라 저장된 결과다. 전체/seen/unseen 표본 수와 정확도의 가중 평균을 교차 확인했다. CSV는 소수점 6자리로 기록했다.
- 보호된 원본 이미지·개별 환자 정보·체크포인트는 복사하지 않았다. 원격 학습을 시작하거나 중단하지 않았다.
- 과거 실험 전체의 세부 기록은 [9/29 보고서](/Users/yunseojune/TRIA_LAB/implant_project/implantsj/docs/reports/remote_experiments_20260929.md)를 함께 참조한다. 아래 최신 확인이 과거의 ‘결과 미확인’ 상태를 갱신한다.

## 1. Gap-SupCon: 학습 완료, 개선 효과는 미검증

원격 경로: `/home/ubuntu/aihub_run_new/outputs/experiment_records/gap-supcon-b256-20260929/`

### 완료 근거와 설정

- `launcher_status.json`: complete, exit code 0. Docker의 `gap-supcon-grid-b256-20260929`도 Exited(0).
- `status.json`, `selected.json`, `selection.csv`, `comparison.csv`, `final_gap/`의 체크포인트·예측 CSV·geometry CSV를 확인했다.
- 실제 launcher는 `driver.py`를 실행했다. `driver_with_baseline.py` 파일이 있다는 사실만으로 baseline이 실행됐다고 볼 수 없다.
- `comparison.csv`는 `final_gap`의 3개 분류기 × 3개 그룹 = 9행뿐이다. 후보 선택 표도 Gap 4개 조합뿐이다.
- `gap-supcon-baseline-stopped-20260929`는 Exited(143). 같은 조건의 SupCon 단독 결과는 확인되지 않았다.
- DINOv2 base / 입력 224 / projection 128 / batch 256 / 클래스당 16장 / 5 epoch / seed 42. 일반 사전학습 가중치에서 시작하며 임플란트 fine-tuning checkpoint는 초기값으로 지정하지 않았다.
- 후보는 margin 0.05 또는 0.1, weight 0.01 또는 0.05. source에서 따로 둔 5개 클래스의 macro accuracy로 선택했다. 선택값은 **margin 0.05, weight 0.01**.
- OAS 공분산은 최종 closed27 학습 임베딩의 클래스 잔차에서만 추정한다. GalCov와 공분산 적합 범위가 다르다.

### 최종 평가

단위: %. 전체 16,570장 = seen 15,349장 + unseen 1,221장.

| 분류기 | 전체 Acc | seen Acc | unseen Acc | unseen Macro-F1 |
|---|---:|---:|---:|---:|
| Cosine prototype | 84.97 | 89.00 | **34.32** | 31.14 |
| Euclidean prototype | 84.99 | 88.98 | 34.89 | 31.16 |
| Pooled OAS Mahalanobis | 87.67 | 90.63 | **50.45** | 48.33 |

같은 모델에서 공분산 분류기가 unseen 정확도를 높였다. 이 비교는 **분류기의 차이**를 보여준다. Gap을 추가한 학습의 효과는 동일한 SupCon 대조군이 있어야 판단할 수 있다.

### 클래스 거리

실제 `src/metric/gap.py`, `evaluation.py`의 정의:

1. 표본 임베딩을 L2 정규화한다.
2. support 평균을 중심 `p_c`로 사용하며, 평균 중심을 다시 정규화하지 않는다.
3. query의 중심 기준 RMS 거리 `r_c = sqrt(mean ‖z_q - p_c‖²)`를 반지름으로 둔다.
4. 최근접 유효 간격은 `g_c = min_j(‖p_c-p_j‖ - r_c - r_j)`이다.
5. loss는 `mean(max(0, margin-g_c)²)`이다. 학습 support/query는 서로 겹치지 않는다.

| 최종 평가 그룹 | RMS 반지름 평균 | 최근접 유효 간격 평균 | 간격이 음수인 클래스 비율 |
|---|---:|---:|---:|
| Seen | 0.1910 | -0.1038 | 74.07% |
| Unseen | **0.4302** | **-0.5763** | **100%** |

unseen 반지름은 seen의 약 **2.25배**다. 이 측정은 클래스 분리가 부족하다는 진단을 뒷받침한다. 다만 RMS 구의 간격이 음수라는 뜻이며, 모든 query가 오분류됐다는 뜻은 아니다. support와 query의 중심 이동도 RMS 반지름에 포함된다.

선택 단계의 마지막 기록은 train radius 0.19296 / gap +0.34259, heldout radius 0.42797 / gap -0.42414다. **학습 클래스에서 벌어진 간격이 새로운 클래스까지 이어지지 않는 현상**을 확인할 필요가 있다. 이 값은 선택 단계의 기록이며 최종 `final_gap`의 epoch 기록으로 혼동하지 않는다.

선택 기록의 SupCon loss는 3.62429, gap loss는 0.01642다. 가중치를 곱한 gap 항은 약 0.000164로, SupCon 항의 약 0.0045%다. 이것은 손실의 크기 비교이며 기울기 기여도는 별도로 측정해야 한다. optimizer update 288, AMP skip 0도 확인했다.

위 geometry는 정규화된 임베딩의 Euclidean 측정이다. CSV의 cosine·Euclidean·OAS 행에 같은 geometry가 반복되므로 OAS 공간에서 다시 측정한 거리로 해석하지 않는다.

## 2. ViT / Band-ViT GalCov: 완료된 후속 결과

원격 경로: `/home/ubuntu/aihub_run_new/outputs/metric/closed27_open15/vit_band_galcov_20260928/`

- `followup-vit-band-galcov-20260928`: Exited(0). queue에도 succeeded / exit 0, 로그에도 grid 완료 문구가 있다.
- `grid_summary_closed27_open15.csv`에서 OAS 조건을 읽었다. 아래는 전체 거리 조합의 최적값 탐색이 아니라 **OAS 조건의 비교**다.
- `galcov.sh`의 `embedding_loss=[none]`, `embedding_source=backbone`. 저장된 `/home/ubuntu/aihub_run_new/checkpoints/backbones/20260918-170637-6ea9b483/{vit,band-vit}_full_closed27_open15_c001.pt`를 불러온다. 이번 grid에서 metric loss를 추가 학습했다는 뜻은 아니다. checkpoint의 원래 학습 이력 전체는 이번에 재검증하지 않았다.
- 평가 코드의 `fit_z`는 closed train + open train gallery를 합친 임베딩이다. 미학습 클래스 support 데이터도 공분산 추정에 포함된다.
- Kmeans의 `5`는 클래스당 중심 수다. support 5장만 사용하는 strict 5-shot과 다르다.

| 모델 | Prototype | 전체 Acc | seen Acc | unseen Acc | 전체 Macro-F1 원문 값* |
|---|---|---:|---:|---:|---:|
| ViT | Mean | 93.54 | 94.66 | 79.52 | 81.85 |
| ViT | Kmeans 5 | **93.77** | 94.46 | **85.18** | 82.99 |
| Band-ViT | Mean | 93.37 | 94.37 | 80.84 | 79.65 |
| Band-ViT | Kmeans 5 | 93.72 | 94.48 | **84.19** | 82.55 |

`*` `evaluate.py`의 `_metric_block`은 전체 클래스 목록에 unknown을 추가해 `compute_metrics`에 전달한다. seen/open 부분집합에서도 전체 목록을 그대로 전달한다. 따라서 **open_macro_f1을 15개 unseen 클래스만의 Macro-F1로 읽으면 안 된다**. 별도 CSV에는 이 범위를 적고 부분집합 F1은 옮기지 않았다. `compute_metrics` 내부의 모든 처리는 이번에 재검증하지 않아 표의 F1은 원문 집계값으로만 보존한다.

코드에 gallery-query SHA256 중복을 제외한 평가 기능은 있다. 이번에 읽은 grid 표가 그 중복 제외 결과라는 근거는 확인하지 못했다. support/query 이미지 및 환자 단위 중복 검증 후 운영 후보를 결정해야 한다.

## 3. 41클래스 OAS 비교

원격 원문: `/home/ubuntu/aihub_run_new/outputs/experiment_records/followup-20260928-2228/oas_comparison.csv`

이 계열은 42→41 라벨 병합 실험으로, 26 seen + 15 unseen이다. protocol 문자열이 `closed27_open15`여도 실제 클래스 수는 별도로 확인해야 한다. 전체 표본 수는 동일하다.

| 모델 | Mean unseen Acc | Kmeans unseen Acc | Kmeans 전체 Acc |
|---|---:|---:|---:|
| Image-only `new_41class` | 71.17 | **79.03** | 84.86 |
| Attribute token 5 epoch | 71.99 | 78.13 | 85.43 |
| Attribute token 20 epoch, patience 3 | 63.80 | 73.05 | 90.29 |
| Image-predicted attribute fusion | 38.74 | **37.51** | 82.65 |

현재 표에서 속성 융합이 unseen 정확도를 개선했다는 근거는 없다. 모델별 학습 조건과 속성 제공 방식이 다르므로 속성 사용만을 원인으로 단정하지 않는다.

`overall_macro_f1`은 open/closed 행에도 반복된다. `macro_f1_41class`는 다른 열이다. 이들을 unseen F1로 잘못 사용하지 않았다. 원문의 Wilson 신뢰구간은 고정 test 이미지 기준이며 학습 seed·환자 반복의 불확실성을 뜻하지 않는다.

## 4. 실행 상태와 빠진 결과

후속 폴더: `/home/ubuntu/aihub_run_new/outputs/experiment_records/followup-20260928-2228/`

| 작업 | 확인 상태 | 남은 사항 |
|---|---|---|
| GalCov | exit 0, 요약 CSV 있음 | 중복 제외 성능, 클래스별 F1, checkpoint 원래 학습 범위 확인 |
| `heldout` | queue exit 0; best valid_macro_f1 0.5594 / epoch 1; test 16,570장 임베딩 저장 로그 | 속성 학습 validation 값이며 최종 식별 성능과 구분 |
| `heldout-frozen` | queue exit 0; best valid_macro_f1 0.5634 / epoch 3 | **test_manifest.csv 누락으로 임베딩 저장 건너뜀**. 최종 평가가 완성됐다고 볼 수 없음 |
| `vit-supcon5` | 12:44 KST 컨테이너 `76b975c1d1cf`, 이름 `fu-vit-supcon5-0928`, Up 2 hours | 최종 지표 미확인. Gap의 중단된 DINOv2 baseline과 다른 작업 |
| heldout seed43/44, frozen seed43/44 | 실험 폴더와 설정 있음 | 반복 학습 완료·종합 결과는 미확인 |
| `dinov2-ft5` | 준비 폴더 있음 | 완료 결과 미확인 |

`verification_status.json`은 9/28 23:16의 과거 파일이다. 현재 queue나 결과 상태의 증거로 대신 쓰지 않았다. 이번 작업은 현재까지 저장된 결과를 정리했으며 진행 중인 실험 완료까지 기다리는 작업은 수행하지 않았다.

## 원문 추적용 파일

- Gap: 위 실행 폴더의 `protocol.json`, `launch_config.json`, `launcher.log`, `launcher_status.json`, `status.json`, `selected.json`, `selection.csv`, `comparison.csv`, `final_gap/test_geometry.csv`, `src/metric/gap.py`, `evaluation.py`.
- GalCov: 위 후속 폴더의 `galcov.sh`, `galcov.log`, `queue.log`; grid 폴더의 `grid_summary_closed27_open15.csv`; `/home/ubuntu/aihub_run_new/outputs/experiment_records/20260924-galcov-5ep/evaluate.py`.
- 41클래스: 후속 폴더의 `oas_comparison.csv`, `oas_verification.json`.
- 상태: 후속 폴더의 `heldout.log`, `heldout-frozen.log`, `queue.log` 및 실시간 Docker 상태.
- 수치 목록: [remote_experiments.csv](/Users/yunseojune/.codex/worktrees/0550/implantsj/docs/experiment_audit_20260930/remote_experiments.csv). 63행은 고유 학습 63회가 아니라 모델·분류기·평가 그룹별 지표와 상태 기록이다.
