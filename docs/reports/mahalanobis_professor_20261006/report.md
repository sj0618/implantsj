# ViT 특징과 OAS 마할라노비스를 이용한 임플란트 분류 결과

작성일: 2026-10-06 · 서버 결과 확인일: 2026-09-30

## 1. 보고 요약

27개 seen 클래스와 15개 unseen 클래스로 구분한 서버 평가에서, 기존 ViT backbone으로 추출한 특징에 **클래스별 평균 prototype과 OAS 마할라노비스 거리**를 적용했다. 기록된 정확도는 **seen 94.66%, unseen 79.52%, 전체 93.54%**였다. 같은 ViT에서 prototype을 클래스당 K-means 중심 5개로 변경한 조건은 **seen 94.46%, unseen 85.18%, 전체 93.77%**였다.

핵심은 **unseen 클래스의 라벨 있는 support로 prototype을 구성하고, seen·unseen gallery를 함께 사용해 공분산을 추정했다는 점**이다. 따라서 이 결과는 support를 등록한 클래스의 식별 성능이며, zero-shot 또는 클래스당 support 5장으로 제한한 결과로 해석하지 않는다.

이 보고서는 당시 서버 화면에서 확인해 저장한 CSV 전사본과 실행 확인 기록을 정리한 것이다. 오늘 서버에 접속해 재평가한 결과나 서버 원본 파일을 내려받아 재계산한 결과는 아니다.

## 2. 실험 설정

| 항목 | 확인된 설정 |
| --- | --- |
| 실험 구분 | `vit_band_galcov_20260928` |
| 클래스 구성 | 총 42개: seen 27개 + unseen 15개 |
| 평가 query | 총 16,570장: seen 15,349장 + unseen 1,221장 |
| 특징 추출 모델 | 저장된 ViT / Band-ViT backbone checkpoint |
| 특징 사용 위치 | `embedding_source=backbone` |
| 이번 평가의 추가 metric 학습 | `embedding_loss=[none]`; 기존 checkpoint를 불러와 평가 |
| 기준 prototype | 클래스별 support 특징의 평균, 중심 1개 |
| 비교 prototype | 클래스당 K-means 중심 5개 |
| 공분산 추정 데이터 | closed train + open train gallery의 특징을 합친 `fit_z` |
| 공분산 추정 방법 | OAS, Oracle Approximating Shrinkage |
| 거리 기반 분류 | OAS 공분산을 이용한 마할라노비스 거리 |
| 완료 근거 | Docker `Exited(0)`, queue `succeeded / exit 0`, grid 완료 로그와 결과 CSV |

**용어:** support는 클래스 중심과 공분산을 구성하는 기준 이미지이고, query는 성능을 측정하는 평가 이미지다. gallery는 이 실험에서 사용하는 support 집합을 뜻한다. `open train`이라는 이름은 unseen support의 데이터 구분이며, 이번 평가에서 backbone 가중치를 추가 학습했다는 뜻은 아니다.

seen/unseen은 당시 평가 프로토콜의 구분이다. 기존 checkpoint가 실제로 어떤 이미지와 클래스로 학습됐는지에 대한 전체 이력은 당시 확인에서 재검증하지 않았다. 엄격한 unseen 일반화 주장에는 학습 manifest 대조가 필요하다.

## 3. 79.52%가 나온 평가 방식

### ① 기존 모델로 이미지 특징 추출

저장된 ViT checkpoint에 support와 query 이미지를 입력해 backbone 특징을 추출한다. 이번 평가에서는 새로운 metric loss로 모델을 추가 학습하는 대신, 기존 특징에 prototype과 거리 기반 분류기를 적용한다.

### ② support 평균으로 클래스별 prototype 구성

각 클래스의 support 특징을 평균 내어 중심 하나를 만든다. unseen 15개 클래스도 라벨 있는 support를 사용해 중심을 구성한다.

예를 들어 어떤 unseen 클래스의 support가 20장이라면, 20개 특징의 평균이 그 클래스의 prototype이 된다. **20장은 설명용 예시이며, 실제 클래스별 support 수는 현재 기록에서 확인되지 않았다.**

`prototype_c = 클래스 c에 속하는 support 특징의 평균`

### ③ seen·unseen gallery로 OAS 공분산 추정

공분산은 특징의 각 방향이 얼마나 변하는지, 특징들 사이에 어떤 상관관계가 있는지를 나타낸다. OAS는 추정 공분산을 단위행렬의 배수 쪽으로 보정하는 shrinkage 방식이다. 고차원 특징의 공분산과 역행렬을 안정적으로 추정하려는 목적이다. [scikit-learn OAS 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html)

이번 기록에서 확인된 공분산 적합 범위는 **seen train gallery + unseen train gallery**다. 공분산 계산 전에 클래스별 평균을 빼는지, 임베딩·prototype을 어떻게 정규화하는지, 실제 shrinkage 계수는 무엇인지까지는 보존된 기록만으로 확정하지 않는다.

### ④ query와 prototype 사이의 마할라노비스 거리 계산

일반적인 거리식은 다음과 같다.

`d²(query, c) = (z_query − prototype_c)ᵀ Σ_OAS⁻¹ (z_query − prototype_c)`

같은 특징 차이라도 공분산에 따라 거리 기여가 달라진다. 거리 기반 분류는 가까운 prototype의 클래스를 선택하는 방식이다. 원격 코드의 후보 클래스 필터와 unknown 거절 임계값 설정은 별도 재확인 대상이다.

### ⑤ seen·unseen query의 정확도를 별도로 집계

- seen 정확도: seen query 15,349장 중 정답 클래스와 일치한 비율.
- unseen 정확도: unseen query 1,221장 중 정답 클래스와 일치한 비율.
- 전체 정확도: query 16,570장 전체에서 정답 클래스와 일치한 비율.

unseen query는 전체의 약 7.37%이므로, 전체 정확도만으로 unseen 성능을 평가하지 않는다. 79.52%는 클래스별 정확도의 단순 평균이 아니라 unseen 이미지 기준 정확도다.

## 4. seen·unseen 결과

정확도 단위: %. 아래 네 조건 모두 OAS 마할라노비스를 사용했다. 표시 수치는 당시 보고서의 소수점 둘째 자리 값을 유지했다.

| 모델 | Prototype | seen 15,349장 | unseen 1,221장 | 전체 16,570장 |
| --- | --- | ---: | ---: | ---: |
| **ViT** | **평균 중심 1개** | **94.66** | **79.52** | **93.54** |
| ViT | K-means 중심 5개 | 94.46 | **85.18** | **93.77** |
| Band-ViT | 평균 중심 1개 | 94.37 | 80.84 | 93.37 |
| Band-ViT | K-means 중심 5개 | 94.48 | 84.19 | 93.72 |

ViT에서 평균 중심을 K-means 중심 5개로 변경한 조건은 unseen 정확도가 **약 5.65%p 증가**했고, seen 정확도는 **약 0.20%p 감소**했다. 차이는 전사 CSV의 소수점 6자리 값으로 계산했다. 표의 반올림된 두 값만 빼면 unseen 차이가 5.66%p로 보일 수 있다.

K-means 중심 5개는 같은 클래스 안의 여러 형태를 복수 중심으로 표현하는 방식이다. 촬영 방향이나 형태 차이를 더 잘 반영했을 가능성이 있으나, 이를 개선 원인으로 확정하려면 각 중심에 속한 이미지와 오분류 변화를 확인해야 한다. **중심 5개는 support 5장을 뜻하지 않는다.**

## 5. 결과 해석과 확인이 필요한 항목

**현재 보고할 수 있는 내용**

- 기존 ViT backbone 특징과 support 기반 OAS 분류 조합에서 unseen 79.52%가 기록됐다.
- 같은 평가 계열의 K-means 중심 5개 조건에서 unseen 85.18%가 기록됐다.
- 이 비교는 prototype 구성에 따른 결과 차이를 보여준다. OAS 자체의 개선 효과를 분리하려면 같은 특징·support/query에서 다른 거리와 비교해야 한다.
- 이 평가에서 새로운 SupCon·Gap loss를 학습해 성능이 향상됐다고 해석할 근거는 없다.

**재현·최종 성능 확정에 필요한 항목**

| 항목 | 현재 확인 수준 |
| --- | --- |
| Backbone 학습 설정 | 정확한 모델명, 사전학습, loss, epoch, batch size, 학습률, seed, 학습 클래스·이미지 전체 이력 재확인 필요 |
| 특징·전처리 | 임베딩 차원, 입력 크기, crop·정규화, 공분산 중심화 방식 재확인 필요 |
| Support 구성 | 클래스별 장수, 선택 seed, gallery manifest 및 query와의 분리 확인 필요 |
| 중복·데이터 누수 | 중복 제외 평가 기능은 있으나, 위 표가 이미지·환자 중복을 제외한 결과인지는 미확인 |
| 예측 조건 | seen/unseen 후보 클래스 범위와 unknown 거절 설정 재확인 필요 |
| 반복 검증 | 동일 조건의 여러 학습 seed·support 분할 평균과 변동 범위 미확인 |
| Macro-F1 | 당시 evaluator가 전체 클래스와 unknown을 집계에 포함하므로, unseen 15클래스 전용 Macro-F1은 별도 재계산 필요 |

현재 로컬의 `tools/evaluate_unseen_mahalanobis.py`는 평균 prototype + Ledoit–Wolf 구현이다. 이 로컬 파일을 그대로 실행하면 보고서의 서버 OAS 설정이 재현되는 것은 아니다.

## 6. 교수님께 설명할 문장

> 27개 seen 클래스와 15개 unseen 클래스로 구분한 평가에서, 기존 ViT backbone으로 이미지 특징을 추출한 뒤 클래스별 support 평균을 prototype으로 사용했습니다. 거리 계산에는 seen과 unseen gallery를 함께 사용해 추정한 OAS 공분산을 적용했습니다. 서버에 저장된 결과는 seen 정확도 94.66%, unseen 정확도 79.52%였습니다. 평균 중심 대신 클래스당 K-means 중심 5개를 사용하는 조건에서는 seen 94.46%, unseen 85.18%가 기록됐습니다. 이 결과는 라벨 있는 unseen support를 사용하는 평가이며, 최종 성능 확정을 위해 checkpoint 학습 이력과 support/query의 이미지·환자 중복 여부를 추가 확인해야 합니다.

## 7. 결과 출처

- 서버 결과 폴더: `/home/ubuntu/aihub_run_new/outputs/metric/closed27_open15/vit_band_galcov_20260928/`
- 서버 원문 CSV: 위 폴더의 `grid_summary_closed27_open15.csv`.
- ViT checkpoint: `/home/ubuntu/aihub_run_new/checkpoints/backbones/20260918-170637-6ea9b483/vit_full_closed27_open15_c001.pt`.
- Band-ViT checkpoint: 같은 폴더의 `band-vit_full_closed27_open15_c001.pt`.
- 평가 코드 확인 기록: `/home/ubuntu/aihub_run_new/outputs/experiment_records/20260924-galcov-5ep/evaluate.py`.
- 실행 설정·로그: `/home/ubuntu/aihub_run_new/outputs/experiment_records/followup-20260928-2228/`의 `galcov.sh`, `galcov.log`, `queue.log`.
- 확인 환경: 2026-09-30 약 12:28–12:46 KST, Windows App `localhost:3389`, `user07-gpu` 터미널.
- [당시 원격 확인 기록](/Users/yunseojune/.codex/worktrees/0550/implantsj/docs/experiment_audit_20260930/remote_results.md).
- [보고서 대상 12행 전사 수치](results_transcribed.csv): 당시 로컬 전사본에서 추출했으며 서버 원본 CSV 다운로드 파일이 아니다.

## 8. 실험 코드의 로컬 보존 여부

2026-10-06 추가 조사에서 **서버 반입용 Docker 이미지가 로컬에 보존된 것을 확인했다.** 처음에는 현재 소스와 소형 백업만 확인해 이 이미지를 빠뜨렸다.

| 보존 항목 | 로컬 위치와 확인 내용 |
| --- | --- |
| 서버 파이프라인 Docker 이미지 | `/Users/yunseojune/TRIA_LAB/implant_project/outputs_docker/ai_hub_pipeline_20260908.tar.gz` (15,320,709,732 bytes) |
| 이미지 내부 평가 소스 | `src/metric/evaluate.py`, `prototypes.py`, `scoring.py`, `src/run_metric.py` 확인 |
| OAS 구현 | `src/metric/scoring.py` 180–183행에서 `OAS().fit(residuals).precision_` 확인 |
| Prototype 구현 | `src/metric/prototypes.py`에서 mean 및 `kmeans_multi` 지원 확인 |
| 이미지 내부 실행 스크립트 | `/workspace/ai_hub_pipeline/scripts/03_metric_learning/run_metric_grid.sh`, `run_metric_fewshot.sh`, `run_clustering_full42.sh` 확인 |
| 수정본 사용설명서 | Downloads의 `implant_pipeline_new_사용설명서.md`; 서버 배포 태그 `implant_pipeline_new:20260917` 설명 |
| 서버 반출 결과 파일 | `/Users/yunseojune/TRIA_LAB/implant_project/07_download_1.tar`; 81개 logits CSV, Python·shell·YAML 파일 없음 |
| 현재 저장소의 관련 평가 코드 | `tools/evaluate_unseen_mahalanobis.py`; 평균 prototype + pooled Ledoit–Wolf로, 이 보고서의 OAS 설정과는 다름 |

**서버 파이프라인 코드가 로컬에 없다는 뜻은 아니다. Docker 이미지 내부에 보존돼 있다.** 다만 9월 8일 이미지와 9월 28일 OAS 후속 실험은 시점이 다르므로, 해당 이미지가 79.52%를 산출한 `galcov.sh` 및 후속 `evaluate.py`와 동일한지는 아직 확인되지 않았다. 실행 코드가 있다는 사실과 해당 결과를 정확히 재현할 버전이 확보됐다는 사실을 구분한다.

이미지 내부 기본 설정은 `configs/03_metric_grid.yaml`에서 covariance `ledoit_wolf`, 공분산 적합은 closed train으로 설명돼 있다. OAS 기능이 구현돼 있다는 것과 후속 GalCov 설정이 기본값으로 저장돼 있다는 것은 다르다. 서버 79.52% 결과의 gallery 공분산 적용 변경·실행 인자는 추가 대조가 필요하다.

## 9. Macro-F1 재계산 상태

2026-10-06 로컬 조사 결과, 해당 OAS 실행의 sample별 정답·예측 또는 혼동행렬을 찾지 못해 seen/unseen Macro-F1 재계산은 완료하지 못했다.

- `07_download_1.tar`의 파일 81개는 모두 `logits.csv`다. 같은 ViT checkpoint의 test 파일도 `sample_id`, `label`, `logits_0000`부터 `logits_0026`까지 27개 head 출력만 포함한다. OAS 예측값이나 backbone 임베딩이 아니다.
- 보존된 전사 CSV에는 전체 Macro-F1만 있고, seen/unseen의 F1 원문 값과 클래스별 TP/FP/FN은 없다.
- 당시 기록된 전체 Macro-F1 81.85%는 원문 값으로만 유지한다. 이를 unseen Macro-F1로 쓰거나 정확도에서 F1을 추정하지 않는다.
- 재계산에는 `vit_band_galcov_20260928` 실행의 mean + mahalanobis_oas 예측 CSV 또는 혼동행렬, seen 27개·unseen 15개 라벨 목록이 필요하다.
- 계산 시 query를 정답 라벨 기준으로 나누고, 각 그룹의 클래스 목록을 macro 평균 대상으로 명시한다. 그룹 밖으로 잘못 예측한 query도 삭제하지 않고 오분류로 반영한다. 전체 query에서 클래스만 묶는 별도 집계가 필요하면 구분해 기록한다.
- 기존 정확도와 표본 수(seen 15,349장 / unseen 1,221장)를 먼저 재현해 같은 실행인지 검증한 뒤 F1을 보고한다.

### 반출 파일 추가 조사

2026-10-06 다운로드·문서·바탕화면·프로젝트·클라우드 동기화 폴더의 파일 목록과 로컬 ZIP 내부 목록, Chrome 다운로드 기록을 추가 확인했다.

- Chrome 기록: `07_download_1.tar`는 2026-10-01 18:42 KST 다운로드 완료, 2,030,059,520 bytes. 현재 파일 존재 및 tar 내부 81개 logits CSV를 확인했다.
- Chrome 기록: `04_download_1.tar`는 2026-07-22 다운로드 완료, 22,368,071,680 bytes. 기록된 바탕화면 경로와 홈 폴더 파일명 검색에서 찾지 못했다. 9월 OAS 실험보다 이전 기록이므로 해당 결과 반출의 근거로 쓰지 않는다.
- 다른 실험의 예측·지표 CSV는 로컬에 있으나, 이 보고서의 `vit_band_galcov_20260928` mean + OAS 예측 또는 혼동행렬과 일치하는 자료는 아직 찾지 못했다.
- 일부 iCloud 전용 파일은 읽기 대기 상태여서 내용 확인을 마치지 못했다. 모든 클라우드 파일이나 반출 파일이 없다는 결론은 아니다.
