# 실험보고서 재현 가이드

기존 학습·평가 코드에 **보고서별 실행 설정, 입력 검사, 실행 기록**을 연결했다.
현재 `implantsj`의 보고서와 보관된 작업본의 누락 보고서를 대상으로 한다.
별도 프로젝트 `implantys`, 발표 PDF/PPTX, 원본 이미지·모델 가중치는 이 목록의 대상이 아니다.

**코드 제공과 과거 점수 재현은 다르다.** 아래 목록은 실행 가능한 범위를 뜻한다.
실제 데이터·checkpoint로 전체 실험을 다시 실행하거나 과거 수치 일치를 확인한 상태가 아니다.
원본 설정이 부족한 실험은 필수 정보가 복구되기 전 실행을 차단한다.

## 실행

저장소 루트에서 실행한다. 목록·계획·입력 경로 검사에는 Python 표준 라이브러리만 필요하다.
학습·평가에는 기존 PyTorch 환경과 `requirements.txt`가 필요하다.

```bash
python3 tools/run.py reproduce list
python3 tools/run.py reproduce plan --report target5_episodic5
```

실행할 보고서에 필요한 경로만 별도 JSON에 넣는다. 경로 예시는
[`inputs.example.json`](../../configs/reproduction/inputs.example.json)에 있다.
`target5_episodic5`에는 다음 세 입력이 필요하다.

```json
{
  "supcon6_checkpoint": "/workspace/checkpoints/supcon6/best.pt",
  "source6_manifest": "/workspace/manifests/source_6.csv",
  "target5_manifest": "/workspace/manifests/target_5.csv"
}
```

이를 `inputs.local.json`으로 저장하고 실행한다. JSON에는 경로만 넣고 인증정보를 넣지 않는다.

```bash
python3 tools/run.py reproduce check --report target5_episodic5 --inputs inputs.local.json
python3 tools/run.py reproduce run --report target5_episodic5 --inputs inputs.local.json --output outputs/reproduction/run01
```

- `plan`: 실제로 사용할 명령, 필수 입력, 재현 한계를 출력한다. 학습·다운로드는 하지 않는다.
- `check`: 필수 경로 존재 여부를 확인한다. 이미지 내용·checkpoint 일치까지 검증한 것은 아니다.
- `run`: 새 출력 폴더에서 순서대로 실행한다. 이미 존재하는 출력 폴더는 거부한다.
- `--report all`: 목록 전체를 선택한다. 차단된 항목이나 누락 입력이 있으면 **어떤 학습도 시작하지 않는다**.
- `fresh_supcon7_to3`, `supcon6_to5_strict10`: 학습 후 생성된 checkpoint 하나를 확인해 다음 평가에 넘긴다.
- 모델 사전학습 가중치는 기존 trainer가 사용한다. 폐쇄망에서는 기존 Docker 반입 절차로 먼저 준비한다.

## 보고서별 범위

| 실행 ID | 원 보고서 | 범위 | 단계 |
|---|---|---|---:|
| `may_classification_transfer` | [EXPERIMENT_SUMMARY.md](../../docs/analysis/EXPERIMENT_SUMMARY.md) | 일부 실행 | 11 |
| `presentation_summary` | [PRESENTATION_SUMMARY.md](../../docs/analysis/PRESENTATION_SUMMARY.md) | 원본 확인 필요 | 0 |
| `umap_comparison` | [TSNE_UMAP_COMPARISON.md](../../docs/analysis/TSNE_UMAP_COMPARISON.md) | 프로토콜 실행 | 1 |
| `metric_diagnosis_training` | [METRIC_DIAGNOSIS_CURRENT_RESULTS.md](../../docs/reports/METRIC_DIAGNOSIS_CURRENT_RESULTS.md) | 일부 실행 | 3 |
| `metric_diagnosis_gt27` | [METRIC_DIAGNOSIS_FINAL_REPORT.md](../../docs/reports/METRIC_DIAGNOSIS_FINAL_REPORT.md) | 원본 확인 필요 | 0 |
| `vit42_local_prototypes` | [REPORT.md](../../docs/reports/vit42ft_local_metric_20260806/REPORT.md) | 프로토콜 실행 | 3 |
| `vit42_local_four_distances` | [REPORT.md](../../docs/reports/vit42ft_local_mean_4dist_20260806/REPORT.md) | 프로토콜 실행 | 3 |
| `cropped_ce7_to3` | [REPORT.md](../../docs/reports/vit7ce_to_3label_mahalanobis_20260806/REPORT.md) | 프로토콜 실행 | 1 |
| `fresh_supcon7_to3` | [REPORT.md](../../docs/reports/vit7supcon_to_3label_mahalanobis_20260807/REPORT.md) | 프로토콜 실행 | 3 |
| `legacy_supcon7_to3` | [REPORT.md](../../docs/reports/legacy_7to3_reproduction_20260807/REPORT.md) | 프로토콜 실행 | 2 |
| `mahalanobis_improvement` | [REPORT.md](../../docs/reports/mahalanobis_improvement_20260813/REPORT.md) | 프로토콜 실행 | 4 |
| `prototype_only_10shot` | [REPORT.md](../../docs/reports/mahalanobis_10shot_prototype_20260814/REPORT.md) | 프로토콜 실행 | 1 |
| `strict_3label_10shot` | [REPORT.md](../../docs/reports/mahalanobis_strict_10shot_20260814/REPORT.md) | 프로토콜 실행 | 1 |
| `supcon6_to5_strict10` | [REPORT.md](../../docs/reports/vit6supcon_to_5label_strict10_20260814/REPORT.md) | 프로토콜 실행 | 2 |
| `target5_episodic5` | [REPORT.md](../../docs/reports/vit6supcon_to_5label_strict10_20260814/episodic_all_target_5shot/REPORT.md) | 프로토콜 실행 | 1 |
| `source_closed_auxiliary` | [REPORT.md](../../docs/reports/vit6supcon_to_5label_strict10_20260814/episodic_all_target_5shot/REPORT.md) | 조건 재구성 | 2 |
| `mahalanobis_method_selection` | [REPORT.md](../../docs/reports/mahalanobis_method_selection_20260816/REPORT.md) | 프로토콜 실행 | 1 |
| `mahalanobis_spectrum` | [MAHALANOBIS_SPECTRUM_VALIDATION_FINAL_REPORT.md](../../docs/reports/MAHALANOBIS_SPECTRUM_VALIDATION_FINAL_REPORT.md) | 프로토콜 실행 | 5 |
| `mahalanobis_spectrum_validation` | [VALIDATION.md](../../docs/reports/mahalanobis_spectrum_validation_20260816/VALIDATION.md) | 일부 실행 | 5 |
| `uncropped7_to3` | [REPORT.md](../../docs/reports/full_uncropped_7to3_metric/report/REPORT.md) | 일부 실행 | 6 |
| `uncropped_mahalanobis_addendum` | [REPORT_TABLE.md](../../docs/reports/full_uncropped_7to3_metric/mahalanobis_addendum_20260807/REPORT_TABLE.md) | 프로토콜 실행 | 2 |
| `server_oas_gallery` | [report.md](../../docs/reports/mahalanobis_professor_20261006/report.md) | 원본 확인 필요 | 0 |
| `server_backbone_comparison` | [implant_experiment_summary_table.md](../../implant_experiment_summary_table.md) | 원본 확인 필요 | 0 |
| `vlm_generation_snapshot` | [VLM_3LABEL_APPLY_STATUS.md](../../docs/vlm/VLM_3LABEL_APPLY_STATUS.md) | 원본 확인 필요 | 0 |

상세 명령과 한계는 [`reports.json`](../../configs/reproduction/reports.json)의 `steps`, `limitations`, `blocked`가 기준이다.
`protocol`은 기록된 평가 방식을 실행한다는 의미이며, 원 보고서의 모든 그림과 과거 점수 일치를 보장하는 상태가 아니다.

## 평가 계약

| 조건 | Prototype | 공분산 | 방법 선택 | Query |
|---|---|---|---|---|
| 일반 gallery | train 평균 / valid GT / train SS | train 전체 | 고정 또는 명시적 validation | valid/test; GT는 test만 |
| Prototype-only 10-shot | train 10장/클래스 | train 전체 | validation | 고정 test |
| Strict 10-shot | train 10장/클래스 | **동일한 10장** | 고정 | 고정 test |
| 전체 pool 5-shot | 전체 pool 중 5장/클래스 | **동일한 5장** | 고정 | pool에서 support를 뺀 나머지 |
| 서버 OAS gallery | 평균 또는 K-means 5개 중심 | seen+unseen gallery | 원본 설정 복구 필요 | seen/unseen query |

- 반복 seed는 `0, 1, 2, 3, 4, 42`이며, uncropped SS 비교는 원 보고서의 `2026`을 사용한다.
- Strict 평가는 support/query의 경로 및 실제 파일 SHA-256 중복을 거부한다.
- Gallery 평가는 원래 test를 유지하고 중복이 있으면 `test_duplicate_free`를 별도로 출력한다.
- `--source-manifest`를 지정하면 source/target 라벨과 이미지 내용이 겹치는지도 검사한다.
- 환자·촬영 그룹 독립성은 파일 해시만으로 입증할 수 없다. 출력에 미확인 상태를 남긴다.
- 새 checkpoint와 원래 checkpoint는 서로 다른 실험 입력이다. 새 학습을 원래 점수의 정확한 재실행으로 표시하지 않는다.
- Validation을 쓰는 개선 실험과 fixed-method strict 실험을 합쳐 평균 내지 않는다.
- 42-class backbone 평가는 raw pre-logit, SupCon 계열은 L2 정규화를 명시한다.
- GT/SS prototype 그림, attention, 그룹별 sensitivity 등 미포함 항목은 각 보고서의 한계에 따로 기록했다.

## 입력과 결과

Manifest는 `path,label,split` CSV이며 split은 `train,valid,test`다.
보고서와 동일한 이미지와 순서·분할을 사용해야 한다. 자동 재분할이나 임의 라벨 합치기는 하지 않는다.
이미지 경로는 실행 환경에서 읽을 수 있어야 한다. 캐시 파일만 다른 컴퓨터로 옮겨도 이미지 경로가 자동 변경되지는 않는다.

임베딩 캐시 입력은 pickle 없이 읽는 NPZ다.
`labels`와 각 split의 `train_x/train_y/train_paths`, `valid_x/valid_y/valid_paths`, `test_x/test_y/test_paths`가 필요하다.
`*_x`는 N×D, `*_y`는 `labels`의 정수 인덱스, `*_paths`는 문자열 배열이다.
일반 평가를 실행하면 같은 형식의 `embedding_cache.npz`를 저장하므로 후속 거리 실험에서 재사용할 수 있다.
기존 cache가 이 형식이 아니라면 원 checkpoint와 manifest로 다시 추출한다. 정규화와 입력 신원은 별도로 대조한다.

```text
outputs/reproduction/run01/
  run.json                     명령, 입력·소스 해시, 패키지 버전, 단계별 실행 상태
  <report_id>/<step>.log        실행 로그
  <report_id>/<evaluation>/
    REPORT.md                  이번 계산 결과만 담은 표
    results.csv                seed·거리·split별 값
    summary.csv                평균, 표준편차, 최솟값, 최댓값, 반복 수
    predictions.csv            이미지별 정답·예측
    membership.csv             support/query 구성과 이미지 해시
    per_class.csv              클래스별 Precision/Recall/F1
    confusion_*.csv            혼동행렬
    protocol.json              실행 조건과 재현 한계
    embedding_cache.npz        후속 평가용 특징
```

원 보고서 수치는 `historical_reference`로만 보관한다. 결과 계산에 사용하거나 누락 결과를 채우지 않는다.
`executed`는 명령 성공과 새 산출물 존재를 확인했다는 의미다. `reproduced`나 과거 점수 일치 판정을 자동으로 붙이지 않는다.
프로세스 실패·산출물 누락은 `failed`로 기록하고 다음 단계를 실행하지 않는다.

## 원본이 더 필요한 실험

- **서버 OAS**: 2026-09-28 `galcov.sh`, evaluator, grid 설정, checkpoint 이력, gallery/query manifest.
  2026-09-08 Docker 소스의 존재만으로 같은 버전이라고 간주할 수 없다. OAS 중심화·정규화·후보 라벨·거절 설정이 필요하다.
- **서버 백본 비교**: 모델 정의와 9월 8일 기본 설정은 복구했다. 9월 18~20일 실제 적용 설정, Band mask, 27/42-class split, checkpoint 및 원래 `summary.json`과 실행 소스 버전은 추가 확인이 필요하다.
- **GT27 최종 진단**: 다섯 모델의 실제 model spec, checkpoint, support 189/query 52의 고정 배정.
  계획했던 canonical 42-way evaluator로 대체하면 다른 실험이 된다.
- **발표문에 연결된 May-26 fine-tuning**: 누락된 `RESULT_ANALYSIS.md`, `args.json`, 원본 split.
- **VLM 진행률 보고**: 당시 deployment/model 버전과 요청·응답 기록. 새 API 호출로 62건 진행 상태를 재현했다고 말할 수 없다.

복구한 Markdown은 당시 기록을 보존한다. 연결된 이미지·cache·checkpoint·절대 경로는 함께 복구하지 않았으므로 일부 링크는 열리지 않을 수 있다.
복구 출처와 원본 해시는 [`restored_sources.json`](../../configs/reproduction/restored_sources.json)에 기록했다.

서버 반입 이미지에서는 [백본·metric 소스](../../legacy/ai_hub_pipeline_20260908/README.md)를 별도로 복구했다.
이 버전은 `closed27_open15`의 공분산을 closed train으로 적합하므로 9월 28일 gallery 적합 보고서와 다르다.
백본 소스가 전혀 없는 상태는 해소했지만, 실제 실행 버전과 설정은 추가 확인이 필요하다.
전체 이미지 대조 결과와 구현 차이는 [Docker 파일 비교](../analysis/DOCKER_REPRODUCTION_COMPARISON_20261026.md)에 정리했다.

## 검증

```bash
python3 -m pytest -q tests
python3 -m tools.improve_unseen_mahalanobis --self-check
```

테스트는 보고서 누락, 명령 인자, 새 출력 강제, 실패 처리, support/query 분리, strict 공분산 제한,
seed별 표본 수, 거리 수식과 SciPy 일치, 입력 오류, 실행 기록을 검사한다.
실제 GPU 학습, 서버 실행, 기존 성능 수치와의 일치는 별도 검증 대상이다.

2026-10-26 로컬 검증: **29 tests + 17 subtests 통과**. 합성 특징으로 실제 실행기를 호출한
strict 3-label 10-shot smoke test도 **6 seeds × 5 methods = 30개 결과 행**을 생성했다.
신규 코드 Ruff 검사와 Python 컴파일 검사를 통과했다. 이 검증에는 GPU 학습·원 데이터 성능 재평가를 포함하지 않는다.
