# Docker 이미지와 재현 코드 비교

확인일: 2026-10-26. 비교 대상 Git 커밋: `5a6534fcbb792b7ab6c67167327a56423469cda3`.

## 확인 범위

- 이미지: `ai_hub_pipeline_20260908.tar.gz`, 15,320,709,732 bytes.
- 생성 시각: 2026-09-08 10:37:25 UTC. Linux/amd64, 작업 폴더 `/workspace/ai_hub_pipeline`.
- OCI manifest: `sha256:8343a45d4d3a13327b22e825024f1e51e1a8d7b4e06a01bbea559de9c343dcd0`.
- 이미지의 모든 레이어에서 53,765개 항목을 열람하고 프로젝트 경로를 레이어 순서로 대조했다. 프로젝트 파일은 142개다.
- Docker 엔진이 실행 중이지 않아 보관된 이미지 파일을 직접 읽었다. 실행 중인 서버 컨테이너나 서버의 연결 폴더는 확인하지 않았다.

## 파일 일치 여부

| 범위 | 이미지 내부 | GitHub 복구본 | 결과 |
|---|---:|---:|---|
| `src/` Python 소스 | 65 | 17 | 복구한 17개 모두 SHA-256 일치 |
| `configs/` YAML | 11 | 3 | 복구한 3개 모두 SHA-256 일치 |
| `scripts/` 실행 스크립트 | 16 | 0 | 복구 범위에 포함하지 않음 |
| 위 세 범위 합계 | 92 | 20 | 변경된 원본 파일 없음, 나머지 72개 미포함 |

파일별 결과는 [CSV](docker_file_comparison_20261026.csv)에 기록했다. 비교 경로는
`legacy/ai_hub_pipeline_20260908/`이며 현재 프로젝트의 동명 `src/`와 구분한다.
나머지 72개에는 데이터 준비, segmentation, 별도 Band/Masked 분류기, attributes/fuse,
few-shot 실행기, 점검 도구와 셸 스크립트가 있다. 이 파일들을 실행 차단한 것은 아니다.
142개 중 나머지 50개는 Docker 운영 파일, 의존성 목록, 문서, 테스트, CSV 및 캐시 등이다.

## 보류한 보고서 재검토

| 보고서 | 이미지에서 확인한 내용 | 아직 필요한 자료 |
|---|---|---|
| 서버 백본 비교 | ViT·DINOv2·Band·Large 모델 정의와 학습 코드, 기본 설정 있음 | 9월 18~20일 실제 설정 변경값, Band mask, split, checkpoint, `summary.json`, 실행 소스 버전 |
| 서버 OAS | OAS, mean/K-means prototype 구현 있음 | 9월 28일 `galcov.sh`, 수정 evaluator, 실행 설정, gallery/query, checkpoint 이력 |
| GT27 진단 | 범용 metric 학습·평가 코드 있음 | 해당 다섯 모델의 spec·checkpoint와 support 189/query 52의 고정 배정 |
| May-26 fine-tuning | 해당 `RESULT_ANALYSIS.md`, `args.json` 파일을 찾지 못함 | 원 보고서, 실행 인자, split |
| VLM 진행률 | 해당 요청·응답 기록을 찾지 못함 | 당시 모델·배포 버전과 62건 처리 기록 |

따라서 **5개 보고서의 정확한 재현은 계속 보류**한다. 서버 백본의 모델 정의가 없다는 설명은 부정확하므로,
재현 목록과 안내 문구를 실제 실행 기록이 필요한 상태로 수정했다. 기본 설정으로 새 실험을 실행할 수 있다는 사실은
과거 보고서와 같은 실험이라는 증거가 아니다.

## 결과에 영향을 주는 구현 차이

| 항목 | 9월 8일 Docker 코드 | 현재 재현 코드 또는 보고서 |
|---|---|---|
| OAS 적합 대상 | `closed27_open15`에서는 closed train만 사용 | 9월 28일 보고서는 closed+open gallery 사용 |
| K-means 중심 | 입력 특징에서 K-means 수행 | 로컬 `multi_*` 후보는 pooled covariance로 whitening한 특징에서 수행 |
| RDA 거리 보정 | `sqrt(d²) + w·logdet` | 로컬 후보는 `d² + w·logdet`; 보정이 있으면 예측이 달라질 수 있음 |
| SS 표본 선택 | seed와 클래스 이름으로 클래스별 난수 생성 | 로컬 재현기는 하나의 난수 생성기로 클래스 순서대로 선택; 같은 seed만으로 동일 표본을 보장하지 않음 |

근거: 복구본의 [`evaluate.py`](../../legacy/ai_hub_pipeline_20260908/src/metric/evaluate.py),
[`prototypes.py`](../../legacy/ai_hub_pipeline_20260908/src/metric/prototypes.py),
[`scoring.py`](../../legacy/ai_hub_pipeline_20260908/src/metric/scoring.py),
로컬 [`improve_unseen_mahalanobis.py`](../../tools/improve_unseen_mahalanobis.py),
[`reproduce_gallery.py`](../../pipelines/eval/reproduce_gallery.py).
두 구현을 교체해서 같은 서버 실험이라고 간주하지 않는다. 이번 비교에서 계산식을 변경하지 않았다.

## 이미지에 실행 결과가 없는 이유

이미지의 `docker/run_container.sh`는 서버의 `$WORK/data`, `$WORK/outputs`, `$WORK/checkpoints`,
`$WORK/seg_data`를 컨테이너에 연결한다. 해당 파일 SHA-256은
`834402892fd962b16d2b2de87328eb871cf0a2d85e1084ea01abf51a6fd26dc5`이다.
따라서 서버에서 생성된 결과가 이 반입 이미지에 자동 저장되지는 않는다.
이미지의 프로젝트 폴더에는 실험 checkpoint·NPZ cache·결과 JSON이 없었다.
이는 이미지의 다른 경로에 사전학습 가중치가 없다는 뜻은 아니다.

실제 재현 검증에는 서버 연결 폴더의 실행 기록이 필요하다. GPU 실행과 과거 수치 일치는 이번 비교에서 검증하지 않았다.
