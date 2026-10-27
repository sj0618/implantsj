# 서버 실험 파이프라인

2026-09-08 Docker 이미지에서 실행 코드를 모두 복구했다.
`src/` Python 65개, `configs/` YAML 11개, `scripts/` 셸 스크립트 16개를 포함한다.
Docker 운영 코드·의존성 목록·원본 테스트도 제공한다. 이 디렉터리는 루트의 `src/`와 별도로 실행한다.

## 코드와 실행 순서

| 단계 | 스크립트 | 코드 설명 |
|---|---|---|
| 데이터 준비 | `scripts/00_data/prepare_data.sh` | [manifest 생성](src/data/README.md) |
| 마스크 준비 | `scripts/02_segmentation/01_sample_finetune_data.sh`부터 `06_make_band_masks.sh`까지 | [SAM·adapter·band mask](src/segmentation/README.md) |
| 백본 학습 | `scripts/01_backbone/finetune_backbones.sh` | [ViT·DINOv2·Band](src/backbone/README.md) |
| Metric | `scripts/03_metric_learning/run_metric_grid.sh` | [거리·prototype 비교](src/metric/README.md) |
| Clustering | `scripts/03_metric_learning/run_clustering_full42.sh` | 42-class 평가 |
| Few-shot | `scripts/03_metric_learning/run_metric_fewshot.sh` | support/query 분리 평가 |
| Band 분류 | `scripts/04_classifier/train_band_vit.sh` | [attention prior](src/band_vit/README.md) |
| Masked 분류 | `scripts/04_classifier/train_masked_vit.sh` | [마스크·원본 입력 비교](src/masked_vit/README.md) |
| 속성 분류 | `scripts/05_attributes/train_attributes.sh` | [독립·공유 모델](src/attributes/README.md) |
| 결과 결합 | `scripts/06_fuse/run_fuse.sh` | [검증 점수로 가중치 결정](src/fuse/README.md) |

SAM1 마스크의 수동 선별은 segmentation ②와 ③ 사이에 수행한다.
Band mask를 만든 뒤 데이터 준비를 다시 실행해 `mask_path`를 연결한다.
과거 split이 있으면 원래 manifest를 사용한다. 새 분할을 과거 실험으로 간주하지 않는다.

## 실행 환경

이 디렉터리에서 실행한다. 원본 환경은 Linux/amd64, PyTorch 2.5.1/CUDA 12.4다.

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchvision==0.20.1
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -r requirements-seg.txt
python src/verify_pipeline.py
python tests/test_smoke.py
python src/run_backbone.py --help
```

[Docker 빌드·반입](docker/README.md), [설정·외부 입력](configs/README.md), [실행 도구](src/README.md)를 참고한다.
`weights/`는 빌드 입력 폴더다. 사전학습 가중치는 빌드 때 받고, 팀 adapter는 사용자가 넣는다.
원본 Dockerfile에서 업로드하지 않는 `EXPERIMENT_PLAN.md`의 COPY 한 줄을 제거했다.
추가한 ignore 파일은 데이터·가중치·산출물의 Git 추적과 불필요한 빌드 입력을 방지한다.

## 재현 범위

- Python·YAML·셸 스크립트는 이미지 원본과 바이트 단위로 일치한다.
- 9월 8일 기본 설정과 9월 18~28일 실행 설정은 구분한다. 당시 설정·가중치·manifest가 필요하다.
- `closed27_open15` metric은 공분산을 closed train으로 적합한다. closed+open gallery 보고서와 다르다.
- 로컬 재현 실행기와 서버 원본은 K-means·RDA·표본 추출이 다르므로 서로 대체하지 않는다.
- 원본 자체 점검: 194 OK, 0 WARN, 0 FAIL. 임베딩·attributes 독립/공유 모델 CPU 테스트 통과.
- 실제 데이터 학습·SAM 추론·GPU 실행·Docker 재빌드·과거 성능 일치는 별도 검증 대상이다.
