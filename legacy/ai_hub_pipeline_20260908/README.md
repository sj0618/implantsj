# 2026-09-08 서버 반입 소스

보관된 Docker 이미지에서 **백본 학습과 metric 평가에 필요한 소스·설정 20개**를 복구했다.
원본 소스를 수정하지 않았으며 파일별 Docker layer와 SHA-256은 [provenance.json](provenance.json)에 있다.
전체 이미지 대조에서도 20개 파일 모두 일치했다. [비교 결과](../../docs/analysis/DOCKER_REPRODUCTION_COMPARISON_20261026.md)를 참고한다.
데이터, 모델, 이미지, Docker 바이너리는 포함하지 않는다.

- 백본: ViT / DINOv2 / Band 변형, full/partial fine-tuning, 27/42-class protocol.
- Metric: prototype, OAS/Ledoit-Wolf/RDA, 거리 평가, seen/open 집계.
- 이 코드는 현재 `src/`와 분리해서 실행한다. 두 `src`를 합치지 않는다.

## 보고서와의 차이

`src/metric/evaluate.py`의 `closed27_open15` 경로는 공분산 적합용 `fit_z`를
**closed train**에서 얻는다. 반면 2026-09-28 보고서는 **closed+open gallery** 적합을 기록했다.
따라서 이 코드를 실행해 79.52%/85.18%를 재현했다고 주장할 수 없다.
9월 18~20일 백본 실행과 설정이 같은지도 확인되지 않았다.

## 실행 인터페이스 확인

저장소 루트에서 다음 명령으로 인터페이스를 확인한다. 학습은 시작하지 않는다.

```bash
python3 legacy/ai_hub_pipeline_20260908/src/run_backbone.py --help
python3 legacy/ai_hub_pipeline_20260908/src/run_metric.py --help
```

실제 실행은 원래 실행 설정을 복구한 뒤 `--config`로 전달한다.
`configs/01_backbone.yaml`과 `configs/03_metric_grid.yaml`은 **9월 8일 기본 설정**이며 보고서별 확정 설정이 아니다.
Band 모델에는 manifest의 `mask_path`와 실제 mask가 필요하다.
경로와 출력 폴더를 지정하고, 사용한 설정을 복사해 보존한다.
기존 서버 스크립트는 출력 폴더 재사용을 허용하므로 반드시 새 출력 폴더를 사용한다.

현재 재현 실행기는 이 버전 차이를 해결할 때까지 두 서버 보고서의 실행을 차단한다.
