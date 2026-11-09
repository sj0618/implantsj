# 서버 실행 설정

`base.yaml`은 경로·모델 레지스트리·공통 학습 값이다. 나머지 YAML은 base를 상속한다.

| 설정 | 실행 코드 |
|---|---|
| `00_data.yaml` | `src/run_prepare_data.py` |
| `01_backbone.yaml` | `src/run_backbone.py` |
| `02_segmentation.yaml` | `src/segmentation/`의 6단계 |
| `03_metric_grid.yaml`, `03_clustering_full42.yaml` | `src/run_metric.py` |
| `03_metric_fewshot.yaml` | `src/run_metric_fewshot.py` |
| `04_band_vit.yaml`, `04_masked_vit.yaml` | 해당 분류 드라이버 |
| `05_attributes.yaml` | `src/run_attributes.py` |
| `06_fuse.yaml` | `src/run_fuse.py` |

원본 기본값을 유지했다. 실제 실행은 설정 복사본이나 `--set`으로 입력·출력·모델을 지정한다.
`paths.*` 상대경로는 이 파이프라인 루트를 기준으로 해석한다.
다른 상대경로도 맞도록 파이프라인 루트에서 명령을 실행한다.

외부 입력은 이미지, manifest, mask, checkpoint, 속성 정답표 및 이전 단계의 valid/test 점수다.
정답표는 `configs/spec_table.csv`에 준비하거나 `--set spec.csv=/path/to/table.csv`로 지정한다.
생성 데이터와 정답표는 Git에 포함하지 않는다.
9월 8일 기본값은 이후 실행별 확정 설정이 아니다. 정확한 재현에는 원 실행 기록을 적용한다.
