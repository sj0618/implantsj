# 실행 도구와 공용 코드

이 서버 파이프라인의 루트에서 실행한다. 일반 드라이버는 `--config`, `--protocol`, `--set key=value`를 받는다.

| 코드 | 역할 / 연결 모듈 |
|---|---|
| `run_prepare_data.py` | [데이터](data/README.md) 준비; `--inspect`는 폴더 검사만 수행 |
| `run_backbone.py` | [백본](backbone/README.md) 학습 |
| `run_metric.py`, `run_metric_fewshot.py` | [Metric](metric/README.md) 학습·평가·few-shot |
| `run_band_vit.py` | [Band 분류](band_vit/README.md) |
| `run_masked_vit.py` | [Masked 분류](masked_vit/README.md) |
| `run_attributes.py` | [속성 분류](attributes/README.md) |
| `run_fuse.py` | [결과 결합](fuse/README.md) |
| `config_utils.py` | YAML 상속·설정 변경·paths 경로 해석 |
| `split_manifest.py` | 학습 프로토콜에 맞는 manifest 선택·분리 |
| `embeddings.py` | 학습된 모델의 backbone/head 특징 NPZ와 해시 기록 |
| `check_backbones.py` | 모델 레지스트리·가중치 연결 검사 |
| `check_open_labels.py` | 설정의 open 클래스와 데이터 라벨 매칭 |
| `check_spec_table.py` | 데이터 라벨과 속성 정답표 연결; `--fill`은 표 수정 |
| `summarize_runs.py` | 로컬 실행 결과 집계; `--outputs`, `--out` 지정 |
| `verify_pipeline.py` | 합성 데이터로 설정·입출력·거리·분할·결합 검사 |

```bash
python src/run_prepare_data.py --config configs/00_data.yaml --inspect
python src/run_metric.py --help
python src/summarize_runs.py --help
python src/verify_pipeline.py
```

실제 데이터·checkpoint·결과 표는 외부 입력 또는 로컬 산출물이다.
패키지의 `__init__.py`는 내부 모듈 연결용이며 직접 실행하지 않는다.
