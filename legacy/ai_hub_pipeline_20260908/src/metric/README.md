# Metric 학습과 평가

| 코드 | 역할 |
|---|---|
| `core.py` | manifest·이미지·학습/평가 공용 처리 |
| `models.py`, `train.py` | 모델·손실·학습·checkpoint 로딩 |
| `evaluate.py` | gallery/query 구성, 거리 조합·seen/open 집계 |
| `fewshot.py` | shot·seed별 support/query 평가 |
| `prototypes.py` | mean·SS·medoid·K-means·reference 중심 구성 |
| `scoring.py` | 특징 변환·거리·공분산 적합 |
| `reference.py` | 대표 이미지 폴더 읽기와 중복 검사 |
| `openset.py` | unknown 점수·거부 지표 |
| `visualize.py` | 임베딩·점수 분포 시각화 |
| `__init__.py` | 패키지 연결 |

```bash
python src/run_metric.py --config configs/03_metric_grid.yaml
python src/run_metric.py --config configs/03_clustering_full42.yaml
python src/run_metric_fewshot.py --config configs/03_metric_fewshot.yaml
```

실행 전 manifest, checkpoint 및 새 출력 경로를 지정한다.
`closed27_open15`는 closed train으로 거리 파라미터를 적합하고 open train을 gallery support로 등록한다.
K-means 5는 중심 5개이며 5-shot을 뜻하지 않는다.
Few-shot 설정의 checkpoint가 null이면 이름과 관계없이 미세조정하지 않은 backbone을 사용한다.
출력은 지표·점수·예측·표본 구성 및 설정에 따른 그림이다.
