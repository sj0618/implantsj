# 개별 임플란트 실험 명령

기존 학습 코드는 수정하지 않는다. `scripts/*.sh`가 데이터·체크포인트·모델·
prototype·distance를 명시적 인자로 하나씩 받아 작업 하나만 실행한다.

`suite`, `run-suite`, 자동 모델 순회, Cartesian product, seed 자동 반복은
지원하지 않는다.

## 설치

```bash
python3 -m pip install -r implantsj/workflow-requirements.txt
export DATA_ROOT=/absolute/path/to/implant-data
```

## 1. 데이터 매니페스트 하나 검증

```bash
bash scripts/validate-data.sh \
  --dataset-manifest manifests/data/implant_v1.yaml
```

## 2. Split 하나 생성

```bash
bash scripts/build-split.sh \
  --dataset-manifest manifests/data/implant_v1.yaml \
  --split-manifest manifests/splits/closed27_open15_seed42.yaml \
  --output outputs/splits/seed42
```

## 3. 모델·seed 하나 학습

```bash
bash scripts/train.sh \
  --dataset-manifest manifests/data/implant_v1.yaml \
  --split-dir outputs/splits/seed42 \
  --experiment-manifest manifests/experiments/vit_closedft.yaml \
  --init-checkpoint checkpoints/vit_base.pt \
  --seed 42 \
  --output outputs/train/closedft_seed42
```

다른 모델이나 seed는 명령을 별도로 호출한다.

## 4. Checkpoint·view 하나의 임베딩 추출

```bash
bash scripts/extract-embeddings.sh \
  --dataset-manifest manifests/data/implant_v1.yaml \
  --split-dir outputs/splits/seed42 \
  --model-manifest manifests/models/vit_closedft.yaml \
  --checkpoint outputs/train/closedft_seed42/checkpoints/best.pt \
  --view closed_test \
  --embedding-output encoder \
  --output outputs/embeddings/closedft/closed_test
```

`open_test`, 다른 checkpoint 또는 projection output은 각각 별도 명령이다.

## 5. Prototype·geometry 하나 생성

```bash
bash scripts/build-prototype.sh \
  --embedding-dir outputs/embeddings/closedft/gt_closed \
  --prototype-type gt \
  --distance-geometry euclidean \
  --output outputs/prototypes/closedft/gt_euclidean
```

```bash
bash scripts/build-prototype.sh \
  --embedding-dir outputs/embeddings/closedft/train \
  --prototype-type ss \
  --samples-per-label 5 \
  --support-seed 42 \
  --distance-geometry cosine \
  --output outputs/prototypes/closedft/ss5_seed42_cosine
```

## 6. Query·prototype·distance·candidate space 하나 평가

```bash
bash scripts/evaluate-prototype.sh \
  --query-embeddings outputs/embeddings/closedft/closed_test \
  --prototype-dir outputs/prototypes/closedft/gt_euclidean \
  --distance euclidean \
  --query-group closed \
  --candidate-space closed27 \
  --output outputs/evaluations/closedft_gt_l2_closed
```

이 명령은 cosine, mean, SS 또는 다른 query를 추가 실행하지 않는다.

## 7. 임베딩·geometry·seed 하나 시각화

```bash
bash scripts/visualize-tsne.sh \
  --embeddings outputs/embeddings/closedft/all_train_test \
  --geometry cosine \
  --seed 42 \
  --output outputs/visualizations/closedft_tsne_cosine
```

## 8. 사전 생성된 결과 두 개 비교

```bash
bash scripts/compare-results.sh \
  --left outputs/evaluations/base_mean_l2_open_joint42 \
  --right outputs/evaluations/closedft_mean_l2_open_joint42 \
  --comparison-id experiment_1a \
  --comparison-label Base_vs_ClosedFT \
  --output outputs/comparisons/experiment_1a
```

비교 명령은 다른 결과를 검색하거나 추가 평가하지 않는다.

## 추가 단일 작업

```bash
bash scripts/diagnose-geometry.sh --help
bash scripts/linear-probe.sh --help
```

모든 결과 디렉터리는 새 경로여야 한다. 기존 결과를 덮어쓰지 않으며, 입력
매니페스트·split·checkpoint·embedding의 hash를 결과 metadata에 기록한다.

정상 종료된 명령은 결과 디렉터리에 `run.json`을 마지막으로 기록한다.
`run.json`에는 `status=completed`, 정규화된 명령 인자, workflow 코드 hash,
완료 시각, 산출물 상대 경로와 크기가 포함된다. `run.json`이 없으면 해당 결과를
완료된 작업으로 취급하지 않는다. JSON metadata는 임시 파일에 먼저 쓴 뒤
원자적으로 교체한다.
