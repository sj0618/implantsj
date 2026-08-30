# 개별 임플란트 실험 명령

기존 학습 코드는 수정하지 않는다. `scripts/*.sh`가 데이터·체크포인트·모델·
prototype·distance를 명시적 인자로 하나씩 받아 작업 하나만 실행한다.

`suite`, `run-suite`, 자동 모델 순회, Cartesian product, seed 자동 반복은
지원하지 않는다.

## 설치

```bash
python3 -m pip install -r workflow-requirements.txt
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

## Canonical metric-learning 모드

기존 명령에 `--protocol-manifest`를 주지 않으면 기존 동작을 유지한다. canonical
모드에서는 protocol, seed와 model role을 함께 명시한다.
canonical `build-split`은 세 seed 매니페스트/assignment의 고정 hash와 동일 sample
universe/common GT를 함께 검증하고, 네 경계(train/validation/test/GT)의
sample/content/group 분리를 `isolation_audit.json`에 기록한다.

```bash
bash scripts/train.sh \
  --protocol-manifest manifests/protocols/metric_learning_v1.yaml \
  --model-role ClosedFT-SupCon \
  --dataset-manifest manifests/data/implant_metric42_seed42_v1.yaml \
  --split-dir outputs/seed42/split \
  --experiment-manifest manifests/experiments/vit_closedft_supcon_study.yaml \
  --init-checkpoint "$BASE_CHECKPOINT" \
  --seed 42 \
  --output outputs/seed42/train/closedft-supcon

bash scripts/extract-embeddings.sh \
  --protocol-manifest manifests/protocols/metric_learning_v1.yaml \
  --model-role ClosedFT-SupCon \
  --seed 42 \
  --dataset-manifest manifests/data/implant_metric42_seed42_v1.yaml \
  --split-dir outputs/seed42/split \
  --model-manifest manifests/models/vit_closedft_supcon_study.yaml \
  --checkpoint outputs/seed42/train/closedft-supcon/checkpoints/last.pt \
  --view validation \
  --embedding-output encoder \
  --output outputs/seed42/embeddings/closedft-supcon/validation
```

validation 평가는 freeze 없이 실행한다. test 평가는 같은 key가 사전에 등록된
freeze를 반드시 전달한다.
freeze는 validation key/artifact 집합과 미래 test key/path 집합의 hash,
`frozen_validation_results.csv` hash, 생성 시각, commitment를 기록한다. test 결과는
freeze 파일 hash와 commitment를 metadata에 남긴다.

```bash
bash scripts/evaluate-prototype.sh \
  --protocol-manifest manifests/protocols/metric_learning_v1.yaml \
  --seed 42 \
  --model-role ClosedFT-SupCon \
  --query-view test \
  --prototype-type gt \
  --query-embeddings outputs/seed42/embeddings/closedft-supcon/test \
  --prototype-dir outputs/seed42/prototypes/closedft-supcon/gt-euclidean \
  --distance euclidean \
  --query-group all \
  --candidate-space full42 \
  --validation-freeze outputs/seed42/validation-freeze \
  --output outputs/seed42/evaluations/closedft-supcon-test-gt-euclidean
```

canonical `compare-results`에는 `--seed`, `--left-model`, `--right-model`,
`--query-view`, `--prototype-type`, `--distance`를 모두 전달한다. 비교 가능한
ordered pair는 protocol의 `treatment_pairs`에 선언된 12개뿐이다.

### Ledger 계약

`freeze-validation` 입력 CSV:

```text
record_type,seed,model,query,prototype,distance,candidate_space,output_path,artifact_sha256,status,protocol_fingerprint
```

evaluation ledger:

```text
seed,model,query,prototype,distance,candidate_space,evaluation_path,evaluation_sha256,status,protocol_fingerprint,freeze_commitment_sha256
```

comparison ledger:

```text
seed,experiment_id,left_model,right_model,query,prototype,distance,candidate_space,left_evaluation_sha256,right_evaluation_sha256,comparison_path,comparison_sha256,status,protocol_fingerprint,freeze_commitment_sha256
```

각 명령은 전달된 단일 조건만 처리한다. ledger 생성, 모델/seed 반복, 누락 결과
검색은 외부 orchestration의 책임이며 `finalSJ` 명령은 이를 자동 실행하지 않는다.
