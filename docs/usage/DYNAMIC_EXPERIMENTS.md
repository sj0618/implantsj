# 실험 명령 모음

동적 실험 코드는 모두 저장소 루트 `src/`에 있다. 별도
`metric_diagnosis` 패키지나 결합 실행기는 사용하지 않는다. 아래 변수만 바꾸면
각 명령을 독립적으로 다시 실행할 수 있다.

```bash
conda activate implant-exp
cd /workspace/implantsj

MODEL="vit_base_patch16_224"
INITIAL_CHECKPOINT="/workspace/checkpoints/pretrained.safetensors"
CHECKPOINT_FORMAT="safetensors"   # auto | ckpt | safetensors
TRAIN_DATA="/workspace/manifests/train.csv"
VALID_DATA="/workspace/manifests/valid.csv"
TEST_DATA="/workspace/manifests/test.csv"
RUN_DIR="/workspace/runs/experiment_01"
```

## CE 학습

```bash
python tools/run.py experiment \
  --experiment-name ce_adamw_01 \
  --model "$MODEL" \
  --checkpoint "$INITIAL_CHECKPOINT" \
  --checkpoint-format "$CHECKPOINT_FORMAT" \
  --train-data "$TRAIN_DATA" \
  --valid-data "$VALID_DATA" \
  --scope all \
  --loss ce \
  --optimizer adamw \
  --scheduler cosine \
  --learning-rate 5e-5 \
  --epochs 15 \
  --batch-size 64 \
  --save-format safetensors \
  --output-dir "$RUN_DIR"
```

SGD 또는 step scheduler 실험은 해당 인자만 바꾼다.

```bash
--optimizer sgd --learning-rate 1e-3 --momentum 0.9 \
--scheduler step --scheduler-step-size 5 --scheduler-gamma 0.1
```

## SupCon 학습

7개 closed 클래스에서 K=8, P=7인 batch 56 예시다.

```bash
RUN_DIR="/workspace/runs/supcon_closed_01"

python tools/run.py experiment \
  --experiment-name supcon_closed_01 \
  --model "$MODEL" \
  --checkpoint "$INITIAL_CHECKPOINT" \
  --checkpoint-format "$CHECKPOINT_FORMAT" \
  --train-data "$TRAIN_DATA" \
  --valid-data "$VALID_DATA" \
  --scope closed \
  --loss supcon \
  --temperature 0.07 \
  --optimizer adamw \
  --scheduler cosine \
  --epochs 15 \
  --batch-size 56 \
  --samples-per-class 8 \
  --save-format safetensors \
  --output-dir "$RUN_DIR"
```

## Triplet 학습

```bash
RUN_DIR="/workspace/runs/triplet_all_01"

python tools/run.py experiment \
  --experiment-name triplet_all_01 \
  --model "$MODEL" \
  --checkpoint "$INITIAL_CHECKPOINT" \
  --checkpoint-format "$CHECKPOINT_FORMAT" \
  --train-data "$TRAIN_DATA" \
  --valid-data "$VALID_DATA" \
  --scope all \
  --loss triplet \
  --triplet-margin 0.2 \
  --optimizer adamw \
  --scheduler cosine \
  --epochs 15 \
  --batch-size 64 \
  --samples-per-class 8 \
  --save-format ckpt \
  --output-dir "$RUN_DIR"
```

## 체크포인트 형식 선택

기존 `.pt`, `.pth`, `.ckpt` 파일:

```bash
--checkpoint /workspace/checkpoints/model.ckpt --checkpoint-format ckpt
```

가중치-only 파일:

```bash
--checkpoint /workspace/checkpoints/model.safetensors \
--checkpoint-format safetensors
```

출력 형식은 별도 인자다.

```bash
--save-format ckpt
# 또는
--save-format safetensors
```

## Embedding 추출

CE 체크포인트:

```bash
CHECKPOINT="$RUN_DIR/best.safetensors"

python -m src.evaluation embed \
  --model "$MODEL" \
  --adapter classifier_checkpoint \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-format safetensors \
  --manifest "$TEST_DATA" \
  --output "$RUN_DIR/test_embeddings.npz" \
  --batch-size 128 \
  --device auto \
  --overwrite
```

SupCon/triplet 체크포인트는 adapter만 바꾼다.

```bash
--adapter supcon_checkpoint --embedding-source backbone
```

학습이 만든 spec을 그대로 사용할 수도 있다.

```bash
python -m src.evaluation embed \
  --model-spec "$RUN_DIR/model_spec.json" \
  --manifest "$TEST_DATA" \
  --output "$RUN_DIR/test_embeddings.npz" \
  --overwrite
```

## Prototype 평가

```bash
python -m src.evaluation evaluate \
  --model-name experiment_01 \
  --support-embeddings "$RUN_DIR/support_embeddings.npz" \
  --gt-embeddings "$RUN_DIR/gt_embeddings.npz" \
  --query-embeddings "$RUN_DIR/test_embeddings.npz" \
  --labels-json /workspace/manifests/labels.json \
  --query-name test \
  --output-dir "$RUN_DIR/evaluation" \
  --ss-count 5 \
  --seed 2026 \
  --allow-noncanonical-labels
```

각 명령은 한 번에 실험 하나만 실행한다. 여러 실험은 위 명령을 각각 복사하고
인자와 출력 폴더를 다르게 지정한다.

## t-SNE 저장

시각화 파일은 실행한 서버의 `--output-dir`에 직접 저장된다.

```bash
FIGURE_DIR="$RUN_DIR/figures/tsne"

python tools/run.py visualize tsne \
  --model-name experiment_01 \
  --embedding "valid=$RUN_DIR/valid_embeddings.npz" \
  --embedding "test=$RUN_DIR/test_embeddings.npz" \
  --output-dir "$FIGURE_DIR" \
  --max-points 5000
```

결과는 `$FIGURE_DIR/tsne.png`, `tsne_coordinates.csv`,
`sampled_paths.txt`, `tsne_metadata.json`으로 남는다.
