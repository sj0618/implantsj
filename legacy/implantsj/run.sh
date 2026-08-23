#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
cd "$IMPLANTSJ_DIR"

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

make_manifests() {
  local cmd=("$PYTHON" -m implantsj.manifests --output-dir "$MANIFEST_DIR" --closed-labels "$CLOSED_LABELS_FILE" --seed "$SEED")
  if [[ -n "$SOURCE_MANIFEST" ]]; then
    cmd+=(--source-manifest "$SOURCE_MANIFEST")
  else
    cmd+=(--train-root "$TRAIN_ROOT" --test-root "$TEST_ROOT")
  fi
  if [[ -n "$GT_MANIFEST" ]]; then
    cmd+=(--gt-manifest "$GT_MANIFEST")
  elif [[ -n "$GT_ROOT" ]]; then
    cmd+=(--gt-root "$GT_ROOT")
  fi
  [[ -n "$ALIASES" ]] && cmd+=(--aliases "$ALIASES")
  [[ "$ALLOW_INCOMPLETE" == "1" ]] && cmd+=(--allow-incomplete)
  run_cmd "${cmd[@]}"
}

save_initial() {
  if [[ "$DRY_RUN" != "1" && -f "$INITIAL_CHECKPOINT" ]]; then
    printf 'keep existing initial checkpoint: %s\n' "$INITIAL_CHECKPOINT"
    return
  fi
  run_cmd "$PYTHON" -m implantsj.models save-initial \
    --architecture "$ARCH" --output "$INITIAL_CHECKPOINT"
}

model_values() {
  "$PYTHON" - "$MODELS_JSON" "$1" <<'PY'
import sys
from implantsj.models import load_model_spec

spec = load_model_spec(sys.argv[1], sys.argv[2])
print("|".join(map(str, (
    spec.architecture,
    spec.training_kind,
    spec.train_scope,
    spec.initial_checkpoint or "",
    int(spec.pretrained),
    spec.input_size,
    spec.projection_dim,
    spec.projection_hidden_dim,
))))
PY
}

train_model() {
  local model_name="$1" values architecture loss scope initial pretrained input_size projection_dim projection_hidden_dim
  values="$(model_values "$model_name")"
  IFS='|' read -r architecture loss scope initial pretrained input_size projection_dim projection_hidden_dim <<< "$values"
  if [[ "$loss" == "none" ]]; then
    printf 'skip train: %s has training_kind=none\n' "$model_name"
    return
  fi
  local cmd=(
    "$PYTHON" -m implantsj.training
    --experiment-name "$model_name"
    --model "$architecture"
    --train-data "$MANIFEST_DIR/train.csv"
    --valid-data "$MANIFEST_DIR/valid.csv"
    --test-data "$MANIFEST_DIR/test.csv"
    --scope "$scope"
    --loss "$loss"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --samples-per-class "$K_SAMPLES_PER_CLASS"
    --learning-rate "$LR"
    --weight-decay "$WEIGHT_DECAY"
    --input-size "$input_size"
    --projection-dim "$projection_dim"
    --projection-hidden-dim "$projection_hidden_dim"
    --num-workers "$NUM_WORKERS"
    --seed "$SEED"
    --device "$DEVICE"
    --save-format ckpt
    --output-dir "$OUTPUT_DIR/$model_name"
  )
  [[ "$pretrained" == "1" ]] && cmd+=(--pretrained) || cmd+=(--no-pretrained)
  [[ -n "$INITIAL_CHECKPOINT" ]] && cmd+=(--checkpoint "$INITIAL_CHECKPOINT" --checkpoint-format auto)
  run_cmd "${cmd[@]}"
}

embed_one() {
  local model_name="$1" split="$2"
  local values architecture loss scope initial pretrained input_size projection_dim projection_hidden_dim checkpoint
  values="$(model_values "$model_name")"
  IFS='|' read -r architecture loss scope initial pretrained input_size projection_dim projection_hidden_dim <<< "$values"
  if [[ "$loss" == "none" ]]; then
    checkpoint="$INITIAL_CHECKPOINT"
  else
    checkpoint="$OUTPUT_DIR/$model_name/checkpoints/$CANONICAL_EPOCH_CKPT"
  fi
  local cmd=(
    "$PYTHON" -m implantsj.evaluation embed
    --model-spec "$MODELS_JSON"
    --model-name "$model_name"
    --manifest "$MANIFEST_DIR/$split.csv"
    --output "$OUTPUT_DIR/$model_name/embeddings/$split.npz"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --device "$DEVICE"
  )
  cmd+=(--checkpoint "$checkpoint" --checkpoint-format ckpt)
  [[ "$OVERWRITE_EMBEDDINGS" == "1" ]] && cmd+=(--overwrite)
  run_cmd "${cmd[@]}"
}

embed_model() {
  local model_name="$1" split="${2:-all}"
  if [[ "$split" == "all" ]]; then
    local item
    for item in support valid test full; do
      embed_one "$model_name" "$item"
    done
    if [[ "$DRY_RUN" == "1" || -f "$MANIFEST_DIR/gt.csv" ]]; then
      embed_one "$model_name" gt
    fi
  else
    embed_one "$model_name" "$split"
  fi
}

evaluate_one() {
  local model_name="$1" query="$2"
  local cmd=(
    "$PYTHON" -m implantsj.evaluation evaluate
    --model-name "$model_name"
    --support-embeddings "$OUTPUT_DIR/$model_name/embeddings/support.npz"
    --query-embeddings "$OUTPUT_DIR/$model_name/embeddings/$query.npz"
    --labels-json "$MANIFEST_DIR/labels.json"
    --query-name "$query"
    --output-dir "$OUTPUT_DIR/$model_name/eval/$query"
    --prototypes "$PROTOTYPES"
    --distances "$DISTANCES"
    --ss-count "$SS_COUNT"
    --seed "$SEED"
  )
  [[ ",${PROTOTYPES}," == *",gt,"* ]] && cmd+=(--gt-embeddings "$OUTPUT_DIR/$model_name/embeddings/gt.npz")
  [[ "$ALLOW_INCOMPLETE" == "1" ]] && cmd+=(--allow-noncanonical-labels)
  run_cmd "${cmd[@]}"
}

evaluate_model() {
  local model_name="$1" query="${2:-all}"
  if [[ "$query" == "all" ]]; then
    evaluate_one "$model_name" valid
    evaluate_one "$model_name" test
  else
    evaluate_one "$model_name" "$query"
  fi
}

make_tsne() {
  local model_name="$1"
  run_cmd "$PYTHON" -m implantsj.visualization tsne \
    --model-name "$model_name" \
    --embedding "train=$OUTPUT_DIR/$model_name/embeddings/support.npz" \
    --embedding "valid=$OUTPUT_DIR/$model_name/embeddings/valid.npz" \
    --embedding "test=$OUTPUT_DIR/$model_name/embeddings/test.npz" \
    --labels-json "$MANIFEST_DIR/labels.json" \
    --output-dir "$OUTPUT_DIR/$model_name/tsne" \
    --max-points "$TSNE_MAX_POINTS" \
    --seed "$SEED"
}

make_report() {
  run_cmd "$PYTHON" -m implantsj.reporting \
    --eval-root "$OUTPUT_DIR" \
    --models "$MODEL_NAMES" \
    --roles "$MODEL_ROLES" \
    --output-dir "$OUTPUT_DIR/report" \
    --expected-rows "$EXPECTED_ROWS"
}

run_all() {
  make_manifests
  save_initial
  local model_name
  IFS=',' read -r -a models <<< "$MODEL_NAMES"
  for model_name in "${models[@]}"; do
    train_model "$model_name"
    embed_model "$model_name"
    evaluate_model "$model_name"
    make_tsne "$model_name"
  done
  make_report
}

usage() {
  printf '%s\n' \
    'Usage: ./run.sh manifests' \
    '       ./run.sh save_initial' \
    '       ./run.sh train MODEL' \
    '       ./run.sh embed MODEL [support|gt|valid|test|full|all]' \
    '       ./run.sh eval MODEL [valid|test|all]' \
    '       ./run.sh test MODEL' \
    '       ./run.sh tsne MODEL' \
    '       ./run.sh report' \
    '       ./run.sh all'
}

case "${1:-}" in
  manifests) make_manifests ;;
  save_initial) save_initial ;;
  train) train_model "${2:?MODEL is required}" ;;
  embed) embed_model "${2:?MODEL is required}" "${3:-all}" ;;
  eval) evaluate_model "${2:?MODEL is required}" "${3:-all}" ;;
  test) evaluate_model "${2:?MODEL is required}" test ;;
  tsne) make_tsne "${2:?MODEL is required}" ;;
  report) make_report ;;
  all) run_all ;;
  *) usage; exit 2 ;;
esac
