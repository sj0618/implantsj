#!/usr/bin/env bash
# test 평가 공용 스크립트. (RULES.md 2번: best epoch 확정 후 딱 한 번만)
#
#   추론 -> threshold 스윕(0.05~0.50) -> 시각화 -> attention -> REPORT -> 검증
#
# 사용 (pod):
#   ./run_eval.sh zeroshot                      -> results/00_zeroshot
#   ./run_eval.sh 01 lr_1e-4                    -> results/01/lr_1e-4 (test)
#   ./run_eval.sh 02 lr_5e-5                    -> results/02/lr_5e-5 (test)

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../train_code/common/config.sh"

TARGET="${1:?사용: ./run_eval.sh <zeroshot|01|02> [lr_xxx]}"
PY="${PYTHON:-python}"

if [[ "$TARGET" == "zeroshot" ]]; then
  RUN_DIR="${RESULT_ROOT}/00_zeroshot"
  LORA_ARG=()
  LABEL="zero-shot (fine-tuning 없음)"
else
  LRDIR="${2:?LR 폴더를 지정할 것. 예: lr_1e-4}"
  RUN_DIR="${RESULT_ROOT}/${TARGET}/${LRDIR}"
  BEST_JSON="${RUN_DIR}/best_epoch.json"
  [[ -f "$BEST_JSON" ]] || { echo "[ERROR] 없음: $BEST_JSON — 학습을 먼저 할 것" >&2; exit 1; }
  BEST=$($PY -c "import json;print(json.load(open('${BEST_JSON}'))['best_epoch'])")
  CKPT="${GDINO_ROOT}/weights/checkpoint_epoch_${BEST}.pth"
  [[ -f "$CKPT" ]] || { echo "[ERROR] checkpoint 없음: $CKPT" >&2; exit 1; }
  LORA_ARG=(--lora "$CKPT")
  LABEL="FT ${TARGET} ${LRDIR} (best epoch ${BEST})"
fi

echo "══════════════════════════════════════════════════════════"
echo " 평가    : ${LABEL}"
echo " run dir : ${RUN_DIR}"
echo
echo " ⚠️ test는 한 번만 연다. 이 results를 보고 epoch·threshold·하이퍼파라미터를"
echo "    되돌아가 바꾸지 않는다. (RULES.md 2번)"
echo "══════════════════════════════════════════════════════════"

require_gdino
setup_pod_env
mkdir -p "$RUN_DIR"

echo; echo "[1/4] 추론 + threshold 스윕 (test 48장)"
$PY "${EVAL_CODE}/common/eval_coco.py" \
    --out "$RUN_DIR" --split test --data-root "$DATA_ROOT" \
    --gdino-root "$GDINO_ROOT" "${LORA_ARG[@]}"

echo; echo "[2/4] 예측 + GT 오버레이"
$PY "${EVAL_CODE}/common/viz_pred.py" \
    --run "$RUN_DIR" --split test --data-root "$DATA_ROOT"

echo; echo "[3/4] attention map (FN 케이스 우선)"
$PY "${EVAL_CODE}/common/attention_map.py" \
    --out "$RUN_DIR" --split test --data-root "$DATA_ROOT" \
    --gdino-root "$GDINO_ROOT" "${LORA_ARG[@]}" \
  || echo "[warn] attention 미구현 — attention_map.py의 TODO 참조 (RULES 7 위반 상태)"

echo; echo "[4/4] 검증"
# attention·REPORT는 현재 보류 상태(사용자 결정: 세 숫자 먼저, attention·LR 스윕은 그 다음).
# 그래서 verify는 여기서 실패한다. 지표 자체는 위에서 이미 나왔으므로 죽이지 않는다.
$PY "${TRAIN_CODE}/common/verify_run.py" --run "$RUN_DIR" --kind eval \
  || echo "[warn] 규칙 4 산출물 미완 — attention/REPORT 보류 중이라 예상된 실패다.
        지표(metrics.json / threshold_sweep.csv)는 정상 생성됐다."

echo
echo "완료: ${RUN_DIR}"
echo "세 모델이 다 끝나면: python \"${EVAL_CODE}/common/compare_models.py\""
