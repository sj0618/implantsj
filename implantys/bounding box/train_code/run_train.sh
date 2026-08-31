#!/usr/bin/env bash
# 규칙 4의 산출물을 한 번에 만들어내는 공용 학습 스크립트.
#
#   변환 -> 학습 -> 곡선 -> best epoch -> 시각화 -> REPORT -> 검증
#
# 마지막 verify_run.py가 실패하면 전체가 실패한다. 규칙 4의
# "하나라도 빠지면 실패로 간주한다"를 여기서 강제한다.
#
# 사용 (pod):
#   ./run_train.sh 01 1e-4          # FT-106, LR 1e-4  -> results/01/lr_1e-4
#   ./run_train.sh 01 5e-5
#   ./run_train.sh 01 1e-5
#   ./run_train.sh 02 <확정LR>       # FT-206          -> results/02/lr_<확정LR>
#
# 02는 01+02 데이터를 합쳐 쓴다 (train 144 / val 62).

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/common/config.sh"

ROUND="${1:?사용: ./run_train.sh <round: 01|02> <lr: 1e-4|5e-5|1e-5>}"
LR="${2:?LR을 지정할 것}"

# 라운드 -> 사용할 데이터 폴더 (RULES.md 3번: dataset2엔 원본만, 병합은 변환 시점에)
case "$ROUND" in
  01) ROUNDS="01"    ;;   # FT-106
  02) ROUNDS="01 02" ;;   # FT-206
  *)  echo "[ERROR] 알 수 없는 라운드: $ROUND" >&2; exit 1 ;;
esac

RUN_DIR="${RESULT_ROOT}/${ROUND}/lr_${LR}"
CFG="${TRAIN_CODE}/${ROUND}/config_lr${LR}.yaml"
PY="${PYTHON:-python}"

echo "══════════════════════════════════════════════════════════"
echo " round     : ${ROUND}  (데이터: ${ROUNDS})"
echo " lr        : ${LR}"
echo " run dir   : ${RUN_DIR}"
echo " env       : ${ENV_NAME}"
echo "══════════════════════════════════════════════════════════"

require_gdino
setup_pod_env
[[ -f "$CFG" ]] || { echo "[ERROR] 설정 없음: $CFG" >&2; exit 1; }

mkdir -p "$RUN_DIR"

# ── 1. COCO -> repo CSV (라운드 병합 + leakage 검사) ─────────────────────────
echo; echo "[1/6] 데이터 변환"
$PY "${TRAIN_CODE}/common/coco_to_csv.py" \
    --data $ROUNDS \
    --data-root "$DATA_ROOT" \
    --out-root "$REPO_DATA"

# ── 2. 학습 ─────────────────────────────────────────────────────────────────
echo; echo "[2/6] 학습"
# TRAIN_PATCH.md 참조: repo train.py가 RunLogger를 통해 아래 경로에 기록한다.
export RUN_DIR DATA_ROOT GDINO_ROOT
export ROUNDS_CSV="${ROUNDS// /,}"
export ROUNDS="$ROUNDS_CSV"
export PYTHONPATH="${TRAIN_CODE}/common:${PYTHONPATH:-}"

cd "$GDINO_ROOT"
$PY train.py --config "$CFG"      # TODO(pod): repo의 실제 진입점/인자에 맞출 것
cd - >/dev/null

# ── 3. 곡선 ─────────────────────────────────────────────────────────────────
echo; echo "[3/6] 학습 곡선"
$PY "${TRAIN_CODE}/common/make_curves.py" --run "$RUN_DIR"

# ── 4. best epoch (val로만. RULES 2·5) ──────────────────────────────────────
echo; echo "[4/6] best epoch 선택"
$PY "${TRAIN_CODE}/common/select_best.py" --run "$RUN_DIR"
BEST=$($PY -c "import json;print(json.load(open('${RUN_DIR}/best_epoch.json'))['best_epoch'])")
CKPT="${GDINO_ROOT}/weights/checkpoint_epoch_${BEST}.pth"

if [[ ! -f "$CKPT" ]]; then
  echo "[ERROR] best epoch ${BEST}의 checkpoint가 없다: ${CKPT}" >&2
  echo "        save_frequency 때문일 수 있다. TRAIN_PATCH.md의 'save_frequency 문제' 참조." >&2
  exit 1
fi

# ── 5. best epoch 시각화 + attention (val 기준. test는 열지 않는다) ─────────
echo; echo "[5/6] best epoch 시각화 (val)"
$PY "${EVAL_CODE}/common/eval_coco.py" \
    --out "$RUN_DIR" --split val --data-root "$DATA_ROOT" \
    --gdino-root "$GDINO_ROOT" --lora "$CKPT"
$PY "${EVAL_CODE}/common/viz_pred.py" \
    --run "$RUN_DIR" --split val --data-root "$DATA_ROOT"
$PY "${EVAL_CODE}/common/attention_map.py" \
    --out "$RUN_DIR" --split val --data-root "$DATA_ROOT" \
    --gdino-root "$GDINO_ROOT" --lora "$CKPT" \
  || echo "[warn] attention 미구현 — attention_map.py의 TODO 참조 (RULES 7 위반 상태)"

# ── 6. REPORT + 검증 ────────────────────────────────────────────────────────
echo; echo "[6/6] REPORT 생성 및 검증"
$PY "${TRAIN_CODE}/common/make_report.py" --run "$RUN_DIR"
$PY "${TRAIN_CODE}/common/verify_run.py" --run "$RUN_DIR" --kind train

echo
echo "══════════════════════════════════════════════════════════"
echo " 완료: ${RUN_DIR}"
echo " best epoch ${BEST} | REPORT.md 확인"
echo
echo " ⚠️ test는 아직 열지 않았다. LR 확정 후 run_eval.sh로 딱 한 번. (RULES 2)"
echo "══════════════════════════════════════════════════════════"
