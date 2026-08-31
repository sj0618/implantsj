#!/usr/bin/env bash
# 공용 경로 설정. 로컬(코드 작성)과 pod(학습 실행) 양쪽에서 동작한다. (RULES.md 8번)
#
# 사용: 다른 스크립트 맨 위에서  source "$(dirname "$0")/common/config.sh"
#
# pod에서 경로가 다르면 아래 값을 환경변수로 덮어쓴다:
#   export GDINO_ROOT=/workspace/Grounding-Dino-FineTuning
#   export PROJ_ROOT=/workspace

set -euo pipefail

# ── 환경 감지 ────────────────────────────────────────────────────────────────
if [[ -d /workspace ]]; then
  ENV_NAME="pod"
  : "${PROJ_ROOT:=/workspace}"
  : "${GDINO_ROOT:=/workspace/Grounding-Dino-FineTuning}"
  : "${VENV:=/workspace/implant-env}"
else
  ENV_NAME="local"
  : "${PROJ_ROOT:=/Users/gim-yesong/Desktop/implant}"
  : "${GDINO_ROOT:=}"          # 로컬엔 repo 없음. 학습은 pod에서만.
  : "${VENV:=}"
fi

# ── 프로젝트 경로 ────────────────────────────────────────────────────────────
DATA_ROOT="${PROJ_ROOT}/dataset2"
RESULT_ROOT="${PROJ_ROOT}/results"
TRAIN_CODE="${PROJ_ROOT}/train_code"
EVAL_CODE="${PROJ_ROOT}/eval_code"

# repo가 CSV/이미지를 기대하는 위치 (coco_to_csv.py의 출력 대상)
REPO_DATA="${GDINO_ROOT}/multimodal-data/implant_dataset"
BASE_WEIGHTS="${GDINO_ROOT}/weights/groundingdino_swint_ogc.pth"
MODEL_CFG="${GDINO_ROOT}/groundingdino/config/GroundingDINO_SwinT_OGC.py"

TEXT_PROMPT="implant"

# ── pod 전용 준비 ────────────────────────────────────────────────────────────
# torch lib 경로. 없으면 ms_deform_attn 로딩이 깨진다. (인수인계 문서 참조)
setup_pod_env() {
  [[ "$ENV_NAME" != "pod" ]] && return 0
  [[ -n "$VENV" && -f "$VENV/bin/activate" ]] && source "$VENV/bin/activate"
  local torch_lib
  torch_lib="$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib"
  export LD_LIBRARY_PATH="${torch_lib}:${LD_LIBRARY_PATH:-}"
  export TOKENIZERS_PARALLELISM=false
  # groundingdino를 repo 밖 어디서 실행해도 import되게 한다.
  # (setup_pod.sh의 pip install -e .가 있어도 editable finder가 안 잡히는 경우가 있다)
  [[ -n "$GDINO_ROOT" ]] && export PYTHONPATH="${GDINO_ROOT}:${PYTHONPATH:-}"
}

require_gdino() {
  if [[ -z "$GDINO_ROOT" || ! -d "$GDINO_ROOT" ]]; then
    echo "[ERROR] GDINO_ROOT를 찾을 수 없음: '${GDINO_ROOT}'" >&2
    echo "        학습·추론은 pod에서만 가능하다. (RULES.md 8번)" >&2
    echo "        pod이면: export GDINO_ROOT=/workspace/Grounding-Dino-FineTuning" >&2
    exit 1
  fi
}
