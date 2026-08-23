#!/usr/bin/env bash
# =====================================================================
# [폐쇄망 안에서 실행]
# 데이터와 산출물 폴더를 붙여서 실험 컨테이너를 띄운다.
#
#   ./docker/run_container.sh              대화형 실험 메뉴
#   ./docker/run_container.sh bash         컨테이너 안 셸
#   ./docker/run_container.sh ./run.sh check
#
# 경로가 다르면 환경변수로 넘긴다:
#   DATA=/mnt/data OUT=/mnt/outputs ./docker/run_container.sh
#
# 데이터는 기본 읽기·쓰기로 붙는다. 원본 TL1/TS1 을 ImageFolder 로 변환하는
# 작업이 데이터 쪽에 파일을 만들기 때문이다. 학습만 돌려서 원본을 절대
# 건드리고 싶지 않을 때는 DATA_RO=1 을 준다.
# =====================================================================
set -euo pipefail

IMAGE="${IMAGE:-implantsj}"
TAG="${TAG:-latest}"
DATA="${DATA:-/workspace/data}"
OUT="${OUT:-/workspace/outputs}"
NAME="${NAME:-implantsj}"

if [[ ! -d "$DATA" ]]; then
  echo "!! 데이터 경로가 없습니다: $DATA" >&2
  echo "   DATA=/실제/데이터/경로 ./docker/run_container.sh 로 지정하세요." >&2
  exit 1
fi
mkdir -p "$OUT"

# GPU 옵션은 있으면 쓰고 없으면 CPU 로 떨어진다(빌드 검증용).
GPU_ARGS=()
if docker info 2>/dev/null | grep -qi nvidia || command -v nvidia-smi >/dev/null 2>&1; then
  GPU_ARGS=(--gpus all --ipc=host)
  echo "GPU 모드로 실행합니다."
else
  echo "!! NVIDIA GPU 를 찾지 못해 CPU 로 실행합니다(학습은 매우 느립니다)."
fi

DATA_MOUNT="$DATA:/workspace/data"
if [[ "${DATA_RO:-0}" == "1" ]]; then
  DATA_MOUNT="$DATA_MOUNT:ro"
  echo "데이터  : $DATA  -> /workspace/data (읽기 전용)"
else
  echo "데이터  : $DATA  -> /workspace/data (읽기·쓰기 — 데이터셋 변환 가능)"
fi
echo "산출물  : $OUT   -> /workspace/outputs"
echo

# --ipc=host: DataLoader worker 가 공유메모리 부족으로 죽는 것을 막는다.
exec docker run --rm -it \
  --name "$NAME" \
  "${GPU_ARGS[@]}" \
  -v "$DATA_MOUNT" \
  -v "$OUT":/workspace/outputs \
  -e DATA_ROOT=/workspace/data \
  -e OUTPUT_ROOT=/workspace/outputs \
  "$IMAGE:$TAG" "$@"
