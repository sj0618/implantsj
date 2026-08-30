#!/usr/bin/env bash
# =====================================================================
# [폐쇄망에서 실행]
# 데이터와 산출물 폴더를 연결해 실험 컨테이너를 띄운다.
#
#   ./docker/run_container.sh              대화형 실험 메뉴
#   ./docker/run_container.sh bash         컨테이너 안 셸
#   ./docker/run_container.sh ./run.sh check
#
# 경로가 다르면 환경변수로 지정한다.
#   DATA=/mnt/data OUT=/mnt/outputs ./docker/run_container.sh
#
# 데이터는 기본적으로 읽기와 쓰기를 허용한다. 원본 TL1/TS1을 ImageFolder로
# 변환할 때 데이터 폴더에 파일을 만들기 때문이다. 원본을 보호하려면 DATA_RO=1을 준다.
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

# GPU를 찾으면 사용하고, 없으면 CPU로 실행한다. 이미지 빌드 검증에도 쓸 수 있다.
GPU_ARGS=()
if docker info 2>/dev/null | grep -qi nvidia || command -v nvidia-smi >/dev/null 2>&1; then
  GPU_ARGS=(--gpus all)
  echo "GPU 모드로 실행합니다."
else
  echo "!! NVIDIA GPU 를 찾지 못해 CPU 로 실행합니다(학습은 매우 느립니다)."
fi

DATA_MOUNT="$DATA:/workspace/data"
if [[ "${DATA_RO:-0}" == "1" ]]; then
  DATA_MOUNT="$DATA_MOUNT:ro"
  echo "데이터  : $DATA  -> /workspace/data (읽기 전용)"
else
  echo "데이터  : $DATA  -> /workspace/data (읽기·쓰기, 데이터셋 변환 가능)"
fi
echo "산출물  : $OUT   -> /workspace/outputs"
echo

# --ipc=host: DataLoader worker가 공유 메모리 부족으로 종료되지 않게 한다.
# GPU가 없어 CPU로 실행할 때도 적용한다.
exec docker run --rm -it \
  --name "$NAME" \
  --ipc=host \
  ${GPU_ARGS[@]+"${GPU_ARGS[@]}"} \
  -v "$DATA_MOUNT" \
  -v "$OUT":/workspace/outputs \
  -e DATA_ROOT=/workspace/data \
  -e OUTPUT_ROOT=/workspace/outputs \
  "$IMAGE:$TAG" "$@"
