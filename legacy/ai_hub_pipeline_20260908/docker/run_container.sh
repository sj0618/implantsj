#!/usr/bin/env bash
# =====================================================================
# [폐쇄망에서 실행]
# host 작업 폴더(WORK) 아래 data/ outputs/ checkpoints/ seg_data/ 를 컨테이너의
# 같은 이름 폴더에 붙여 스테이지 스크립트를 돌린다.
#
#   WORK=/workspace/ai_hub ./docker/run_container.sh                       # 셸
#   WORK=/workspace/ai_hub ./docker/run_container.sh bash scripts/01_backbone/finetune_backbones.sh
#   WORK=... SET="grid.backbone=vit train.epochs=1" ./docker/run_container.sh bash scripts/01_backbone/finetune_backbones.sh
#
# 폴더 대응 (yaml 의 상대경로가 그대로 맞는다):
#   $WORK/data        -> data/        (data/datas/<클래스>/*.jpg, manifest CSV)
#   $WORK/outputs     -> outputs/
#   $WORK/checkpoints -> checkpoints/ (이미지에 구운 SAM 가중치는 시작 때 여기로 링크된다)
#   $WORK/seg_data    -> seg_data/
# =====================================================================
set -euo pipefail

IMAGE="${IMAGE:-ai_hub_pipeline}"
TAG="${TAG:-latest}"
WORK="${WORK:-/workspace/ai_hub}"
NAME="${NAME:-ai_hub_pipeline}"
ROOT=/workspace/ai_hub_pipeline

for sub in data outputs checkpoints seg_data; do
  mkdir -p "$WORK/$sub"
done
if [[ -z "$(ls -A "$WORK/data" 2>/dev/null)" ]]; then
  echo "!! $WORK/data 가 비어 있습니다. data/datas/<클래스>/*.jpg 구조로 데이터를 넣으세요." >&2
fi

GPU_ARGS=()
if docker info 2>/dev/null | grep -qi nvidia || command -v nvidia-smi >/dev/null 2>&1; then
  GPU_ARGS=(--gpus all)
  echo "GPU 모드로 실행합니다."
else
  echo "!! NVIDIA GPU 를 찾지 못해 CPU 로 실행합니다 (학습은 매우 느립니다)."
fi
echo "작업 폴더: $WORK  -> $ROOT/{data,outputs,checkpoints,seg_data}"
echo

# --ipc=host : DataLoader worker 가 공유 메모리 부족으로 죽지 않게 한다.
# PROTOCOL / SET / CONFIG 은 scripts/common.sh 가 읽는 오버라이드 변수라 그대로 넘긴다.
exec docker run --rm -it \
  --name "$NAME" \
  --ipc=host \
  ${GPU_ARGS[@]+"${GPU_ARGS[@]}"} \
  -v "$WORK/data":"$ROOT/data" \
  -v "$WORK/outputs":"$ROOT/outputs" \
  -v "$WORK/checkpoints":"$ROOT/checkpoints" \
  -v "$WORK/seg_data":"$ROOT/seg_data" \
  -e "PROTOCOL=${PROTOCOL:-}" \
  -e "SET=${SET:-}" \
  -e "CONFIG=${CONFIG:-}" \
  "$IMAGE:$TAG" "$@"
