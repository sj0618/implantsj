#!/usr/bin/env bash
# =====================================================================
# [폐쇄망 반입 전: 인터넷이 되는 곳에서 실행]
# ai_hub_pipeline 이미지를 빌드하고 반입용 tar.gz 로 저장한다.
#
#   ./docker/build_and_export.sh                 # SAM1 vit_h(2.5GB) 포함
#   PREFETCH_SAM1=0 ./docker/build_and_export.sh # adapter 가 이미 있어 Stage 0 ①②를 안 돌릴 때
#
# 빌드 전에 팀 자산을 weights/ 에 둔다 (있으면 굽고, 없으면 경고만):
#   weights/best_sam2.1_tiny_adapter_unet.pth
# 산출물: ai_hub_pipeline_<날짜>.tar.gz — 폐쇄망에는 이 파일만 반입한다.
# =====================================================================
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

IMAGE="${IMAGE:-ai_hub_pipeline}"
TAG="${TAG:-latest}"
OUT="${OUT:-ai_hub_pipeline_$(date +%Y%m%d).tar.gz}"
PREFETCH_SAM1="${PREFETCH_SAM1:-1}"

echo "=============================================================="
echo " 1/3  build context 확인"
echo "=============================================================="
echo "저장소 전체: $(du -sh . 2>/dev/null | cut -f1)  (.dockerignore 로 data/outputs/checkpoints 제외)"
if [[ -f weights/best_sam2.1_tiny_adapter_unet.pth ]]; then
  echo "adapter    : weights/best_sam2.1_tiny_adapter_unet.pth ($(du -sh weights/best_sam2.1_tiny_adapter_unet.pth | cut -f1)) → 이미지에 굽습니다"
else
  echo "adapter    : 없음 → 폐쇄망 안에서 Stage 0 ①~⑤ 로 만들어야 합니다 (경고만, 계속)"
fi
echo "SAM1 vit_h : $([[ "$PREFETCH_SAM1" == "1" ]] && echo '포함 (2.5GB)' || echo '생략')"
echo

echo "=============================================================="
echo " 2/3  docker build  ->  $IMAGE:$TAG"
echo "=============================================================="
# 폐쇄망 서버가 x86_64 이므로 플랫폼을 고정한다. Apple Silicon 에서도 같은 이미지를 만든다.
docker build --platform linux/amd64 \
  --build-arg "PREFETCH_SAM1=${PREFETCH_SAM1}" \
  -t "$IMAGE:$TAG" .

echo
echo "=============================================================="
echo " 3/3  docker save  ->  $OUT"
echo "=============================================================="
docker save "$IMAGE:$TAG" | gzip -c > "$OUT"
echo
echo "완료: $OUT  ($(du -sh "$OUT" | cut -f1))"
echo
echo "폐쇄망 반입 후:"
echo "  gunzip -c $(basename "$OUT") | docker load"
echo "  WORK=/경로/ai_hub ./docker/run_container.sh bash scripts/00_data/prepare_data.sh"
