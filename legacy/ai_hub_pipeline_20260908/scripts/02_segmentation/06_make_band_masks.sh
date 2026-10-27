#!/usr/bin/env bash
# ⑦ 예측 마스크에서 band(고리) 영역 추출 → band-vit/band-dinov2 학습 입력
set -euo pipefail
source "$(dirname "$0")/../common.sh"
CONFIG="${CONFIG:-${CONFIG_DIR}/02_segmentation.yaml}"
log "=== Seg ⑦ band 마스크 추출 ==="
run_with_config "${SEG_BAND}" "${CONFIG}"
