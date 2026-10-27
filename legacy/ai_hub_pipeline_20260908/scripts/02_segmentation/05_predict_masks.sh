#!/usr/bin/env bash
# ⑥ 파인튜닝된 seg 모델로 전체 데이터(42클래스) 마스크 예측 + 후처리
set -euo pipefail
source "$(dirname "$0")/../common.sh"
CONFIG="${CONFIG:-${CONFIG_DIR}/02_segmentation.yaml}"
log "=== Seg ⑥ 전체 데이터 마스크 예측 ==="
run_with_config "${SEG_PREDICT}" "${CONFIG}"
