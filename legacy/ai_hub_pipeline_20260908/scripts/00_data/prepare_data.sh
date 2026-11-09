#!/usr/bin/env bash
# =============================================================
# Stage 0-a. 데이터 준비 — 파이프라인의 입구. 가장 먼저 실행할 것.
#   설정: configs/00_data.yaml
#   데이터셋 폴더 → manifest CSV 3개 (full42 / closed27 / open15)
#   band mask가 있으면 mask_path 컬럼까지 채운다
# 사용법:
#   bash prepare_data.sh
#   ※ Stage 0(segmentation)으로 band mask를 만든 뒤 다시 실행하면
#     mask_path가 채워진다 (band-* 백본과 band-vit에 필요)
# =============================================================
set -euo pipefail
source "$(dirname "$0")/../common.sh"

CONFIG="${CONFIG:-${CONFIG_DIR}/00_data.yaml}"
log "=== Stage 0-a. 데이터 준비 ==="
run_with_config "${PREPARE_DATA}" "${CONFIG}"
log "데이터 준비 완료"
