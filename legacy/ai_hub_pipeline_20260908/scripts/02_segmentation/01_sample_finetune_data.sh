#!/usr/bin/env bash
# ① train data에서 seg finetuning용 이미지 n장 샘플링 (클래스별 라운드로빈)
set -euo pipefail
source "$(dirname "$0")/../common.sh"
CONFIG="${CONFIG:-${CONFIG_DIR}/02_segmentation.yaml}"
log "=== Seg ① finetuning 데이터 샘플링 ==="
run_with_config "${SEG_SAMPLE}" "${CONFIG}"
