#!/usr/bin/env bash
# ④ (이미지, 마스크) 쌍을 train/val/test 7:2:1 분할
# ※ 선행: SAM1 마스크 중 잘 된 것을 사람이 선별해 seg_dataset_new 폴더 구성 완료
set -euo pipefail
source "$(dirname "$0")/../common.sh"
CONFIG="${CONFIG:-${CONFIG_DIR}/02_segmentation.yaml}"
log "=== Seg ④ 데이터셋 7:2:1 분할 ==="
run_with_config "${SEG_SPLIT}" "${CONFIG}"
