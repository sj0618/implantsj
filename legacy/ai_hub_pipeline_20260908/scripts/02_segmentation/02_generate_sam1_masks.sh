#!/usr/bin/env bash
# ② 순정 SAM(vit_h)으로 초기 마스크 자동 생성 + 임플란트 형태 필터
# 완료 후 사람이 잘 생성된 마스크를 선별해 image:mask 1:1 폴더(seg_dataset_new) 구성
set -euo pipefail
source "$(dirname "$0")/../common.sh"
CONFIG="${CONFIG:-${CONFIG_DIR}/02_segmentation.yaml}"
log "=== Seg ② SAM1 초기 마스크 생성 ==="
run_with_config "${SEG_SAM1}" "${CONFIG}"
log "다음 단계: 생성된 마스크 중 잘 된 것을 사람이 선별하여"
log "  seg_data/seg_dataset_new/{images,masks}/<클래스>/ 구조로 구성 후 03_split_dataset.sh 실행"
