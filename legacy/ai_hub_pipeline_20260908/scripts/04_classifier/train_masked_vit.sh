#!/usr/bin/env bash
# =============================================================
# Stage 3-<2>. Classifier — masked-vit 방식 (masked_vit 패키지)
#   설정: configs/04_masked_vit.yaml
#   학습(band-only crop) → test는 원본 이미지(input-mode original)로 평가
#   + 비교용 band 모드 평가, 각각 attention map/gradient rollout n개 저장
# 사용법:
#   bash train_masked_vit.sh
# =============================================================
set -euo pipefail
source "$(dirname "$0")/../common.sh"

CONFIG="${CONFIG:-${CONFIG_DIR}/04_masked_vit.yaml}"
log "=== Stage 3-2. masked-vit classifier (train + original-image eval) ==="
run_with_config "${RUN_MASKED_VIT}" "${CONFIG}"
log "masked-vit 완료"
