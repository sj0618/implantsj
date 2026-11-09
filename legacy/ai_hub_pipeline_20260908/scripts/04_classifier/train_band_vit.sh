#!/usr/bin/env bash
# =============================================================
# Stage 3-<1>. Classifier — band-vit 방식 (band_vit 패키지)
#   설정: configs/04_band_vit.yaml (alpha/lambda/block sweep + 시각화 개수)
#   각 조합마다: 학습 → test 평가(attention map n개) → gradient rollout n개
# 사용법:
#   bash train_band_vit.sh                       # yaml에 적힌 전 조합 (3x3x2=18)
#   # 조합 하나만 (sweep은 의무가 아니다)
#   SET="sweep.alpha=8 sweep.lambda=0.8 sweep.block=6" bash train_band_vit.sh
# =============================================================
set -euo pipefail
source "$(dirname "$0")/../common.sh"

CONFIG="${CONFIG:-${CONFIG_DIR}/04_band_vit.yaml}"
log "=== Stage 3-1. band-vit classifier sweep ==="
run_with_config "${RUN_BAND_VIT}" "${CONFIG}"
log "band-vit sweep 완료"
