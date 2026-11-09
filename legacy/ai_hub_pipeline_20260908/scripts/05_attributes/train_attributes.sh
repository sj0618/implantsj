#!/usr/bin/env bash
# =============================================================
# Stage 4. Attributes — feature별 모델 + spec table 매칭
#   설정: configs/05_attributes.yaml
#   학습: train.mode = shared(동시) | separate(feature마다 따로)
#   판정: fuse.mode  = learned(가중치 자동 학습) | min_match(n개 이상 일치)
#   출력: [N,C] 점수 행렬 + predictions.csv → Stage 5에서 다른 브랜치와 결합
# 사용법:
#   bash train_attributes.sh
#   PROTOCOL=full42 bash train_attributes.sh
#   SET="grid.backbone=cnn" bash train_attributes.sh                    # 백본 하나만
#   SET="run.features=[System,Connection_1]" bash train_attributes.sh   # feature 나눠 돌기
# =============================================================
set -euo pipefail
source "$(dirname "$0")/../common.sh"

CONFIG="${CONFIG:-${CONFIG_DIR}/05_attributes.yaml}"
log "=== Stage 4. Attributes ==="
run_with_config "${TRAIN_ATTRIBUTES}" "${CONFIG}"
log "Attributes 완료"
