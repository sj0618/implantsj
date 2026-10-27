#!/usr/bin/env bash
# =============================================================
# Stage 1. Backbone 학습 — 재사용 가능한 백본 4종 제작
#   설정: configs/01_backbone.yaml
#   백본: vit / band-vit / dinov2 / band-dinov2  (cnn 제외)
#   옵션: finetune_mode = full(전체) | partial(마지막 n block만)
#         protocol      = closed27_open15(27클래스만) | full42(42클래스)
#   산출: checkpoints/backbones/<backbone>_<mode>_<protocol>.pt
#         → base.yaml의 backbones.<name>.checkpoint 에 넣어 다른 스테이지가 사용
# 사용법:
#   bash finetune_backbones.sh                    # closed27_open15 (yaml 기본값, 8조합)
#   PROTOCOL=full42 bash finetune_backbones.sh    # 42클래스 전부로 학습
#   # 백본 하나만 (그리드는 의무가 아니다)
#   SET="grid.backbone=band-vit grid.finetune_mode=partial" bash finetune_backbones.sh
# =============================================================
set -euo pipefail
source "$(dirname "$0")/../common.sh"

CONFIG="${CONFIG:-${CONFIG_DIR}/01_backbone.yaml}"
log "=== Stage 1. Backbone 학습 ==="
run_with_config "${TRAIN_BACKBONE}" "${CONFIG}"
log "Backbone 학습 완료"
