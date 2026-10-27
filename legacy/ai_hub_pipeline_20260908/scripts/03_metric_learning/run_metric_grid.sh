#!/usr/bin/env bash
# =============================================================
# Stage 2. Metric learning grid (closed27_open15)
#   설정: configs/03_metric_grid.yaml
#   4 backbone x 4 loss = 16 학습, 각각 x 3 distance = 48 평가
#   (grid 순회 + 거리별 재평가는 엔트리포인트가 yaml을 읽어 수행)
# 사용법:
#   bash run_metric_grid.sh                          # yaml에 적힌 전 조합
#   # 조합 하나만 (그리드는 의무가 아니다 — yaml을 고치거나 SET=으로 좁힌다)
#   SET="grid.backbone=dinov2 grid.embedding_loss=supcon" bash run_metric_grid.sh
# 참고: full42 프로토콜은 run_clustering_full42.sh 사용
# =============================================================
set -euo pipefail
source "$(dirname "$0")/../common.sh"

CONFIG="${CONFIG:-${CONFIG_DIR}/03_metric_grid.yaml}"
log "=== Stage 2. Metric learning grid ==="
run_with_config "${TRAIN_METRIC}" "${CONFIG}"
log "Metric learning grid 완료"
