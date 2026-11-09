#!/usr/bin/env bash
# =============================================================
# Stage 2b. Embedding clustering (full42 전용)
#   설정: configs/03_clustering_full42.yaml
#   metric learning 없이 embedding clustering만. 4 backbone x 3 distance
# 사용법:
#   bash run_clustering_full42.sh
#   SET="grid.backbone=dinov2" bash run_clustering_full42.sh   # 백본 하나만
# =============================================================
set -euo pipefail
source "$(dirname "$0")/../common.sh"

CONFIG="${CONFIG:-${CONFIG_DIR}/03_clustering_full42.yaml}"
log "=== Stage 2b. Embedding clustering (full42) ==="
run_with_config "${RUN_CLUSTERING}" "${CONFIG}"
log "Clustering 완료"
