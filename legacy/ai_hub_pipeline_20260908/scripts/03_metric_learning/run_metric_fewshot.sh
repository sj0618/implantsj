#!/usr/bin/env bash
# =============================================================
# Stage 2c. Few-shot target adaptation (별개 실험)
#   설정: configs/03_metric_fewshot.yaml
#   "새 시스템을 1/5/10장만 등록해도 인식되는가" — 재학습 없이 평가만,
#   support 뽑기를 여러 seed로 반복해 평균±표준편차 산출
#   ※ 실행 전 yaml의 models[].checkpoint에 Stage 2 best.pt 경로를 채울 것
# 사용법:
#   bash run_metric_fewshot.sh
# =============================================================
set -euo pipefail
source "$(dirname "$0")/../common.sh"

CONFIG="${CONFIG:-${CONFIG_DIR}/03_metric_fewshot.yaml}"
log "=== Stage 2c. Few-shot target adaptation ==="
run_with_config "${METRIC_FEWSHOT}" "${CONFIG}"
log "few-shot 완료"
