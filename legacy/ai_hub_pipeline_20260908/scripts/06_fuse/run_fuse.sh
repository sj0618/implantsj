#!/usr/bin/env bash
# =============================================================
# Stage 5. Fuse — 3브랜치 융합 → 최종 분류 (System ID or __unknown__)
#   설정: configs/06_fuse.yaml
#   가중치: weights.mode  = learned(valid에서 자동 학습) | manual
#   임계값: threshold.mode = auto(자동 결정) | manual | none
#   평가:   evaluation.top_k = [1] 기본, [1,2]면 top2도 함께
#   open-set gate: metric이 open 판정한 샘플은 metric 단독으로 결정
#
# 사용법:
#   # (a) yaml에 적힌 경로 그대로
#   bash run_fuse.sh
#
#   # (b) 브랜치별 결과를 실행할 때 직접 지정 (yaml보다 우선)
#   bash run_fuse.sh \
#     --metric     outputs/metric/closed27_open15_dinov2_supcon \
#     --classifier outputs/classifier/band_vit/closed27_open15_a8_l0.8_b6 \
#     --attributes outputs/attributes/closed27_open15_cnn_separate
#
#   각 인자는 run 디렉토리 / best.pt / scores CSV 아무거나 받는다.
#   metric은 prototype×distance 조합이 여러 개라, 생략하면 그 run의
#   summary.csv에서 가장 좋은 조합을 자동으로 고른다.
#   직접 고르려면: --metric-variant mean_cosine
#
#   # (c) yaml 항목을 이번 실행만 바꾸기
#   SET="evaluation.top_k=[1,2] threshold.mode=none" bash run_fuse.sh
# =============================================================
set -euo pipefail
source "$(dirname "$0")/../common.sh"

CONFIG="${CONFIG:-${CONFIG_DIR}/06_fuse.yaml}"
log "=== Stage 5. Fuse ==="
# SET은 공백으로 나눠 여러 개를 넘긴다. --set은 뒤따르는 값을 모두 먹으므로 항상 마지막
overrides=()
[[ -n "${SET}" ]] && read -r -a overrides <<< "${SET}"
log "run: ${RUN_FUSE} --config ${CONFIG} $* ${SET:+(set: ${SET})}"
(cd "${PROJECT_ROOT}" && ${PY} "${RUN_FUSE}" --config "${CONFIG}" \
    ${PROTOCOL:+--protocol "${PROTOCOL}"} "$@" \
    ${overrides:+--set "${overrides[@]}"})
log "Fuse 완료"
