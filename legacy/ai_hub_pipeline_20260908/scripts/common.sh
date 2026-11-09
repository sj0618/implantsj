#!/usr/bin/env bash
# =============================================================
# common.sh — 모든 실험 스크립트가 공유하는 설정
# 실험 설정(그리드/하이퍼파라미터/경로)은 configs/*.yaml 에 있음.
# 셸 스크립트는 해당 yaml을 엔트리포인트에 넘기기만 한다.
#
# 오버라이드 예:
#   PROTOCOL=full42 bash run_xxx.sh      # yaml의 protocol 덮어쓰기
#   CONFIG=/path/to/custom.yaml bash run_xxx.sh
#
# ---- 그리드 대신 하나만 돌리고 싶을 때 ----
# 그리드는 의무가 아니다. yaml의 리스트를 한 개로 줄이면 조합 1개만 돈다:
#   grid: {backbone: [vit], embedding_loss: [supcon]}   또는  backbone: vit
# yaml을 고치지 않고 이번 실행만 좁히려면 SET= 을 쓴다 (yaml 문법, 공백으로 구분):
#   SET="grid.backbone=vit grid.embedding_loss=supcon" bash run_metric_grid.sh
#   SET="grid.backbone=vit train.epochs=1" bash finetune_backbones.sh   # 짧은 점검용
# =============================================================

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export CONFIG_DIR="${CONFIG_DIR:-${PROJECT_ROOT}/configs}"
export SRC_DIR="${SRC_DIR:-${PROJECT_ROOT}/src}"

# PROTOCOL을 설정하면 yaml의 protocol을 덮어씀 (빈 값이면 yaml 값 사용)
export PROTOCOL="${PROTOCOL:-}"
# SET을 설정하면 yaml의 아무 항목이나 덮어씀 (빈 값이면 yaml 값 그대로)
export SET="${SET:-}"

# ---- 파이썬 엔트리포인트 ----
export PY="${PY:-python}"
# Stage 0-a. 데이터 준비 (파이프라인 입구)
export PREPARE_DATA="${PREPARE_DATA:-${SRC_DIR}/run_prepare_data.py}"
# Stage 0. Segmentation (src/segmentation/ 에 구현 완료)
export SEG_SAMPLE="${SEG_SAMPLE:-${SRC_DIR}/segmentation/sample_finetune_data.py}"
export SEG_SAM1="${SEG_SAM1:-${SRC_DIR}/segmentation/generate_sam1_masks.py}"
export SEG_SPLIT="${SEG_SPLIT:-${SRC_DIR}/segmentation/split_seg_dataset.py}"
export SEG_TRAIN="${SEG_TRAIN:-${SRC_DIR}/segmentation/train_sam2_adapter.py}"
export SEG_PREDICT="${SEG_PREDICT:-${SRC_DIR}/segmentation/predict_masks.py}"
export SEG_BAND="${SEG_BAND:-${SRC_DIR}/segmentation/make_band_masks.py}"
# Stage 3. Classifier (src/에 구현 완료: band_vit_2 / band_vit_3(+eval) 기반)
export RUN_BAND_VIT="${RUN_BAND_VIT:-${SRC_DIR}/run_band_vit.py}"
export RUN_MASKED_VIT="${RUN_MASKED_VIT:-${SRC_DIR}/run_masked_vit.py}"
# Stage 2. Metric learning (src/metric/ 구현 완료 — 학습+평가 통합 드라이버)
export TRAIN_METRIC="${TRAIN_METRIC:-${SRC_DIR}/run_metric.py}"
export RUN_CLUSTERING="${RUN_CLUSTERING:-${SRC_DIR}/run_metric.py}"
export METRIC_FEWSHOT="${METRIC_FEWSHOT:-${SRC_DIR}/run_metric_fewshot.py}"
# Stage 1. Backbone (src/backbone/ 구현 완료 — 재사용 백본 4종 제작)
export TRAIN_BACKBONE="${TRAIN_BACKBONE:-${SRC_DIR}/run_backbone.py}"
# Stage 4. Attributes (src/attributes/ 구현 완료)
export TRAIN_ATTRIBUTES="${TRAIN_ATTRIBUTES:-${SRC_DIR}/run_attributes.py}"
# Stage 5 나머지 (코드 도착 후 실제 파일명으로 교체)
export RUN_FUSE="${RUN_FUSE:-${SRC_DIR}/run_fuse.py}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# yaml 설정으로 엔트리포인트 실행.
#   PROTOCOL이 설정돼 있으면 --protocol 로, SET이 설정돼 있으면 --set 으로 덮어씀
# yaml의 상대경로가 항상 프로젝트 루트 기준으로 풀리도록 PROJECT_ROOT에서 실행
run_with_config() {
    local entry="$1" config="$2"
    # SET은 공백으로 나눠 여러 개를 넘긴다 (따옴표 없이 전개하는 것이 의도된 동작)
    local -a overrides=()
    [[ -n "${SET}" ]] && read -r -a overrides <<< "${SET}"
    log "run: ${entry} --config ${config} ${PROTOCOL:+(protocol=${PROTOCOL})} ${SET:+(set: ${SET})}"
    (cd "${PROJECT_ROOT}" && ${PY} "${entry}" --config "${config}" \
        ${PROTOCOL:+--protocol "${PROTOCOL}"} \
        ${overrides:+--set "${overrides[@]}"})
}
