#!/usr/bin/env bash
# ⑤ SAM2(hiera-tiny) encoder frozen + Adapter + U-Net decoder 파인튜닝
# ※ 사전 준비: sam2 repo 설치 (git clone https://github.com/facebookresearch/sam2 && pip install -e .)
set -euo pipefail
source "$(dirname "$0")/../common.sh"
CONFIG="${CONFIG:-${CONFIG_DIR}/02_segmentation.yaml}"
log "=== Seg ⑤ SAM2+Adapter+U-Net 파인튜닝 ==="
run_with_config "${SEG_TRAIN}" "${CONFIG}"
