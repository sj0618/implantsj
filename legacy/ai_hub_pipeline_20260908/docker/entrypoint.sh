#!/usr/bin/env bash
# 컨테이너 시작 때 한 번: 이미지에 구운 가중치(/opt/weights)를 yaml 이 기대하는
# checkpoints/ 경로에 놓는다. checkpoints/ 는 보통 host volume 이다.
#   - SAM1 / SAM2 base 는 읽기 전용 입력 → symlink (2.5GB 를 볼륨에 복사하지 않는다)
#   - 파인튜닝 adapter 는 Stage 0 ⑤ 가 같은 경로에 **덮어쓰는 출력**이기도 하다 →
#     symlink 면 재학습 결과가 컨테이너 레이어(--rm 으로 사라짐)에 써지므로 반드시 복사한다.
# 이미 파일이 있으면 건드리지 않고, 이전 이미지가 남긴 깨진 링크는 갈아끼운다.
set -euo pipefail

ROOT="${AI_HUB_ROOT:-/workspace/ai_hub_pipeline}"
WEIGHTS="${AI_HUB_WEIGHTS:-/opt/weights}"
mkdir -p "$ROOT/checkpoints" "$ROOT/outputs" "$ROOT/data" "$ROOT/seg_data"

place_weight() {
  local how="$1" rel="$2" src="$WEIGHTS/$2" dst="$ROOT/checkpoints/$2"
  if [[ -e "$dst" ]]; then
    return 0                                   # 사용자가 둔 파일 우선
  fi
  if [[ -L "$dst" ]]; then
    rm -f "$dst"                               # 대상이 사라진 옛 symlink
  fi
  if [[ ! -e "$src" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "$dst")"
  if [[ "$how" == "link" ]]; then
    ln -s "$src" "$dst"
    echo "[entrypoint] checkpoints/$rel -> 이미지에 구운 가중치 (link)"
  else
    cp "$src" "$dst.tmp" && mv "$dst.tmp" "$dst"      # 중간에 죽어도 잘린 파일이 남지 않게
    echo "[entrypoint] checkpoints/$rel <- 이미지에 구운 가중치 (copy, 재학습 시 덮어써도 볼륨에 남는다)"
  fi
}
place_weight link sam/sam_vit_h_4b8939.pth
place_weight link sam2/sam2.1_hiera_tiny.pt
place_weight copy best_sam2.1_tiny_adapter_unet.pth

if [[ ! -e "$ROOT/checkpoints/best_sam2.1_tiny_adapter_unet.pth" ]]; then
  echo "[entrypoint] !! 파인튜닝 adapter 없음 (checkpoints/best_sam2.1_tiny_adapter_unet.pth)."
  echo "             Stage 0 ①~⑤ 로 학습하거나 checkpoints/ 에 넣으면 masked-vit / band mask 가 된다."
fi
cd "$ROOT"
exec "$@"
