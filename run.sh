#!/usr/bin/env bash
# =====================================================================
# implantsj 실험 실행기 (단일 진입점)
# ---------------------------------------------------------------------
# 사용법:
#   ./run.sh              대화형 메뉴 실행 (폐쇄망 권장 — 번호만 누르면 됨)
#   ./run.sh <명령> ...   짧은 명령으로 바로 실행 (아래 usage 참고)
#
# 폐쇄망을 위해 설계했다. 폐쇄망 *안에서는* 복붙이 되지만 망 경계를 넘는
# 복붙은 안 되므로, 필요한 텍스트는 전부 이미지에 구워 안으로 들여보낸다:
#  - 긴 경로/옵션은 exp.conf 에 한 번만 적어두고 재사용한다.
#  - 데이터셋/모델/체크포인트는 자동 검색해 번호로 고른다(경로 오타 방지).
#  - 실행할 실제 명령을 항상 화면에 먼저 출력하고 로그로 남긴다.
#  - 긴 명령 예시는 docs/usage/EXPERIMENT_GUIDE.md 에 있고, 이 문서도
#    이미지에 함께 들어가므로 폐쇄망 안에서 복사해 쓸 수 있다.
# =====================================================================
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# --- 설정 로드 -------------------------------------------------------
# shellcheck source=/dev/null
source "$ROOT/exp.conf"

PYTHON="${PYTHON:-python3}"
STATE_DIR="$OUTPUT_ROOT"
STATE_FILE="$OUTPUT_ROOT/.exp_state"
LOG_DIR="$OUTPUT_ROOT/logs"
MANIFEST_DIR="$OUTPUT_ROOT/manifests"

# 상태(마지막으로 고른 데이터셋/모델/체크포인트)
DATASET=""
MODEL="${MODELS[0]}"
CHECKPOINT=""

C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'; C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_OFF=$'\033[0m'

# --- 유틸 ------------------------------------------------------------
say()  { printf '%s\n' "$*"; }
hr()   { printf '%s\n' "----------------------------------------------------------------"; }
err()  { printf '%s%s%s\n' "$C_ERR" "$*" "$C_OFF" >&2; }
ok()   { printf '%s%s%s\n' "$C_OK" "$*" "$C_OFF"; }

load_state() {
  [[ -f "$STATE_FILE" ]] || return 0
  # shellcheck source=/dev/null
  source "$STATE_FILE" 2>/dev/null || true
}
save_state() {
  mkdir -p "$STATE_DIR"
  {
    printf 'DATASET=%q\n' "$DATASET"
    printf 'MODEL=%q\n'   "$MODEL"
    printf 'CHECKPOINT=%q\n' "$CHECKPOINT"
  } > "$STATE_FILE"
}

# tools/run.py 로 위임 실행. 명령을 화면에 출력하고 로그로 남긴다.
py() {
  mkdir -p "$LOG_DIR"
  local ts; ts="$(date +%Y%m%d_%H%M%S)"
  local log="$LOG_DIR/${ts}_$1.log"
  say ""
  say "${C_BOLD}>>> $PYTHON tools/run.py $*${C_OFF}"
  say "${C_DIM}    (로그: $log)${C_OFF}"
  hr
  "$PYTHON" tools/run.py "$@" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  hr
  if [[ $rc -eq 0 ]]; then ok "완료 (exit 0)"; else err "실패 (exit $rc) — 위 로그 확인"; fi
  return $rc
}

# 명령의 출력을 한 줄씩 배열 PICK_ITEMS 에 담는다.
# mapfile/readarray 는 bash 4+ 전용이라 macOS 기본 bash 3.2 에서 깨진다.
# 컨테이너는 bash 5 지만, 컨테이너 밖에서도 그대로 돌아가도록 이식성 있게 쓴다.
PICK_ITEMS=()
collect() {
  PICK_ITEMS=()
  local line
  while IFS= read -r line; do
    [[ -n "$line" ]] && PICK_ITEMS+=("$line")
  done < <("$@")
}

# 목록에서 번호로 하나 고르기. 고른 값은 REPLY_PICK 에 담긴다.
REPLY_PICK=""
pick_from_list() {
  local title="$1"; shift
  local items=("$@")
  REPLY_PICK=""
  if [[ ${#items[@]} -eq 0 ]]; then err "$title: 항목이 없습니다."; return 1; fi
  say ""; say "${C_BOLD}$title${C_OFF}"
  local i
  for i in "${!items[@]}"; do printf '  %2d) %s\n' "$((i+1))" "${items[$i]}"; done
  local n
  read -r -p "번호 선택 (엔터=취소): " n
  [[ -z "$n" ]] && return 1
  if [[ "$n" =~ ^[0-9]+$ ]] && (( n>=1 && n<=${#items[@]} )); then
    REPLY_PICK="${items[$((n-1))]}"; return 0
  fi
  err "잘못된 번호"; return 1
}

# $DATA_ROOT 아래 데이터셋 폴더 자동 검색.
# train/valid/test 하위 폴더를 가진 곳을 "학습 가능한 데이터셋"으로 본다.
# 변환 결과는 <name>/imagefolders/data_* 처럼 한 단계 안쪽에 생기므로
# depth 3 까지 훑는다. 반환 값은 $DATA_ROOT 기준 상대 경로다.
discover_datasets() {
  [[ -d "$DATA_ROOT" ]] || return 0
  {
    local d
    while IFS= read -r d; do
      if [[ -d "$d/train" || -d "$d/test" || -d "$d/valid" || -d "$d/val" ]]; then
        printf '%s\n' "${d#"$DATA_ROOT"/}"
      fi
    done < <(find "$DATA_ROOT" -mindepth 1 -maxdepth 3 -type d -not -path '*/.*' 2>/dev/null)
    # data.yaml 을 가진 YOLO 데이터셋도 후보로 넣는다.
    while IFS= read -r y; do
      d="$(dirname "$y")"
      printf '%s\n' "${d#"$DATA_ROOT"/}"
    done < <(find "$DATA_ROOT" -mindepth 1 -maxdepth 3 -name data.yaml 2>/dev/null)
  } | sort -u
}

# $OUTPUT_ROOT 아래 체크포인트(.pt) 자동 검색 (최근 것 먼저)
discover_checkpoints() {
  [[ -d "$OUTPUT_ROOT" ]] || return 0
  find "$OUTPUT_ROOT" -type f \( -name 'best.pt' -o -name 'last.pt' -o -name '*.pt' \) 2>/dev/null \
    | grep -v '/manifests/' | sort -u
}

sanitize() { printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_'; }

# 현재 DATASET 에 대한 학습용 manifest 를 보장하고 경로를 echo 한다.
ensure_manifest() {
  local ds="$1"
  local mani="$MANIFEST_DIR/$(sanitize "$ds").csv"
  mkdir -p "$MANIFEST_DIR"
  if [[ ! -f "$mani" ]]; then
    py manifest --data-root "$DATA_ROOT/$ds" --out "$mani" >/dev/null || return 1
  fi
  # split 컬럼(마지막 열)에 서로 다른 값이 2개 이상(train/test 등) 있으면 그대로 사용.
  local nsplits
  nsplits="$(awk -F, 'NR>1{print $NF}' "$mani" | sort -u | grep -vc '^$')"
  if [[ "${nsplits:-0}" -ge 2 ]]; then
    printf '%s\n' "$mani"; return 0
  fi
  # 없으면 8:1:1 로 stratified split 생성.
  local split="$MANIFEST_DIR/$(sanitize "$ds").split.csv"
  py split --manifest "$mani" --out "$split" --train 8 --valid 1 --test 1 --seed "$SEED" >/dev/null || return 1
  printf '%s\n' "$split"
}

# DEVICE 가 cuda 가 아니면 분류/metric 파이프라인에 --allow-cpu 를 붙인다.
allow_cpu_flag() { [[ "$DEVICE" != "cuda" ]] && printf -- '--allow-cpu'; }

need_dataset() {
  if [[ -z "$DATASET" ]]; then
    err "먼저 데이터셋을 선택하세요 (메뉴 2번)."; return 1
  fi
  if [[ ! -d "$DATA_ROOT/$DATASET" ]]; then
    err "데이터셋 폴더가 없습니다: $DATA_ROOT/$DATASET"; return 1
  fi
}

# --- 액션 -----------------------------------------------------------
act_check() { py check; }

act_pick_dataset() {
  collect discover_datasets
  if pick_from_list "데이터셋 선택  ($DATA_ROOT)" "${PICK_ITEMS[@]}"; then
    DATASET="$REPLY_PICK"; save_state; ok "데이터셋 = $DATASET"
  fi
}

act_pick_model() {
  if pick_from_list "모델 선택" "${MODELS[@]}"; then
    MODEL="$REPLY_PICK"; save_state; ok "모델 = $MODEL"
  fi
}

act_prepare() {
  need_dataset || return 1
  local mani; mani="$(ensure_manifest "$DATASET")" || return 1
  ok "manifest 준비됨: $mani"
}

# 폐쇄망 원본 구조(TL1/TS1/VL1/VS1)를 학습용 ImageFolder 로 변환한다.
# 원본은 병원/촬영건 폴더 + JSON 라벨 분리 구조라 그대로는 학습에 못 쓴다.
act_build_dataset() {
  say ""
  say "${C_BOLD}원본 데이터 -> 학습용 데이터셋 변환${C_OFF}"
  say "${C_DIM}원본 구조: TL1(train JSON) TS1(train 이미지) VL1(test JSON) VS1(test 이미지)${C_OFF}"
  say "${C_DIM}엔터만 치면 [ ] 안의 기본값을 씁니다.${C_OFF}"

  local raw
  read -r -p "원본 루트 (TL1/TS1/VL1/VS1 의 상위 폴더) [$DATA_ROOT/raw]: " raw
  raw="${raw:-$DATA_ROOT/raw}"
  local missing=()
  local d
  for d in TL1 TS1 VL1 VS1; do
    [[ -d "$raw/$d" ]] || missing+=("$d")
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    err "$raw 아래에 다음 폴더가 없습니다: ${missing[*]}"
    say "${C_DIM}실제 폴더 이름이 다르면 아래 명령으로 직접 지정하세요:${C_OFF}"
    say "  $PYTHON tools/run.py prepare --help"
    return 1
  fi

  local name
  read -r -p "만들 데이터셋 이름 [closed_network]: " name
  name="$(sanitize "${name:-closed_network}")"

  local unseen
  read -r -p "unseen 으로 뺄 희귀 라벨 개수 (0=전부 학습) [15]: " unseen
  unseen="${unseen:-15}"

  local out="$DATA_ROOT/$name"
  if [[ -e "$out" ]]; then
    err "이미 존재합니다: $out  (다른 이름을 쓰거나 먼저 지우세요)"
    return 1
  fi

  py prepare \
    --train-json-root "$raw/TL1" --train-image-root "$raw/TS1" \
    --test-json-root  "$raw/VL1" --test-image-root  "$raw/VS1" \
    --output-root "$out" \
    --unseen-label-count "$unseen" \
    --seed "$SEED" \
    --allow-missing-json \
    --materialize-imagefolders --link-mode hardlink || return 1

  say ""
  ok "변환 완료. 만들어진 데이터셋:"
  find "$out/imagefolders" -mindepth 1 -maxdepth 1 -type d 2>/dev/null \
    | while read -r p; do printf '   %s\n' "${p#$DATA_ROOT/}"; done
  say "${C_DIM}메뉴 2번에서 위 폴더를 데이터셋으로 선택해 학습하면 됩니다.${C_OFF}"
}

act_train_supervised() {
  need_dataset || return 1
  local mani; mani="$(ensure_manifest "$DATASET")" || return 1
  local name; name="$(sanitize "${DATASET}__${MODEL}")_sup"
  py train \
    --manifest "$mani" --model-name "$MODEL" \
    --output-root "$OUTPUT_ROOT" --experiment-name "$name" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --image-size "$IMAGE_SIZE" \
    --lr "$LR" --seed "$SEED" --num-workers "$NUM_WORKERS" $(allow_cpu_flag)
}

act_train_metric() {
  need_dataset || return 1
  local mani; mani="$(ensure_manifest "$DATASET")" || return 1
  local name; name="$(sanitize "${DATASET}__${MODEL}")_supcon"
  py metric \
    --manifest "$mani" --model-name "$MODEL" \
    --output-root "$OUTPUT_ROOT" --experiment-name "$name" \
    --loss-name supcon --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --image-size "$IMAGE_SIZE" --lr "$LR" --seed "$SEED" \
    --num-workers "$NUM_WORKERS" $(allow_cpu_flag)
}

# VLM feature 를 ViT 패치 토큰으로 붙여 학습한다.
# 대조군(none/zero/shuffle)을 같은 명령으로 고를 수 있어야 실험이 성립한다.
act_train_patch_feature() {
  need_dataset || return 1
  if [[ ! -f "$VLM_FEATURES" ]]; then
    err "VLM feature 파일이 없습니다: $VLM_FEATURES"
    say "${C_DIM}exp.conf 의 VLM_FEATURES 를 고치거나 VLM_FEATURES=... ./run.sh 로 실행하세요.${C_OFF}"
    return 1
  fi
  pick_from_list "feature 주입 방식" \
    "patch    - feature 를 패치 토큰으로 주입 (본 실험)" \
    "none     - 이미지만 (baseline)" \
    "zero     - 토큰만 붙이고 입력 0 (파라미터 효과 대조)" \
    "shuffle  - feature 를 남의 것과 섞음 (진짜 쓰는지 검증)" || return 1
  local fmode="${REPLY_PICK%% *}"

  local mani; mani="$(ensure_manifest "$DATASET")" || return 1
  local name; name="$(sanitize "${DATASET}__${MODEL}")_patch_${fmode}"
  py patchtrain \
    --manifest "$mani" --vlm-features "$VLM_FEATURES" \
    --model-name "$MODEL" --feature-mode "$fmode" \
    --output-root "$OUTPUT_ROOT" --experiment-name "$name" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --image-size "$IMAGE_SIZE" \
    --lr "$LR" --seed "$SEED" --num-workers "$NUM_WORKERS" $(allow_cpu_flag)
}

# 학습된 패치-feature 모델을 평가한다. 학습 때와 다른 주입 방식을 골라
# 같은 가중치로 ablation 을 볼 수 있다.
act_eval_patch_feature() {
  need_dataset || return 1
  _pick_checkpoint || return 1
  local mani; mani="$(ensure_manifest "$DATASET")" || return 1
  pick_from_list "평가할 feature 주입 방식 (학습 때와 달라도 됨)" \
    "(학습 때 설정 그대로)" "patch" "none" "zero" "shuffle" || return 1
  local extra=()
  [[ "$REPLY_PICK" != "(학습 때 설정 그대로)" ]] && extra=(--feature-mode "$REPLY_PICK")
  py patcheval \
    --checkpoint "$CHECKPOINT" --manifest "$mani" \
    --vlm-features "$VLM_FEATURES" --split test \
    "${extra[@]+"${extra[@]}"}" $(allow_cpu_flag)
}

_pick_checkpoint() {
  collect discover_checkpoints
  pick_from_list "체크포인트 선택  ($OUTPUT_ROOT)" "${PICK_ITEMS[@]}" || return 1
  CHECKPOINT="$REPLY_PICK"; save_state
}

act_eval_classifier() {
  need_dataset || return 1
  _pick_checkpoint || return 1
  local mani; mani="$(ensure_manifest "$DATASET")" || return 1
  local name; name="$(sanitize "${DATASET}")_eval"
  py eval \
    --checkpoint "$CHECKPOINT" --manifest "$mani" --split test \
    --output-root "$OUTPUT_ROOT" --experiment-name "$name" $(allow_cpu_flag)
}

act_eval_mahalanobis() {
  need_dataset || return 1
  _pick_checkpoint || return 1
  local mani; mani="$(ensure_manifest "$DATASET")" || return 1
  local out; out="$OUTPUT_ROOT/mahalanobis/$(sanitize "$DATASET")_$(date +%Y%m%d_%H%M%S)"
  py maha \
    --checkpoint "$CHECKPOINT" --manifest "$mani" \
    --output-dir "$out" --device "$DEVICE"
}

act_yolo_train() {
  collect discover_datasets
  pick_from_list "YOLO 데이터셋 선택 (data.yaml 포함 폴더)" "${PICK_ITEMS[@]}" || return 1
  local yaml="$DATA_ROOT/$REPLY_PICK/data.yaml"
  if [[ ! -f "$yaml" ]]; then err "data.yaml 이 없습니다: $yaml"; return 1; fi
  py yolotrain \
    --data "$yaml" --epochs "$EPOCHS" --batch "$BATCH_SIZE" \
    --imgsz 640 --device "$DEVICE" \
    --project "$OUTPUT_ROOT/yolo" --name "$(sanitize "$REPLY_PICK")"
}

act_yolo_crop() {
  collect discover_datasets
  pick_from_list "crop 대상 데이터셋(이미지 루트) 선택" "${PICK_ITEMS[@]}" || return 1
  local src="$DATA_ROOT/$REPLY_PICK"
  _pick_checkpoint || return 1   # YOLO weights(.pt) 를 고른다
  local out="$DATA_ROOT/$(sanitize "$REPLY_PICK")_cropped"
  py yolocrop \
    --source "$src" --weights "$CHECKPOINT" --output-root "$out" \
    --device "$DEVICE"
}

act_settings() {
  say ""; say "${C_BOLD}현재 설정${C_OFF}"
  printf '  DATA_ROOT   = %s\n' "$DATA_ROOT"
  printf '  OUTPUT_ROOT = %s\n' "$OUTPUT_ROOT"
  printf '  DEVICE=%s  EPOCHS=%s  BATCH=%s  IMG=%s  LR=%s  SEED=%s  WORKERS=%s\n' \
    "$DEVICE" "$EPOCHS" "$BATCH_SIZE" "$IMAGE_SIZE" "$LR" "$SEED" "$NUM_WORKERS"
  say "${C_DIM}값을 바꾸려면 exp.conf 를 편집하거나 환경변수로 실행하세요. 예) EPOCHS=5 ./run.sh${C_OFF}"
  local k v
  read -r -p "임시로 바꿀 항목 (device/epochs/batch/img/lr/seed/workers, 엔터=취소): " k
  [[ -z "$k" ]] && return 0
  read -r -p "새 값: " v
  [[ -z "$v" ]] && return 0
  case "$k" in
    device)  DEVICE="$v" ;;
    epochs)  EPOCHS="$v" ;;
    batch)   BATCH_SIZE="$v" ;;
    img)     IMAGE_SIZE="$v" ;;
    lr)      LR="$v" ;;
    seed)    SEED="$v" ;;
    workers) NUM_WORKERS="$v" ;;
    *) err "알 수 없는 항목: $k"; return 1 ;;
  esac
  ok "$k = $v (이번 실행에만 적용)"
}

# --- 메뉴 -----------------------------------------------------------
menu() {
  load_state
  while true; do
    say ""
    hr
    say "${C_BOLD} implantsj 실험 실행기${C_OFF}"
    printf ' %s데이터=%s%s  %s모델=%s%s  %s장치=%s%s\n' \
      "$C_DIM" "${DATASET:-(미선택)}" "$C_OFF" \
      "$C_DIM" "$MODEL" "$C_OFF" \
      "$C_DIM" "$DEVICE" "$C_OFF"
    hr
    say "  1) 환경 체크 (GPU/패키지 확인)"
    say "  ${C_DIM}--- 데이터 ---${C_OFF}"
    say "  2) 원본 -> 학습용 데이터셋 변환 ${C_DIM}(폐쇄망에서 맨 처음 1회)${C_OFF}"
    say "  3) 데이터셋 선택"
    say "  4) 모델 선택"
    say "  5) manifest + split 만들기"
    say "  ${C_DIM}--- 학습/평가 ---${C_OFF}"
    say "  6) 학습 · supervised 분류"
    say "  7) 학습 · SupCon metric"
    say "  8) 학습 · ViT + VLM feature 패치 주입"
    say "  9) 평가 · 분류 (checkpoint)"
    say " 10) 평가 · Mahalanobis (unseen)"
    say " 11) 평가 · ViT + feature 패치 (ablation)"
    say "  ${C_DIM}--- YOLO ---${C_OFF}"
    say " 12) YOLO · 검출 학습"
    say " 13) YOLO · crop 전처리"
    say "  ${C_DIM}---${C_OFF}"
    say " 14) 설정 보기/임시 변경"
    say "  0) 종료"
    hr
    local c
    read -r -p "선택: " c
    case "$c" in
      1) act_check ;;
      2) act_build_dataset ;;
      3) act_pick_dataset ;;
      4) act_pick_model ;;
      5) act_prepare ;;
      6) act_train_supervised ;;
      7) act_train_metric ;;
      8) act_train_patch_feature ;;
      9) act_eval_classifier ;;
      10) act_eval_mahalanobis ;;
      11) act_eval_patch_feature ;;
      12) act_yolo_train ;;
      13) act_yolo_crop ;;
      14) act_settings ;;
      0|q|quit|exit) ok "종료"; break ;;
      "") ;;
      *) err "메뉴에 없는 선택: $c" ;;
    esac
  done
}

usage() {
  cat <<'EOF'
implantsj 실험 실행기
  ./run.sh                 대화형 메뉴 (권장)
  ./run.sh check           환경 체크
  ./run.sh prep   <ds>     데이터 준비(manifest+split)
  ./run.sh train  <ds> <model>   supervised 분류 학습
  ./run.sh metric <ds> <model>   SupCon metric 학습
  ./run.sh eval   <ds> <ckpt>    분류 평가
  ./run.sh maha   <ds> <ckpt>    Mahalanobis 평가
  ./run.sh patch  <ds> <model> [patch|none|zero|shuffle]
                                 ViT + VLM feature 패치 주입 학습
  ds  = $DATA_ROOT 아래 데이터셋 폴더 이름
  model = configs/backbones.txt 의 timm 모델 이름
EOF
}

# --- 진입점 ---------------------------------------------------------
main() {
  load_state
  local sub="${1:-menu}"
  case "$sub" in
    menu|"") menu ;;
    check)   act_check ;;
    prep)    DATASET="${2:?ds required}"; act_prepare ;;
    train)   DATASET="${2:?ds required}"; MODEL="${3:-$MODEL}"; act_train_supervised ;;
    metric)  DATASET="${2:?ds required}"; MODEL="${3:-$MODEL}"; act_train_metric ;;
    eval)    DATASET="${2:?ds required}"; CHECKPOINT="${3:?ckpt required}";
             mani="$(ensure_manifest "$DATASET")" || exit 1
             py eval --checkpoint "$CHECKPOINT" --manifest "$mani" --split test \
               --output-root "$OUTPUT_ROOT" --experiment-name "$(sanitize "$DATASET")_eval" $(allow_cpu_flag) ;;
    maha)    DATASET="${2:?ds required}"; CHECKPOINT="${3:?ckpt required}";
             mani="$(ensure_manifest "$DATASET")" || exit 1
             py maha --checkpoint "$CHECKPOINT" --manifest "$mani" \
               --output-dir "$OUTPUT_ROOT/mahalanobis/$(sanitize "$DATASET")_$(date +%Y%m%d_%H%M%S)" \
               --device "$DEVICE" ;;
    patch)   DATASET="${2:?ds required}"; MODEL="${3:-$MODEL}"; FMODE="${4:-patch}"
             mani="$(ensure_manifest "$DATASET")" || exit 1
             py patchtrain --manifest "$mani" --vlm-features "$VLM_FEATURES" \
               --model-name "$MODEL" --feature-mode "$FMODE" \
               --output-root "$OUTPUT_ROOT" \
               --experiment-name "$(sanitize "${DATASET}__${MODEL}")_patch_${FMODE}" \
               --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --image-size "$IMAGE_SIZE" \
               --lr "$LR" --seed "$SEED" --num-workers "$NUM_WORKERS" $(allow_cpu_flag) ;;
    -h|--help|help) usage ;;
    *) err "알 수 없는 명령: $sub"; usage; exit 2 ;;
  esac
}

main "$@"
