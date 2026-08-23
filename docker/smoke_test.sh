#!/usr/bin/env bash
# =====================================================================
# 폐쇄망 자가진단 — 실제 데이터 없이도 파이프라인 전체가 도는지 확인한다.
#
#   ./docker/smoke_test.sh
#
# 합성 데이터를 만들어서 변환 -> manifest -> 학습 -> 평가까지 1 epoch 로
# 돌려본다. 여기서 통과하면 "코드/환경 문제"가 아니라 "데이터 문제"로
# 범위를 좁힐 수 있다.
# =====================================================================
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
WORK="${WORK:-${TMPDIR:-/tmp}/implantsj_smoke_$$}"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

pass=0; fail=0
step() {
  local name="$1"; shift
  printf '\n=== %s ===\n' "$name"
  if "$@"; then printf 'PASS  %s\n' "$name"; pass=$((pass+1));
  else printf 'FAIL  %s\n' "$name"; fail=$((fail+1)); fi
}

s_env() { "$PYTHON" tools/run.py check; }

s_fixture() {
  "$PYTHON" - "$WORK/raw" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
root = Path(sys.argv[1]); rng = np.random.default_rng(0)
labels = [f"Model{chr(65+i)}" for i in range(4)]
def make(img, js, split, per):
    for li, lab in enumerate(labels):
        for k in range(per):
            h = f"h{(k%2)+1:02d}"; c = f"{h}_{k:04d}"
            (root/img/h/c).mkdir(parents=True, exist_ok=True)
            (root/js/h/c).mkdir(parents=True, exist_ok=True)
            stem = f"{split}_{lab}_{k}"
            arr = np.clip(rng.normal(40+li*50, 10, (64,64,3)), 0, 255).astype('uint8')
            Image.fromarray(arr).save(root/img/h/c/f"{stem}.png")
            (root/js/h/c/f"{stem}.json").write_text(
                json.dumps({"meta": {"모델명": lab}}, ensure_ascii=False), encoding="utf-8")
make("TS1","TL1","train",12); make("VS1","VL1","test",4)
print("fixture ok")
PY
}

s_prepare() {
  "$PYTHON" tools/run.py prepare \
    --train-json-root "$WORK/raw/TL1" --train-image-root "$WORK/raw/TS1" \
    --test-json-root  "$WORK/raw/VL1" --test-image-root  "$WORK/raw/VS1" \
    --output-root "$WORK/ds" --unseen-label-count 0 \
    --allow-missing-json --materialize-imagefolders --link-mode hardlink
}

s_train() {
  DATA_ROOT="$WORK/ds/imagefolders" OUTPUT_ROOT="$WORK/out" \
  EPOCHS=1 BATCH_SIZE=8 IMAGE_SIZE=64 NUM_WORKERS=0 \
    ./run.sh train data_4label resnet18.a1_in1k
}

s_eval() {
  local ck; ck="$(find "$WORK/out" -name best.pt | head -1)"
  [[ -n "$ck" ]] || { echo "checkpoint 없음"; return 1; }
  DATA_ROOT="$WORK/ds/imagefolders" OUTPUT_ROOT="$WORK/out" NUM_WORKERS=0 \
    ./run.sh eval data_4label "$ck"
}

echo "작업 폴더: $WORK"
step "1. 환경(패키지/GPU)"        s_env
step "2. 합성 원본 데이터 생성"     s_fixture
step "3. 원본 -> 학습용 변환"       s_prepare
step "4. 학습 1 epoch"             s_train
step "5. 평가"                     s_eval

printf '\n----------------------------------------\n'
printf '통과 %d / 실패 %d\n' "$pass" "$fail"
[[ $fail -eq 0 ]] && echo "전부 통과 — 코드와 환경은 정상입니다." \
                  || echo "실패한 단계의 출력을 확인하세요."
exit $(( fail > 0 ))
