#!/usr/bin/env bash
# =====================================================================
# 폐쇄망 자가진단. 실제 데이터 없이 파이프라인 전체를 확인한다.
#
#   ./docker/smoke_test.sh
#
# 합성 데이터를 만들어 변환, manifest 생성, 학습, 평가를 1 epoch씩 실행한다.
# 통과하면 문제 범위를 코드나 환경이 아닌 실제 데이터 쪽으로 좁힐 수 있다.
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
            # case id 에 split 과 라벨을 모두 넣는다. 실제 데이터에서 촬영건 하나는
            # 한 시스템만 담고 train/test 가 같은 촬영건을 공유하지 않는다.
            # split_train_valid 가 두 조건을 모두 검사하므로 fixture 도 지켜야 한다.
            h = f"h{(k%2)+1:02d}"; c = f"{h}_{split}_L{li}_{k:04d}"
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
  EPOCHS=1 BATCH_SIZE=8 IMAGE_SIZE=64 NUM_WORKERS=0 DEVICE="${DEVICE:-cpu}" \
    ./run.sh train data_4label resnet18.a1_in1k
}

s_eval() {
  local ck; ck="$(find "$WORK/out" -name best.pt | head -1)"
  [[ -n "$ck" ]] || { echo "checkpoint 없음"; return 1; }
  DATA_ROOT="$WORK/ds/imagefolders" OUTPUT_ROOT="$WORK/out" NUM_WORKERS=0 \
  DEVICE="${DEVICE:-cpu}" \
    ./run.sh eval data_4label "$ck"
}

# 얼굴인식(ArcFace) 경로. 학습 -> 갤러리 등록 -> 인식까지 한 번에 돈다.
# 여기서 통과하면 loader(return_path) / 갤러리 저장 / threshold 판정 배선이 정상이다.
s_arcface() {
  DATA_ROOT="$WORK/ds/imagefolders" OUTPUT_ROOT="$WORK/out" \
  EPOCHS=1 BATCH_SIZE=8 IMAGE_SIZE=64 NUM_WORKERS=0 DEVICE="${DEVICE:-cpu}" \
    ./run.sh arcface data_4label resnet18.a1_in1k
}

# 클래스 수가 42 가 아닌 데이터셋에서도 manifest 빌더가 도는지 (--closed-labels auto).
s_autolabels() { "$PYTHON" tests/test_manifests_auto_labels.py; }

# ArcFace + 13 feature 표 채우기 (메뉴 18). 클래스 이름을 configs/class_features.csv
# 에 실제로 매칭되는 표기로 만들어야 attribute 정답이 붙는다. holdout 으로 unseen 을
# 만들어 head 분기가 실제로 도는지까지 본다.
s_afa_fixture() {
  "$PYTHON" - "$WORK/afa" <<'PY'
import sys
from pathlib import Path
import numpy as np
from PIL import Image
root = Path(sys.argv[1]) / "afa_ds"
rng = np.random.default_rng(0)
labels = ["01_Osstem_GSII", "03_Osstem_SSII", "05_Osstem_USII",
          "09_Dentium_Superline", "11_NeoBiotech_IS I", "20_MegaGen_AnyRidge"]
for split, per in (("train", 8), ("valid", 3), ("test", 3)):
    for li, lab in enumerate(labels):
        d = root / split / lab
        d.mkdir(parents=True, exist_ok=True)
        for k in range(per):
            arr = np.clip(rng.normal(30 + li * 30, 12, (64, 64, 3)), 0, 255).astype("uint8")
            Image.fromarray(arr).save(d / f"{split}_{li}_{k}.png")
print(f"afa fixture: {len(labels)} classes")
PY
}

s_afa() {
  # threshold 를 높게 잡아 head 분기가 반드시 돌게 한다. 낮게 두면 학습이 안 된
  # 임베딩끼리도 유사도가 높아 전부 갤러리로 흘러 head 가 한 번도 검사되지 않는다.
  # 갤러리 분기 쪽 판정은 src/arcface_attributes.py 의 selftest 가 따로 본다.
  DATA_ROOT="$WORK/afa" OUTPUT_ROOT="$WORK/afa_out" \
  EPOCHS=1 BATCH_SIZE=8 IMAGE_SIZE=64 NUM_WORKERS=0 DEVICE="${DEVICE:-cpu}" \
  ARCFACE_HOLDOUT=2 ARCFACE_THRESHOLD=0.95 \
    ./run.sh afa afa_ds resnet18.a1_in1k
}

# YOLO 가중치가 이미지에 실제로 있고 네트워크 없이 로드되는지. 폐쇄망에서
# ultralytics 가 가중치를 받으려 하다 실패하는 사고를 여기서 잡는다.
s_yolo() {
  "$PYTHON" - <<'PY'
from pathlib import Path
import zipfile
root = Path("/workspace/implantsj") if Path("/workspace/implantsj").is_dir() else Path(".")
names = [l.split("#",1)[0].strip()
         for l in (root/"configs"/"yolo_weights.txt").read_text(encoding="utf-8").splitlines()]
names = [n for n in names if n]
assert names, "configs/yolo_weights.txt 가 비어 있다"
for n in names:
    p = root/"weights"/n
    assert p.is_file(), f"이미지에 없다: {p}"
    assert zipfile.is_zipfile(p), f"torch 체크포인트가 아니다: {p}"
    print(f"  {n}  {p.stat().st_size/1048576:.1f} MB")
from ultralytics import YOLO
m = YOLO(str(root/"weights"/names[0]))
print(f"ultralytics 로드 OK: {names[0]}  ({sum(x.numel() for x in m.model.parameters())/1e6:.1f}M params)")
PY
}

echo "작업 폴더: $WORK"
step "1. 환경(패키지/GPU)"        s_env
step "2. 합성 원본 데이터 생성"     s_fixture
step "3. 원본 -> 학습용 변환"       s_prepare
step "4. 학습 1 epoch"             s_train
step "5. 평가"                     s_eval
step "6. 얼굴인식(ArcFace) 전체"   s_arcface
step "7. 동적 클래스 수 manifest"  s_autolabels
step "8. ArcFace 13feature fixture" s_afa_fixture
step "9. ArcFace 13feature 전체"   s_afa
step "10. YOLO 가중치 오프라인"    s_yolo

printf '\n----------------------------------------\n'
printf '통과 %d / 실패 %d\n' "$pass" "$fail"
[[ $fail -eq 0 ]] && echo "전부 통과. 코드와 환경은 정상입니다." \
                  || echo "실패한 단계의 출력을 확인하세요."
exit $(( fail > 0 ))
