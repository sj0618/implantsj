#!/usr/bin/env bash
# =====================================================================
# 통합 이미지 자가진단. 실제 데이터 없이 돈다.
#
#   docker run --rm implantid:latest smoke
#
# 세 프로젝트 모두 합성 데이터로 **실제 학습 1 epoch + 평가**까지 돌린다.
# import 가 되는 것과 학습이 도는 것은 다른 문제이고, 폐쇄망 안에서 후자가
# 깨지면 고칠 수 없기 때문이다.
#
# 정확도 숫자는 의미 없다. 무작위 잡음 이미지에 1 epoch 이라 당연히 낮다.
# 여기서 보는 것은 배선이 끝까지 도는지다.
# =====================================================================
set -uo pipefail
ROOT=/workspace
WORK="${WORK:-${TMPDIR:-/tmp}/implantid_smoke_$$}"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

pass=0; fail=0
step() {
  local name="$1"; shift
  printf '\n=== %s ===\n' "$name"
  if "$@"; then printf 'PASS  %s\n' "$name"; pass=$((pass+1));
  else printf 'FAIL  %s\n' "$name"; fail=$((fail+1)); fi
}

# ys 는 ms_deform_attn CUDA 확장이 libc10.so 를 찾아야 한다.
ys_ld() {
  export LD_LIBRARY_PATH="$(/opt/venv/ys/bin/python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH:-}"
}

# ---------------------------------------------------------------- 1. 격리
s_isolation() {
  /opt/venv/sj/bin/python -c "import torch,numpy;print(f'  sj  torch {torch.__version__}  numpy {numpy.__version__}')"
  /opt/venv/ys/bin/python -c "import torch,numpy;print(f'  ys  torch {torch.__version__}  numpy {numpy.__version__}')"
  /opt/venv/ny/bin/python -c "import torch,numpy;print(f'  ny  torch {torch.__version__}  numpy {numpy.__version__}')"
  # 핀이 실제로 갈라져 있는지. 하나로 합쳐졌으면 어느 팀의 제약이 깨진 것이다.
  /opt/venv/ys/bin/python -c "import numpy,sys;sys.exit(0 if numpy.__version__.startswith('1.26') else 1)" \
    || { echo "  !! ys 의 numpy 가 1.26 이 아니다 (pycocotools ABI 제약)"; return 1; }
  /opt/venv/ny/bin/python -c "import numpy,sys;sys.exit(0 if numpy.__version__.startswith('2.') else 1)" \
    || { echo "  !! ny 의 numpy 가 2.x 가 아니다"; return 1; }
}

# ---------------------------------------------------------------- 2. sj
s_sj_full() {
  cd "$ROOT/implantsj" && PYTHON=/opt/venv/sj/bin/python ./docker/smoke_test.sh
}

# ---------------------------------------------------------------- 3~5. ys
s_ys_gdino() {
  ys_ld
  cd "$ROOT/implantys"
  /opt/venv/ys/bin/python -c "
import torch, numpy, cv2, transformers, timm
from groundingdino import _C
from groundingdino.util.inference import load_model
from pathlib import Path
w = Path('weights/groundingdino_swint_ogc.pth')
assert w.is_file() and w.stat().st_size > 500*1048576, w
print(f'  gdino CUDA 확장 로드 OK | {w.name} {w.stat().st_size/1048576:.0f} MB')
"
}

s_ys_offline_bert() {
  # 폐쇄망에서 gdino 가 텍스트 인코더를 받으러 나가면 학습이 멈춘다.
  # HF_HUB_OFFLINE=1 상태에서 캐시만으로 로드되는지 본다.
  /opt/venv/ys/bin/python -c "
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained('bert-base-uncased')
AutoModel.from_pretrained('bert-base-uncased')
print('  bert-base-uncased 오프라인 로드 OK')
"
}

s_ys_train() {
  # metric learning 스크립트는 split.json / ckpt / logs 를 자기 디렉터리에 쓴다.
  # /workspace 를 오염시키지 않도록 사본에서 돌린다. mount 로 저장소를 물린
  # 경우에도 원본이 그대로 남는다.
  local ml="$WORK/ys_ml"
  cp -r "$ROOT/implantys/metric learning" "$ml" || return 1
  rm -rf "$ml/embeddings" "$ml/ckpt" "$ml/logs"

  /opt/venv/ys/bin/python - "$WORK/ys_data" <<'PY'
import sys
from pathlib import Path
import numpy as np
from PIL import Image
root = Path(sys.argv[1]); rng = np.random.default_rng(1)
# make_split 은 <root>/<클래스>/*.jpg 구조를 스캔한다.
labels = ["adin", "bego", "bicon", "dentium", "mis", "nobel"]
for li, lab in enumerate(labels):
    d = root/lab; d.mkdir(parents=True, exist_ok=True)
    for k in range(10):
        arr = np.clip(rng.normal(35+li*35, 14, (120, 90, 3)), 0, 255).astype("uint8")
        Image.fromarray(arr).save(d/f"{lab}_{k:03d}.jpg")
print(f"  fixture: {len(labels)} classes x 10")
PY

  cd "$ml" || return 1
  # min-open 6 이면 10장짜리 클래스가 전부 open 으로 분류된다.
  /opt/venv/ys/bin/python make_split.py --data-dir "$WORK/ys_data" --min-open 6 >/dev/null 2>&1 \
    || { echo "  !! make_split 실패"; return 1; }
  # M2 = frozen DINOv2 + linear head. 조건 중 가장 가볍고, 학습과 test 평가를
  # train.py 가 함께 수행한다.
  /opt/venv/ys/bin/python train.py --cond M2 --epochs 1 --batch-size 4 \
      --num-workers 0 --device cpu 2>&1 | grep -E "^\[M2\]|train_ce|best:|softmax 참조값" \
    || return 1
  [[ -f "$ml/ckpt/M2.pt" ]] || { echo "  !! 체크포인트가 안 만들어졌다"; return 1; }
  # 임베딩 추출까지 돌려야 metric learning 파이프라인이 끝까지 도는 것이다.
  /opt/venv/ys/bin/python extract.py --cond M2 --ckpt ckpt/M2.pt \
      --batch-size 8 --num-workers 0 --device cpu >/dev/null 2>&1 \
    || { echo "  !! extract 실패"; return 1; }
  local n; n=$(ls "$ml/embeddings/M2"/*.npz 2>/dev/null | wc -l | tr -d ' ')
  [[ "$n" -ge 1 ]] || { echo "  !! 임베딩이 안 나왔다"; return 1; }
  echo "  임베딩 $n개 추출 OK"
}

# ---------------------------------------------------------------- 6~7. ny
s_ny_sam2() {
  cd "$ROOT/implant_ny"
  /opt/venv/ny/bin/python -c "
import torch, numpy, cv2, sam2, segment_anything
from pathlib import Path
import zipfile
p = Path('checkpoints/sam2.1_hiera_tiny.pt')
assert p.is_file(), p
assert zipfile.is_zipfile(p), 'torch 체크포인트가 아니다'
assert p.stat().st_size > 100*1048576, f'받다 만 파일: {p.stat().st_size/1048576:.1f} MB'
print(f'  sam2 / segment_anything import OK | {p.name} {p.stat().st_size/1048576:.0f} MB')
"
}

s_ny_train() {
  /opt/venv/ny/bin/python - "$WORK/ny" <<'PY'
import csv, sys
from pathlib import Path
import numpy as np
from PIL import Image
W = Path(sys.argv[1]); rng = np.random.default_rng(0)
# band_vit 은 path,label 두 컬럼의 manifest CSV 를 읽는다.
# mask_path 는 --band-loss-weight 가 0 보다 클 때만 쓰이므로 여기선 필요 없다.
labels = [f"L{i}" for i in range(4)]
for split, per in (("train", 8), ("valid", 3), ("test", 3)):
    rows = []
    for li, lab in enumerate(labels):
        d = W/"img"/split/lab; d.mkdir(parents=True, exist_ok=True)
        for k in range(per):
            arr = np.clip(rng.normal(40+li*45, 12, (96, 96, 3)), 0, 255).astype("uint8")
            p = d/f"{split}_{li}_{k}.png"; Image.fromarray(arr).save(p)
            rows.append((str(p), lab))
    with open(W/f"{split}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["path", "label"]); w.writerows(rows)
print(f"  fixture: {len(labels)} classes")
PY

  cd "$ROOT/implant_ny" || return 1
  # --pretrained(기본) 로 둔다. 이미지에 구워둔 timm 가중치를 오프라인으로 읽는지
  # 함께 확인하기 위해서다. 여기서 다운로드를 시도하면 폐쇄망에서 깨진다는 뜻이다.
  /opt/venv/ny/bin/python -m band_vit_2.training \
      --model vit_base_patch16_224 \
      --train-data "$WORK/ny/train.csv" \
      --valid-data "$WORK/ny/valid.csv" \
      --test-data  "$WORK/ny/test.csv" \
      --output-dir "$WORK/ny/out" \
      --epochs 1 --batch-size 4 --num-workers 0 --device cpu --input-size 224 \
      2>&1 | grep -E "epoch=|test_accuracy|accuracy=" || return 1
  # 산출물이 실제로 남았는지. 지표 파일이 없으면 평가가 안 돈 것이다.
  local m; m=$(find "$WORK/ny/out" -name "metrics.json" | wc -l | tr -d ' ')
  [[ "$m" -ge 1 ]] || { echo "  !! metrics.json 이 안 나왔다"; return 1; }
  echo "  metrics.json $m개 생성 OK"
}

step "1. 환경 격리 (venv 3개 핀)"            s_isolation
step "2. implantsj 전체 파이프라인 10단계"    s_sj_full
step "3. implantys gdino CUDA 확장"          s_ys_gdino
step "4. implantys BERT 오프라인 로드"        s_ys_offline_bert
step "5. implantys metric learning 학습+평가" s_ys_train
step "6. implant_ny SAM2 체크포인트"          s_ny_sam2
step "7. implant_ny band_vit 학습+평가"       s_ny_train

printf '\n----------------------------------------\n'
printf '통과 %d / 실패 %d\n' "$pass" "$fail"
if [[ $fail -eq 0 ]]; then
  echo "전부 통과."
  echo
  echo "무엇을 본 것인지: 2·5·7 은 합성 데이터로 실제 학습 1 epoch 과 평가를 돌렸다."
  echo "정확도 숫자는 무작위 잡음 이미지 기준이라 의미가 없다 — 배선이 끝까지"
  echo "도는지만 본 것이다. 3·4·6 은 확장 로드와 가중치 존재 확인이다."
  echo
  echo "검사하지 않은 것: implantys 의 gdino 파인튜닝 학습(COCO 형식 검출 데이터가"
  echo "필요하다)과 implant_ny 의 band_vit_3(파인튜닝된 SAM2 adapter 체크포인트가"
  echo "저장소에 없다). 이 둘은 실제 데이터로 처음 돌릴 때 따로 확인해야 한다."
else
  echo "실패한 단계의 출력을 확인하세요."
fi
exit $(( fail > 0 ))
