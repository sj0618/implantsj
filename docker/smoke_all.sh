#!/usr/bin/env bash
# =====================================================================
# 통합 이미지 자가진단. 실제 데이터 없이 돈다.
#
#   docker run --rm implantid:latest smoke
#
# implantsj 는 합성 데이터로 파이프라인 전체를 돌리는 기존 10단계를 그대로 쓴다.
# implantys / implant_ny 는 이 저장소에 자가진단이 없어서, 환경이 살아 있는지
# (import / 확장 로드 / 가중치 존재)까지만 본다. 그 이상을 검사한다고 말하지 않는다.
# =====================================================================
set -uo pipefail
ROOT=/workspace

pass=0; fail=0
step() {
  local name="$1"; shift
  printf '\n=== %s ===\n' "$name"
  if "$@"; then printf 'PASS  %s\n' "$name"; pass=$((pass+1));
  else printf 'FAIL  %s\n' "$name"; fail=$((fail+1)); fi
}

s_sj_full() {
  cd "$ROOT/implantsj" && PYTHON=/opt/venv/sj/bin/python ./docker/smoke_test.sh
}

s_ys_env() {
  cd "$ROOT/implantys"
  export LD_LIBRARY_PATH="$(/opt/venv/ys/bin/python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH:-}"
  /opt/venv/ys/bin/python -c "
import torch, numpy, cv2, transformers, timm
from groundingdino import _C
from groundingdino.util.inference import load_model
from pathlib import Path
w = Path('weights/groundingdino_swint_ogc.pth')
assert w.is_file() and w.stat().st_size > 500*1048576, w
print(f'torch {torch.__version__} numpy {numpy.__version__} | gdino CUDA 확장 OK | {w.name} {w.stat().st_size/1048576:.0f} MB')
"
}

s_ys_offline_bert() {
  # 폐쇄망에서 gdino 가 텍스트 인코더를 받으러 나가면 학습이 멈춘다.
  # HF_HUB_OFFLINE=1 상태에서 캐시만으로 로드되는지 본다.
  cd "$ROOT/implantys"
  /opt/venv/ys/bin/python -c "
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained('bert-base-uncased')
AutoModel.from_pretrained('bert-base-uncased')
print('bert-base-uncased 오프라인 로드 OK')
"
}

s_ny_env() {
  cd "$ROOT/implant_ny"
  /opt/venv/ny/bin/python -c "
import torch, numpy, cv2, sam2, segment_anything
from pathlib import Path
import zipfile
p = Path('checkpoints/sam2.1_hiera_tiny.pt')
assert p.is_file() and zipfile.is_zipfile(p) and p.stat().st_size > 100*1048576, p
print(f'torch {torch.__version__} numpy {numpy.__version__} | sam2 OK | {p.name} {p.stat().st_size/1048576:.0f} MB')
"
}

s_isolation() {
  # 세 환경이 실제로 분리돼 있는지. 하나로 합쳐지면 핀이 깨진 것이다.
  /opt/venv/sj/bin/python -c "import torch,numpy;print('sj', torch.__version__, numpy.__version__)"
  /opt/venv/ys/bin/python -c "import torch,numpy;print('ys', torch.__version__, numpy.__version__)"
  /opt/venv/ny/bin/python -c "import torch,numpy;print('ny', torch.__version__, numpy.__version__)"
  /opt/venv/ys/bin/python -c "import numpy,sys;sys.exit(0 if numpy.__version__.startswith('1.26') else 1)" || return 1
  /opt/venv/ny/bin/python -c "import numpy,sys;sys.exit(0 if numpy.__version__.startswith('2.') else 1)" || return 1
}

step "1. 환경 격리 (venv 3개 핀 확인)"      s_isolation
step "2. implantsj 전체 파이프라인 10단계"   s_sj_full
step "3. implantys 환경 + gdino CUDA 확장"  s_ys_env
step "4. implantys BERT 오프라인 로드"       s_ys_offline_bert
step "5. implant_ny 환경 + SAM2 체크포인트"  s_ny_env

printf '\n----------------------------------------\n'
printf '통과 %d / 실패 %d\n' "$pass" "$fail"
if [[ $fail -eq 0 ]]; then
  echo "전부 통과."
  echo "주의: 3~5 는 환경과 가중치가 살아 있는지만 본 것이다."
  echo "      implantys / implant_ny 의 학습·평가 로직은 이 저장소에 자가진단이 없어"
  echo "      검사하지 않았다. 실제 데이터로 처음 돌릴 때 따로 확인해야 한다."
else
  echo "실패한 단계의 출력을 확인하세요."
fi
exit $(( fail > 0 ))
