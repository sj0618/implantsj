#!/usr/bin/env bash
# RunPod 환경 재구축. pod을 새로 띄울 때마다 이 스크립트 하나로 끝낸다.
#
# 전제 도커 이미지 (반드시 devel — nvcc가 있어야 ms_deform_attn을 빌드한다):
#   runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#
# 사용:
#   cd /workspace && bash setup_pod.sh
#
# 끝나면:
#   source /workspace/implant-env/bin/activate
#
# ⚠️ 작업 후엔 terminate 말고 stop. terminate하면 /workspace까지 날아가서
#    이 스크립트를 처음부터 다시 돌려야 한다.

set -euo pipefail

PROJ="${PROJ_ROOT:-/workspace}"
VENV="${PROJ}/implant-env"
REPO="${PROJ}/Grounding-Dino-FineTuning"
REPO_URL="https://github.com/Asad-Ismail/Grounding-Dino-FineTuning"
WEIGHTS_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

say() { echo; echo "═══ $* ═══"; }

# ── 0. 사전 점검 ─────────────────────────────────────────────────────────────
say "0/7 사전 점검"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "[ERROR] nvcc가 없다. devel 이미지가 아니다." >&2
  echo "        ms_deform_attn CUDA 확장을 빌드할 수 없어 학습이 불가능하다." >&2
  echo "        이미지: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" >&2
  exit 1
fi
echo "  nvcc      : $(nvcc --version | grep release | sed 's/.*release //')"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi가 없다. GPU pod이 맞는지 확인할 것." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/  GPU       : /'

echo "  디스크    : $(df -h "$PROJ" | awk 'NR==2{print $4" 여유"}')"
# torch(~2.5GB) + 가중치(~700MB) + 빌드 산출물. 최소 15GB는 있어야 안전하다.

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
[[ -d "$CUDA_HOME" ]] || { echo "[ERROR] CUDA_HOME 없음: $CUDA_HOME" >&2; exit 1; }
echo "  CUDA_HOME : $CUDA_HOME"

# ── 1. venv ─────────────────────────────────────────────────────────────────
say "1/7 venv 생성: $VENV"
# /workspace에 두는 이유: RunPod에서 여기만 영구 볼륨이다.
# 시스템 python에 설치하면 pod stop/start 시 패키지가 전부 사라진다.
if [[ -d "$VENV" ]]; then
  echo "  이미 존재 — 재사용"
else
  python -m venv "$VENV"
  echo "  생성 완료"
fi

# ── 2. LD_LIBRARY_PATH 영구 설정 ────────────────────────────────────────────
say "2/7 LD_LIBRARY_PATH 영구 설정"
# ms_deform_attn 확장이 libc10.so 등 torch의 .so를 런타임에 찾는다.
# 이걸 안 잡으면 import 시점에 깨진다. activate에 박아 매번 안 잡아도 되게 한다.
if ! grep -q "IMPLANT_LD_PATH" "$VENV/bin/activate"; then
  cat >> "$VENV/bin/activate" <<'EOF'

# IMPLANT_LD_PATH — ms_deform_attn이 torch의 .so를 찾게 한다. 지우지 말 것.
_torch_lib="$("$VIRTUAL_ENV/bin/python" -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))' 2>/dev/null || true)"
if [ -n "$_torch_lib" ]; then
  export LD_LIBRARY_PATH="${_torch_lib}:${LD_LIBRARY_PATH:-}"
fi
unset _torch_lib
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
EOF
  echo "  activate에 추가"
else
  echo "  이미 설정됨"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip -q
echo "  python    : $(python -V)  ($(which python))"

# ── 3. torch (버전 고정) ────────────────────────────────────────────────────
say "3/7 torch 설치 (버전 고정)"
# 이 핀들을 어기면 알려진 에러가 재발한다:
#   torch 2.6+        -> ms_deform_attn CUDA 컴파일 에러
#   transformers 5.x  -> BERT get_head_mask 에러
#   huggingface_hub 신버전 -> get_full_repo_name 없음
pip install -q \
  torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu124
python - <<'EOF'
import torch
print(f"  torch     : {torch.__version__}")
print(f"  cuda      : {torch.version.cuda} | available={torch.cuda.is_available()}")
assert torch.cuda.is_available(), "CUDA를 못 잡았다. GPU pod인지 확인할 것."
EOF

# 2단계에서 activate에 심어둔 LD_LIBRARY_PATH 설정은 그때 torch가 없어서 빈 값이었다.
# 이제 torch가 있으니 현재 셸에도 직접 잡아준다. 이걸 빠뜨리면 아래 5단계 빌드와
# 7단계 검증에서 ms_deform_attn이 libc10.so를 못 찾아 깨진다.
TORCH_LIB="$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))')"
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
echo "  LD_LIBRARY_PATH += ${TORCH_LIB}"

# ── 4. 나머지 의존성 (버전 고정) ────────────────────────────────────────────
say "4/7 의존성 설치"

# 빌드 사전 준비. ninja가 없으면 CUDA 확장 컴파일이 극도로 느리거나 실패한다.
pip install -q wheel setuptools ninja

pip install -q \
  "transformers==4.44.2" \
  "huggingface_hub==0.24.6"

# 지원 패키지. peft는 LoRA의 핵심 — 없으면 use_lora=true에서 바로 터진다.
pip install -q \
  peft \
  supervision \
  opencv-python-headless \
  pycocotools \
  timm \
  ema-pytorch

# 이 프로젝트 스크립트용 (make_curves / viz_pred / config yaml)
pip install -q matplotlib pillow pyyaml

python - <<'EOF'
import transformers, huggingface_hub, peft, numpy, cv2
print(f"  transformers   : {transformers.__version__}")
print(f"  huggingface_hub: {huggingface_hub.__version__}")
print(f"  peft           : {peft.__version__}")
print(f"  opencv         : {cv2.__version__}")
print(f"  numpy          : {numpy.__version__}")
EOF

# numpy는 핀하지 않는다. 기존에 검증된 환경 기록에 numpy 핀이 없었고,
# 검증된 조합에 근거 없는 핀을 더하면 의존성 해석이 깨질 수 있다.
# 다만 numpy 2.x에서 pycocotools가 ABI 에러("numpy.dtype size changed")를
# 내는 사례가 있다. 그 에러가 나면 그때 아래를 실행할 것:
#     pip install -q "numpy<2" && pip install -q --force-reinstall --no-cache-dir pycocotools
python - <<'EOF'
import numpy
from pycocotools.coco import COCO      # numpy 2.x ABI 충돌이면 여기서 깨진다
print(f"  pycocotools    : import OK (numpy {numpy.__version__})")
EOF

# ── 5. repo clone + 빌드 ────────────────────────────────────────────────────
say "5/7 repo clone + CUDA 확장 빌드"
if [[ -d "$REPO/.git" ]]; then
  echo "  이미 존재 — clone 건너뜀: $REPO"
else
  git clone "$REPO_URL" "$REPO"
fi
cd "$REPO"
echo "  repo rev  : $(git rev-parse --short HEAD)"

# --no-build-isolation: 격리하면 pip이 빌드용 torch를 새로 받아 위 핀이 무시된다.
# 반드시 이 옵션으로 빌드해야 방금 설치한 torch 2.4.1에 맞춰 컴파일된다.
pip install -e . --no-build-isolation

# ── 6. 사전학습 가중치 ──────────────────────────────────────────────────────
say "6/7 사전학습 가중치"
mkdir -p "$REPO/weights"
W="$REPO/weights/groundingdino_swint_ogc.pth"
if [[ -f "$W" ]]; then
  echo "  이미 존재 — 건너뜀 ($(du -h "$W" | cut -f1))"
else
  wget -q --show-progress -O "$W" "$WEIGHTS_URL"
  echo "  받음: $(du -h "$W" | cut -f1)"
fi
# 받다 만 파일이면 로딩 시점에 이상한 에러가 난다. 크기로 미리 거른다. (정상 ~660MB)
SZ=$(stat -c%s "$W" 2>/dev/null || stat -f%z "$W")
if (( SZ < 500000000 )); then
  echo "[ERROR] 가중치가 너무 작다(${SZ} bytes). 받다 말았다. 지우고 다시 받을 것:" >&2
  echo "        rm $W && bash setup_pod.sh" >&2
  exit 1
fi

# BERT 텍스트 인코더 미리 받기 (첫 학습 때 받으면 네트워크 이슈로 중간에 죽는다)
python - <<'EOF'
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained("bert-base-uncased")
AutoModel.from_pretrained("bert-base-uncased")
print("  bert-base-uncased 캐시 완료")
EOF

# ── 7. 검증 ─────────────────────────────────────────────────────────────────
say "7/7 검증"
cd "$REPO"
python - <<'EOF'
import torch
# ms_deform_attn이 실제로 로드되는지 — 여기서 깨지면 빌드가 실패한 것이다.
from groundingdino.util.inference import load_model, load_image, predict
import groundingdino
print(f"  groundingdino import OK")

class MC:
    config_path  = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    weights_path = "weights/groundingdino_swint_ogc.pth"
    lora_weights = None
    lora_weigths = None

m = load_model(MC(), use_lora=False)
n = sum(p.numel() for p in m.parameters())
print(f"  모델 로드 OK — 파라미터 {n/1e6:.1f}M")
EOF

echo
echo "══════════════════════════════════════════════════════════"
echo " 환경 재구축 완료"
echo
echo "   source ${VENV}/bin/activate"
echo "   cd ${REPO}"
echo
echo " 다음:"
echo "   1) 로컬에서 dataset2/ 와 'train_code'/'eval_code'를 pod으로 복사"
echo "   2) TRAIN_PATCH.md 대로 repo의 train.py에 RunLogger 심기"
echo "   3) attention_map.py의 hook 채우기 (RULES 7: 학습 전에)"
echo
echo " ⚠️ 작업 후 terminate 금지. stop을 쓸 것."
echo "    terminate하면 /workspace가 날아가 이 스크립트를 처음부터 다시 돌려야 한다."
echo "══════════════════════════════════════════════════════════"
