#!/usr/bin/env bash
# GroundingDINO _C(Deformable Attention) 확장을 현재 GPU(A4500, sm_86)로 재빌드
set -euo pipefail
R=/workspace/Grounding-Dino-FineTuning; cd $R
source $R/implant-env/bin/activate
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="8.6+PTX"   # A4500 sm_86 + PTX(향후 GPU 바뀌어도 JIT 대응)

echo "[1/3] 기존 _C.so / build 제거"
rm -f groundingdino/_C*.so
rm -rf build
echo "[2/3] 재빌드 (sm_86) — 1~3분 소요"
python setup.py build_ext --inplace 2>&1 | tail -12
echo "[3/3] 커널 검증"
python - <<'PYEOF'
import torch
print("torch", torch.__version__, "| GPU", torch.cuda.get_device_name(0), "| cap", torch.cuda.get_device_capability())
import groundingdino._C  # import 자체 확인
try:
    from groundingdino.models.GroundingDINO.ms_deform_attn import MultiScaleDeformableAttnFunction as MF
    v=torch.rand(1,3,8,32,device="cuda"); sh=torch.tensor([[1,3]],device="cuda",dtype=torch.long)
    ls=torch.tensor([0],device="cuda",dtype=torch.long)
    lo=torch.rand(1,2,8,1,4,2,device="cuda"); at=torch.rand(1,2,8,1,4,device="cuda")
    o=MF.apply(v,sh,ls,lo,at,64); torch.cuda.synchronize()
    print(">>> MSDA CUDA 커널 실행 OK", tuple(o.shape), "<<<  재빌드 성공")
except Exception as e:
    print(">>> 커널 테스트:", repr(e)[:140])
PYEOF
echo "=== 완료. 이제 run_all.sh 다시 실행하세요 ==="
