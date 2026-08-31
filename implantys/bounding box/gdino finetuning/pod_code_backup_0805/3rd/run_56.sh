#!/usr/bin/env bash
# 05·06 라운드 연속 학습: FT-706(+05) -> FT-1000(+06)
# (306/506은 이미 완료됨. 매 run 원본 pretrained에서 새로, 2nd와 동일 설정)
set -uo pipefail
R=/workspace/Grounding-Dino-FineTuning; cd $R/3rd
source $R/implant-env/bin/activate
export PYTHONPATH="$R:${PYTHONPATH:-}"
export CUDA_HOME=/usr/local/cuda-12.4; export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH="$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false

run(){ local RID=$1; shift
  echo "==================== $RID  (data: $*)  $(date +%F_%H:%M) ===================="
  python -u train_code/train.py --data "$@" --lr 1e-4 --run-id "$RID" --results-root $R/3rd/results 2>&1 | tee /tmp/${RID}.log
  local st=${PIPESTATUS[0]}
  [ $st -eq 0 ] || { echo "!! $RID 실패 (exit $st) — 중단"; exit 1; }
}

run FT-706  01 02 03 04 05
run FT-1000 01 02 03 04 05 06

echo "=== ALL DONE (FT-706 / FT-1000) ==="
