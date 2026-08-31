#!/usr/bin/env bash
# 3rd 라운드 학습 연속 실행. 진행상황 터미널 실시간 출력 + /tmp 로그.
# 매 run 원본 pretrained에서 새로 시작 (2nd와 동일: lr 1e-4, batch 4, seed 42, 30 epoch, best=val mAP)
set -uo pipefail
R=/workspace/Grounding-Dino-FineTuning; cd $R/3rd
source $R/implant-env/bin/activate
export PYTHONPATH="$R:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false

run(){ local RID=$1; shift
  echo "==================== $RID  (data: $*)  $(date +%F_%H:%M) ===================="
  python -u train_code/train.py --data "$@" --lr 1e-4 --run-id "$RID" --results-root $R/3rd/results 2>&1 | tee /tmp/${RID}.log
  local st=${PIPESTATUS[0]}
  [ $st -eq 0 ] || { echo "!! $RID 실패 (exit $st) — 중단"; exit 1; }
}

# ===== 이번엔 3·4·5만 =====
run FT-306  01 02 03
run FT-506  01 02 03 04
run FT-706  01 02 03 04 05
# run FT-1000 01 02 03 04 05 06   # ← 라운드6(1000, test평가)은 나중에 이 줄 주석 해제

echo "=== ALL DONE (FT-306 / FT-506 / FT-706) ==="
