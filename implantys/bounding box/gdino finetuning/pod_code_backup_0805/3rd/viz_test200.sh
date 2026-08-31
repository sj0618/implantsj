#!/usr/bin/env bash
# test200 시각화 — zero-shot ~ FT-1000 (7개) 각 모델 예측 박스 그림 생성.
# zero-shot은 test200 평가가 아직 없으므로 여기서 함께 산출.
# viz threshold=0.30 통일(모델 간 비교용). GT=초록 / TP=파랑 / FP=빨강 / FN=노랑점선.
set -uo pipefail
R=/workspace/Grounding-Dino-FineTuning
cd "$R/3rd"
source "$R/implant-env/bin/activate"
export PYTHONPATH="$R:${PYTHONPATH:-}"
export CUDA_HOME=/usr/local/cuda-12.4; export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false

DATA_ROOT="$R/3rd/dataset"; SPLIT="test200"
EVAL="$R/2nd/eval_code/evaluate.py"; VIZ="$R/2nd/eval_code/viz_pred.py"
OUTBASE="$R/3rd/results/test200"; THR=0.30

# 1) zero-shot (base 가중치만) — predictions 없으면 생성
if [ ! -f "$OUTBASE/zeroshot/predictions.json" ]; then
  echo "════════ zero-shot -> test200 평가 ════════"
  python "$EVAL" --out "$OUTBASE/zeroshot" --data-root "$DATA_ROOT" --split "$SPLIT" || echo "  !! zeroshot 평가 실패"
fi

# 2) viz — 7개 run 전부
for NM in zeroshot FT-106 FT-206 FT-306 FT-506 FT-706 FT-1000; do
  RD="$OUTBASE/$NM"
  echo "════════ viz: $NM (thr $THR) ════════"
  if [ ! -f "$RD/predictions.json" ]; then echo "  !! predictions.json 없음: $RD — 건너뜀"; continue; fi
  python "$VIZ" --run "$RD" --threshold "$THR" --data-root "$DATA_ROOT" --split "$SPLIT" \
    || echo "  !! $NM viz 실패"
done
echo "=== viz 완료: $OUTBASE/<model>/viz/thr_0.30/ ==="
