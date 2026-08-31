#!/usr/bin/env bash
# 4차 — test200(200장) 평가. 5개 모델을 같은 잣대로.
#
#   bash run_test200.sh
#
# ⚠️ RULES 2: test 개방은 마일스톤에서 한 번. 운영 threshold는 이미 val에서 확정했고
#    (val_sweep/operating_threshold.json) 여기서는 그 값을 적용만 한다.
#    test 결과를 보고 epoch·threshold를 되돌아가 바꾸지 않는다.
#
# 대상: zeroshot / FT-206v2 / FT-306v2 / FT-506v2 / FT-1000(3차 ep09)
# 산출: 모델별 predictions.json / threshold_sweep.csv / metrics.json / config.json
#       + viz/thr_<운영thr>/ 200장
set -uo pipefail

R=/workspace/Grounding-Dino-FineTuning
DS=$R/4th/dataset
OUT=$R/4th/results/test200
EVAL=$R/2nd/eval_code/evaluate.py
VIZ=$R/2nd/eval_code/viz_pred.py
OPJSON=$R/4th/results/val_sweep/operating_threshold.json

cd $R/4th
source $R/implant-env/bin/activate
export PYTHONPATH="$R:${PYTHONPATH:-}"
export CUDA_HOME=/usr/local/cuda-12.4; export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH="$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false

[ -f "$OPJSON" ] || { echo "!! 운영 threshold 없음: $OPJSON — run_val_sweep.sh 먼저"; exit 1; }
mkdir -p "$OUT"

NAMES=(zeroshot); CKPTS=("")     # zeroshot은 --weights 없이
for rid in FT-206v2 FT-306v2 FT-506v2; do
  ck=$(python -c "import json;print(json.load(open('$R/4th/results/$rid/best_epoch.json'))['checkpoint'])")
  NAMES+=("$rid"); CKPTS+=("$ck")
done
NAMES+=("FT-1000"); CKPTS+=("$R/3rd/results/FT-1000/weights/epoch_09.pth")

# ── 1. 평가 ─────────────────────────────────────────────────────────────────
for i in "${!NAMES[@]}"; do
  NM="${NAMES[$i]}"; CK="${CKPTS[$i]}"
  echo "════════ $NM  ->  test200 평가 ════════"
  if [ -n "$CK" ] && [ ! -f "$CK" ]; then echo "  !! checkpoint 없음: $CK — 건너뜀"; continue; fi
  if [ -n "$CK" ]; then
    echo "  ckpt: $CK"
    python -u "$EVAL" --out "$OUT/$NM" --data-root "$DS" --split test200 --weights "$CK" \
      || echo "  !! $NM 평가 실패"
  else
    echo "  zero-shot (base 가중치만)"
    python -u "$EVAL" --out "$OUT/$NM" --data-root "$DS" --split test200 \
      || echo "  !! $NM 평가 실패"
  fi
done

# ── 2. 시각화 (val에서 정한 운영 thr 적용) ──────────────────────────────────
echo ""
echo "════════ 시각화 (운영 thr) ════════"
for NM in "${NAMES[@]}"; do
  [ -f "$OUT/$NM/predictions.json" ] || continue
  # zeroshot은 val 기준이 없으므로 0.30 (3차 고정 threshold 표와 동일 기준)
  THR=$(python -c "
import json
op = json.load(open('$OPJSON'))
print(op.get('$NM', 0.30))")
  echo "  $NM: thr $THR"
  python -u "$VIZ" --run "$OUT/$NM" --data-root "$DS" --split test200 --threshold "$THR" \
    || echo "  !! $NM viz 실패"
done

echo ""
echo "=== 완료: $OUT ==="
