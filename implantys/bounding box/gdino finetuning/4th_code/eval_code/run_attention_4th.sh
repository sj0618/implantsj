#!/usr/bin/env bash
# 4차 — attention map 추출. (4th_plan ④)
#
#   bash run_attention_4th.sh
#
# 1) FT-1000으로 test200 200장 전부  -> results/attention/FT-1000/
# 2) 케이스(FN 전부 + FP·TP 대표)를 4모델 각각 -> results/attention/<model>/
#    (패널은 make_panels_4th.py가 이 결과를 가로로 붙인다)
#
# 2nd/eval_code/attention_map.py를 그대로 쓴다 (--images-dir/--images를 받으므로 수정 불필요).
# encoder fusion layer의 텍스트("implant")↔이미지 cross-attention. (RULES 7)
set -uo pipefail

R=/workspace/Grounding-Dino-FineTuning
DS=$R/4th/dataset/test200/images
OUT=$R/4th/results/attention
ATT=$R/2nd/eval_code/attention_map.py
CASES=$OUT/cases.json

cd $R/4th
source $R/implant-env/bin/activate
export PYTHONPATH="$R:${PYTHONPATH:-}"
export CUDA_HOME=/usr/local/cuda-12.4; export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH="$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false

[ -f "$CASES" ] || { echo "!! cases.json 없음 — select_cases_4th.py 먼저"; exit 1; }

CKPT_1000=$R/3rd/results/FT-1000/weights/epoch_09.pth

# ── 1. FT-1000: test200 전량 ────────────────────────────────────────────────
echo "════════ FT-1000 | test200 200장 전량 ════════"
ALL=$(ls "$DS" | grep -E "\.jpg$" | tr "\n" " ")
python -u "$ATT" --out "$OUT/FT-1000" --images-dir "$DS" --weights "$CKPT_1000" \
    --images $ALL 2>&1 | tail -5

# ── 2. 케이스: 나머지 3모델 (FT-1000은 위에서 이미 전량 처리됨) ─────────────
CASE_IMGS=$(python -c "
import json
d = json.load(open('$CASES'))
print(' '.join(sorted(set(d['fn'] + d['fp'] + d['tp']))))")
echo ""
echo "케이스 $(echo $CASE_IMGS | wc -w)장 (FN 전부 + FP·TP 대표)"

for rid in FT-206v2 FT-306v2 FT-506v2; do
  CK=$(python -c "import json;print(json.load(open('$R/4th/results/$rid/best_epoch.json'))['checkpoint'])")
  echo "════════ $rid | 케이스 ════════"
  python -u "$ATT" --out "$OUT/$rid" --images-dir "$DS" --weights "$CK" \
      --images $CASE_IMGS 2>&1 | tail -3
done

echo ""
echo "=== attention 추출 완료: $OUT ==="
ls "$OUT"
