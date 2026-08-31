#!/usr/bin/env bash
# test 48장 평가 (규칙2/5: 마일스톤에서 딱 한 번). 기본 대상 = FT-1000(라운드6).
# 사용법:  bash eval_test.sh            # FT-1000
#          bash eval_test.sh FT-706     # 다른 run도 원하면
# ⚠️ GPU 사용 → 학습(run_56.sh) 완전히 끝난 뒤 실행
set -uo pipefail
R=/workspace/Grounding-Dino-FineTuning; cd $R/3rd
source $R/implant-env/bin/activate
export PYTHONPATH="$R:${PYTHONPATH:-}"
export CUDA_HOME=/usr/local/cuda-12.4; export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH="$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false

RID=${1:-FT-1000}
RD=$R/3rd/results/$RID
OUT=$R/3rd/results/${RID}_test
[ -f "$RD/best_epoch.json" ] || { echo "!! $RID 미완료(best_epoch.json 없음) — test 안 엶"; exit 1; }
BEST=$(python -c "import json;print(json.load(open('$RD/best_epoch.json'))['checkpoint'])")

echo "════════ $RID  test 48장 평가 ════════"
echo "best checkpoint: $BEST"
echo "[1/2] evaluate (threshold 스윕)"
python $R/2nd/eval_code/evaluate.py --out "$OUT" --split test --weights "$BEST"
echo "[2/2] viz (test 예측 시각화)"
python $R/2nd/eval_code/viz_pred.py --run "$OUT" --threshold 0.30 || echo "  !! viz 실패"

echo "════════ test 결과 요약 ($RID) ════════"
python -c "import json;d=json.load(open('$OUT/metrics.json'));print(json.dumps(d,indent=2,ensure_ascii=False)[:1000])" 2>/dev/null || cat "$OUT/metrics.json"
echo ""
echo "참고 비교(2nd, 같은 test 48장): zero-shot mAP 0.446 / FT-106 0.653 / FT-206 0.732"
echo "=== test 평가 완료: $OUT ==="
