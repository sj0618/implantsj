#!/usr/bin/env bash
# test200 (200장) 평가 — FT-106 ~ FT-1000 (6개 모델) 한 번에.
# 각 모델 best checkpoint 를 고정 test200 에 얹어 같은 잣대로 지표 산출.
# 사용:  bash eval_test200.sh
# ⚠️ GPU 사용 — 학습 등 다른 GPU 작업 없을 때 실행.
set -uo pipefail
R=/workspace/Grounding-Dino-FineTuning
cd "$R/3rd"
source "$R/implant-env/bin/activate"
export PYTHONPATH="$R:${PYTHONPATH:-}"
export CUDA_HOME=/usr/local/cuda-12.4; export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false

DATA_ROOT="$R/3rd/dataset"
SPLIT="test200"
EVAL="$R/2nd/eval_code/evaluate.py"
OUTBASE="$R/3rd/results/test200"

NAMES=(FT-106 FT-206 FT-306 FT-506 FT-706 FT-1000)
CKPTS=(
  "$R/2nd/results/01/FT-106_lr1e-4/weights/epoch_24.pth"
  "$R/2nd/results/02/FT-206_lr1e-4/weights/epoch_19.pth"
  "$R/3rd/results/FT-306/weights/epoch_18.pth"
  "$R/3rd/results/FT-506/weights/epoch_08.pth"
  "$R/3rd/results/FT-706/weights/epoch_11.pth"
  "$R/3rd/results/FT-1000/weights/epoch_09.pth"
)

[ -f "$DATA_ROOT/$SPLIT/annotations.json" ] || { echo "!! GT 없음: $DATA_ROOT/$SPLIT/annotations.json"; exit 1; }

for i in "${!NAMES[@]}"; do
  NM="${NAMES[$i]}"; CK="${CKPTS[$i]}"
  echo "════════ $NM  ->  test200 평가 ════════"
  if [ ! -f "$CK" ]; then echo "  !! checkpoint 없음: $CK — 건너뜀"; continue; fi
  echo "  ckpt: $CK"
  python "$EVAL" --out "$OUTBASE/$NM" --data-root "$DATA_ROOT" --split "$SPLIT" --weights "$CK" \
    || echo "  !! $NM 평가 실패"
done

echo ""
echo "════════ test200 요약 (200장) ════════"
python - <<'PY'
import json, os
base="/workspace/Grounding-Dino-FineTuning/3rd/results/test200"
order=["FT-106","FT-206","FT-306","FT-506","FT-706","FT-1000"]
print(f"{'model':8s} {'AP50':>7s} {'mAP':>7s} {'thr*':>5s} {'P':>6s} {'R':>6s} {'F1':>6s}  {'TP/FP/FN':>12s}")
prev=None
for nm in order:
    mp=os.path.join(base,nm,"metrics.json")
    if not os.path.exists(mp):
        print(f"{nm:8s}  (metrics.json 없음)"); continue
    d=json.load(open(mp)); ap=d.get("ap",{}); sweep=d.get("threshold_sweep",[])
    best=max(sweep, key=lambda r: r["f1"]) if sweep else None
    m=ap.get("map",0)
    delta=f"  (Δ {m-prev:+.4f})" if prev is not None else ""
    if best:
        tfn=f"{best['TP']}/{best['FP']}/{best['FN']}"
        print(f"{nm:8s} {ap.get('ap50',0):7.4f} {m:7.4f} {best['threshold']:5.2f} "
              f"{best['precision']:6.3f} {best['recall']:6.3f} {best['f1']:6.3f}  {tfn:>12s}{delta}")
    else:
        print(f"{nm:8s} {ap.get('ap50',0):7.4f} {m:7.4f}   (sweep 없음){delta}")
    prev=m
print("\n* thr = F1 최대 threshold. AP50/mAP 은 threshold 무관(전체 예측 기준).")
print("* 참고(2nd, test 48장): FT-106 mAP 0.653 / FT-206 0.732")
PY
echo "=== 완료: $OUTBASE/<model>/ (metrics.json, threshold_sweep.csv, predictions.json) ==="
