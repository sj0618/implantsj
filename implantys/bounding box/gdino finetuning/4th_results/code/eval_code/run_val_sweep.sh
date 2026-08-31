#!/usr/bin/env bash
# 4차 — 고정 val(299장)에서 threshold sweep. 운영 threshold를 val로만 정한다. (RULES 2)
#
#   bash run_val_sweep.sh
#
# test를 열기 전에 먼저 돌린다. 여기서 정한 thr를 test200에 그대로 적용한다.
# 2nd/eval_code/evaluate.py를 그대로 쓴다 (--data-root/--split를 받으므로 수정 불필요).
# 대상: FT-206v2 / FT-306v2 / FT-506v2 (각 best epoch) + FT-1000 (3차 ep09 그대로)
set -uo pipefail

R=/workspace/Grounding-Dino-FineTuning
DS=$R/4th/dataset
OUT=$R/4th/results/val_sweep
EVAL=$R/2nd/eval_code/evaluate.py

cd $R/4th
source $R/implant-env/bin/activate
export PYTHONPATH="$R:${PYTHONPATH:-}"
export CUDA_HOME=/usr/local/cuda-12.4; export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH="$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT"

# v2 3개: best_epoch.json에서 체크포인트 경로를 읽는다 (하드코딩하지 않는다)
NAMES=(); CKPTS=()
for rid in FT-206v2 FT-306v2 FT-506v2; do
  ck=$(python -c "import json;print(json.load(open('$R/4th/results/$rid/best_epoch.json'))['checkpoint'])")
  NAMES+=("$rid"); CKPTS+=("$ck")
done
# FT-1000: 3차 run 그대로 (고정 val = 이 run의 val이라 재학습 불필요)
NAMES+=("FT-1000"); CKPTS+=("$R/3rd/results/FT-1000/weights/epoch_09.pth")

for i in "${!NAMES[@]}"; do
  NM="${NAMES[$i]}"; CK="${CKPTS[$i]}"
  echo "════════ $NM  ->  val sweep ════════"
  if [ ! -f "$CK" ]; then echo "  !! checkpoint 없음: $CK — 건너뜀"; continue; fi
  echo "  ckpt: $CK"
  python -u "$EVAL" --out "$OUT/$NM" --data-root "$DS" --split val --weights "$CK" \
    || echo "  !! $NM 실패"
done

echo ""
echo "════════ val 운영 threshold (F1 최대) ════════"
python - <<'PY'
import json, os
base = "/workspace/Grounding-Dino-FineTuning/4th/results/val_sweep"
print("%-10s %7s %7s %6s %7s %7s %7s" % ("model", "AP50", "mAP", "thr*", "P", "R", "F1"))
op = {}
for nm in ["FT-206v2", "FT-306v2", "FT-506v2", "FT-1000"]:
    p = os.path.join(base, nm, "metrics.json")
    if not os.path.exists(p):
        print("%-10s (없음)" % nm); continue
    d = json.load(open(p))
    ap = d.get("ap", {}); sweep = d.get("threshold_sweep", [])
    b = max(sweep, key=lambda r: r["f1"]) if sweep else None
    if b:
        op[nm] = b["threshold"]
        print("%-10s %7.4f %7.4f %6.2f %7.3f %7.3f %7.3f"
              % (nm, ap.get("ap50", 0), ap.get("map", 0), b["threshold"],
                 b["precision"], b["recall"], b["f1"]))
json.dump(op, open(os.path.join(base, "operating_threshold.json"), "w"), indent=2)
print("\n-> 운영 threshold 저장:", os.path.join(base, "operating_threshold.json"))
print("   (이 값을 test200에 그대로 적용한다. test를 보고 고르지 않는다 — RULES 2)")
PY
