#!/usr/bin/env bash
# 학습 완료된 run들의 결과물 생성 (2nd 구조 맞춤): curves + REPORT.md + viz
# 사용법:  bash post_process.sh              # FT-306/506/706/1000 전부
#          bash post_process.sh FT-706       # 특정 run만
# ⚠️ viz는 GPU 사용 → 학습 다 끝난 뒤 실행할 것
set -uo pipefail
R=/workspace/Grounding-Dino-FineTuning; cd $R/3rd
source $R/implant-env/bin/activate
export PYTHONPATH="$R:${PYTHONPATH:-}"
export CUDA_HOME=/usr/local/cuda-12.4; export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH="$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false

RUNS=("$@")
[ ${#RUNS[@]} -eq 0 ] && RUNS=(FT-306 FT-506 FT-706 FT-1000)

for RID in "${RUNS[@]}"; do
  RD=$R/3rd/results/$RID
  [ -f "$RD/best_epoch.json" ] || { echo "[skip] $RID — best_epoch.json 없음(미완료)"; continue; }
  echo "════════ 후처리: $RID ════════"
  echo "[1/3] curves"; python train_code/make_curves.py --run "$RD" || echo "  !! curves 실패"
  echo "[2/3] REPORT"; python train_code/make_report.py --run "$RD" || echo "  !! report 실패"
  echo "[3/3] viz   "; python train_code/viz_val.py --run "$RD" --threshold 0.30 || echo "  !! viz 실패"
  echo "-> $RID: $(ls $RD | tr '\n' ' ')"
done
echo "=== 후처리 전체 완료 ==="
