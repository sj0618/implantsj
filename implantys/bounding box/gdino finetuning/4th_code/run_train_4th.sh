#!/usr/bin/env bash
# 4차 재학습 — FT-206v2 / FT-306v2 / FT-506v2 (고정 val 299장)
#
#   bash run_train_4th.sh            # 세 개 순차 실행
#   bash run_train_4th.sh 306        # 하나만
#   bash run_train_4th.sh 306 506    # 골라서
#
# 노트북 덮어도 안 죽게 하려면 (권장):
#   cd /workspace/Grounding-Dino-FineTuning/4th
#   setsid bash run_train_4th.sh > /tmp/4th_all.log 2>&1 < /dev/null &
#   tail -f /tmp/4th_all.log        # 진행 확인 (Ctrl-C 해도 학습은 계속됨)
#
# 4차는 자체 코드로 돈다 (2nd/3rd 코드·데이터를 건드리지 않는다):
#   4th/train_code/train_4th.py        학습 (2nd/train_code/train.py 복사 + 데이터 지정만 변경)
#   4th/train_code/coco_to_csv_4th.py  COCO -> CSV (--train-dir/--val-dir 직접 지정)
#   4th/dataset/{train_206,train_306,train_506,val,test200}
#
# 후처리(curves/REPORT/viz)는 데이터 배치와 무관하게 run 폴더만 읽으므로 2nd 것을 그대로 쓴다.
# 설정은 3차와 동일: lr 1e-4 · batch 4 · 30 epoch · seed 42 · best=val mAP@.5:.95
set -uo pipefail

R=/workspace/Grounding-Dino-FineTuning
TC=$R/4th/train_code
RESULTS=$R/4th/results

cd $R/4th
source $R/implant-env/bin/activate
export PYTHONPATH="$R:${PYTHONPATH:-}"
export CUDA_HOME=/usr/local/cuda-12.4; export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH="$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false

SETS=("$@")
[ ${#SETS[@]} -eq 0 ] && SETS=(206 306 506)

mkdir -p "$RESULTS"

for N in "${SETS[@]}"; do
  RID="FT-${N}v2"
  RD="$RESULTS/$RID"

  if [ -f "$RD/best_epoch.json" ]; then
    echo "[skip] $RID — 이미 완료됨 (best_epoch.json 있음). 다시 돌리려면 폴더를 옮기거나 지울 것"
    continue
  fi

  # 디스크가 막히면 학습이 중간에 죽는다 (2차에서 epoch 28에 당함) — 매번 확인
  if ! dd if=/dev/zero of="$R/4th/.write_test" bs=1M count=12 2>/dev/null; then
    echo "!! 디스크 쓰기 실패 — 공간 확보 후 재시도. 중단"; rm -f "$R/4th/.write_test"; exit 1
  fi
  rm -f "$R/4th/.write_test"

  echo ""
  echo "==================== $RID  (train_$N + 고정 val)  $(date +%F_%H:%M) ===================="
  python -u "$TC/train_4th.py" \
      --set "$N" --lr 1e-4 --run-id "$RID" --results-root "$RESULTS" \
      2>&1 | tee "/tmp/${RID}.log"
  st=${PIPESTATUS[0]}
  [ $st -eq 0 ] || { echo "!! $RID 학습 실패 (exit $st) — 중단"; exit 1; }

  # ── 후처리: curves / REPORT / viz (규칙 4 산출물) ─────────────────────────
  echo "──── 후처리: $RID ────"
  python $R/2nd/train_code/make_curves.py --run "$RD"                  || echo "  !! curves 실패"
  python $R/2nd/train_code/make_report.py --run "$RD"                  || echo "  !! report 실패"
  python $R/2nd/train_code/viz_val.py     --run "$RD" --threshold 0.30 || echo "  !! viz 실패"

  # ── 산출물 검증 (하나라도 없으면 실패로 본다) ─────────────────────────────
  miss=""
  for f in config.json metrics.csv steps.csv best_epoch.json \
           curves/loss_curve.png curves/lr_schedule.png REPORT.md; do
    [ -e "$RD/$f" ] || miss="$miss $f"
  done
  [ -d "$RD/viz" ] && [ "$(ls "$RD/viz" | wc -l)" -gt 0 ] || miss="$miss viz/"
  if [ -n "$miss" ]; then
    echo "  !! 산출물 누락:$miss"
  else
    echo "  산출물 확인 OK — best epoch $(python -c "import json;print(json.load(open('$RD/best_epoch.json'))['best_epoch'])")"
  fi
done

echo ""
echo "=== 학습 완료: ${SETS[*]} ==="
echo "결과: $RESULTS   로그: /tmp/FT-*v2.log"
