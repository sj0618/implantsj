#!/usr/bin/env bash
# test 48장으로 zero-shot / FT-106 / FT-206을 나란히 재고 비교표를 만든다.
# (RULES 2의 마일스톤: best epoch 확정 후 test를 딱 한 번 연다)
#
# 이 스크립트만 실행하면 3단계(test 3모델) + 4단계(비교표)가 한 번에 돈다.
#
#   bash run_test_compare.sh
#
# 전제:
#   - FT-106_lr1e-4, FT-206_lr1e-4 학습이 끝나 best_epoch.json이 있어야 한다.
#     (없으면 명확히 멈춘다 — 미완성 모델로 test를 열지 않는다)
#   - zero-shot은 이미 results/00_zeroshot/에 있다.

set -euo pipefail

R=/workspace/Grounding-Dino-FineTuning
B="$R/2nd"
cd "$B"

# ── 환경 ─────────────────────────────────────────────────────────────────────
source /workspace/implant-env/bin/activate
export PYTHONPATH="$R:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false

FT106_RUN="results/FT-106_lr1e-4"
FT206_RUN="results/FT-206_lr1e-4"
FT106_OUT="results/FT-106_test"
FT206_OUT="results/FT-206_test"
ZS_OUT="results/00_zeroshot"

say() { echo; echo "══════════════════════════════════════════════════════════"; echo " $*"; echo "══════════════════════════════════════════════════════════"; }

# ── 0. 전제 검사 ─────────────────────────────────────────────────────────────
say "0. 학습 완료 여부 검사 (RULES 2: 미완성 모델로 test 열지 않는다)"

check_done() {  # $1=run dir, $2=label
  local bj="$1/best_epoch.json"
  if [[ ! -f "$bj" ]]; then
    echo "[STOP] $2 학습이 안 끝났다 — 없음: $bj"
    echo "       30 epoch가 끝나야 best_epoch.json이 생긴다. test는 그 뒤에만 연다."
    exit 1
  fi
  local ckpt
  ckpt=$(python3 -c "import json;print(json.load(open('$bj'))['checkpoint'])")
  if [[ ! -f "$ckpt" ]]; then
    echo "[STOP] $2 best 체크포인트 없음: $ckpt"
    exit 1
  fi
  local ep vm
  ep=$(python3 -c "import json;print(json.load(open('$bj'))['best_epoch'])")
  vm=$(python3 -c "import json;print(f\"{json.load(open('$bj'))['selected']['val_map']:.4f}\")")
  echo "  ✓ $2  best epoch $ep  (val mAP $vm)"
  echo "$ckpt"
}

if [[ ! -f "$ZS_OUT/metrics.json" ]]; then
  echo "[STOP] zero-shot 결과 없음: $ZS_OUT/metrics.json"
  exit 1
fi
echo "  ✓ zero-shot  이미 있음"
FT106_CKPT=$(check_done "$FT106_RUN" "FT-106" | tail -1)
FT206_CKPT=$(check_done "$FT206_RUN" "FT-206" | tail -1)

# ── 1. test 추론 (FT-106, FT-206) ────────────────────────────────────────────
# zero-shot은 이미 있으니 다시 안 연다. 같은 evaluate.py / 같은 48장이라 비교 가능.
say "1. FT-106 test 48장 (best epoch 체크포인트)"
python eval_code/evaluate.py --out "$FT106_OUT" --weights "$FT106_CKPT"

say "2. FT-206 test 48장 (best epoch 체크포인트)"
python eval_code/evaluate.py --out "$FT206_OUT" --weights "$FT206_CKPT"

# ── 2. 비교표 ────────────────────────────────────────────────────────────────
say "3. 비교표 — zero-shot / FT-106 / FT-206 (같은 test 48장)"
python3 - "$ZS_OUT/metrics.json" "$FT106_OUT/metrics.json" "$FT206_OUT/metrics.json" <<'PY'
import json, sys

specs = [("zero-shot", sys.argv[1]),
         ("FT-106",    sys.argv[2]),
         ("FT-206",    sys.argv[3])]

rows = []
for name, path in specs:
    m = json.load(open(path, encoding="utf-8"))
    ap = m["ap"]                                   # 전체 예측 기준 (threshold 무관)
    sweep = m["threshold_sweep"]
    op = max(sweep, key=lambda r: r["f1"])          # 운영점 = F1 최대 threshold
    rows.append({
        "name": name, "ap50": ap["ap50"], "map": ap["map"],
        "thr": op["threshold"], "P": op["precision"],
        "R": op["recall"], "F1": op["f1"],
        "TP": op["TP"], "FP": op["FP"], "FN": op["FN"],
    })

# 콘솔 표
print(f"\n{'model':<11} {'AP50':>7} {'mAP':>7} │ {'thr':>4} {'P':>6} {'R':>6} {'F1':>6}  "
      f"({'TP':>3}/{'FP':>3}/{'FN':>3})")
print("─" * 74)
for r in rows:
    print(f"{r['name']:<11} {r['ap50']:>7.4f} {r['map']:>7.4f} │ "
          f"{r['thr']:>4.2f} {r['P']:>6.3f} {r['R']:>6.3f} {r['F1']:>6.3f}  "
          f"({r['TP']:>3}/{r['FP']:>3}/{r['FN']:>3})")
print("─" * 74)
print("AP50/mAP: 전체 예측 기준 (threshold 무관) · P/R/F1: F1 최대 threshold(운영점)")

# 증분 해석 (숫자만, 판단은 사람이)
def delta(a, b, k): return rows[b][k] - rows[a][k]
print(f"\n증분:")
print(f"  zero-shot → FT-106 : mAP {delta(0,1,'map'):+.4f}  (파인튜닝 기여)")
print(f"  FT-106   → FT-206 : mAP {delta(1,2,'map'):+.4f}  (데이터 2배 효과)")

# 파일로도 저장
out = {"note": "같은 test 48장. AP50/mAP는 전체 예측, P/R/F1은 F1 최대 threshold.",
       "models": rows}
with open("results/comparison.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

# markdown 표도
md = ["# test 48장 비교 (zero-shot / FT-106 / FT-206)\n",
      "| model | AP50 | mAP@.5:.95 | thr | P | R | F1 | TP/FP/FN |",
      "|---|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['name']} | {r['ap50']:.4f} | {r['map']:.4f} | {r['thr']:.2f} "
              f"| {r['P']:.3f} | {r['R']:.3f} | {r['F1']:.3f} "
              f"| {r['TP']}/{r['FP']}/{r['FN']} |")
md += ["", "- AP50/mAP: 전체 예측 기준 (threshold 무관)",
       "- P/R/F1: F1 최대 threshold(운영점)",
       f"- zero-shot → FT-106: mAP {delta(0,1,'map'):+.4f}",
       f"- FT-106 → FT-206: mAP {delta(1,2,'map'):+.4f} (데이터 2배 효과)"]
with open("results/comparison.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")

print(f"\n저장: results/comparison.json · results/comparison.md")
PY

say "완료"
echo "  results/comparison.md   ← 비교표"
echo "  results/FT-106_test/    ← FT-106 test 상세 (threshold_sweep.csv 등)"
echo "  results/FT-206_test/    ← FT-206 test 상세"
