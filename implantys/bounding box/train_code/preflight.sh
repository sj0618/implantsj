#!/usr/bin/env bash
# zero-shot(00) 돌리기 전 점검. 실제로 추론을 시작하기 전에 깨질 것들을 미리 잡는다.
#
# 사용 (pod):
#   export PROJ_ROOT=/workspace/Grounding-Dino-FineTuning/2nd
#   source /workspace/implant-env/bin/activate
#   bash "$PROJ_ROOT/train_code/preflight.sh"

set -uo pipefail        # -e 없음: 전부 검사하고 마지막에 한 번에 보고한다
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/common/config.sh"

FAIL=0
ok()   { echo "  ✅ $*"; }
bad()  { echo "  ❌ $*"; FAIL=$((FAIL+1)); }
warn() { echo "  ⚠️  $*"; }

echo "═══ 경로 ═══"
echo "  PROJ_ROOT   : $PROJ_ROOT"
echo "  GDINO_ROOT  : $GDINO_ROOT"
echo "  DATA_ROOT   : $DATA_ROOT"
echo "  RESULT_ROOT : $RESULT_ROOT"

for d in "$PROJ_ROOT" "$GDINO_ROOT" "$DATA_ROOT" "$RESULT_ROOT" "$TRAIN_CODE" "$EVAL_CODE"; do
  [[ -d "$d" ]] && ok "존재: $d" || bad "없음: $d"
done

echo
echo "═══ venv ═══"
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  ok "activate됨: $VIRTUAL_ENV"
else
  bad "venv가 activate되지 않았다 — source /workspace/implant-env/bin/activate"
fi
echo "  python: $(which python)"

echo
echo "═══ 패키지 ═══"
python - <<'EOF' || true
import importlib, sys
need = [("torch","2.4.1"), ("torchvision","0.19.1"), ("transformers","4.44.2"),
        ("huggingface_hub","0.24.6"), ("peft",None), ("cv2",None),
        ("pycocotools",None), ("timm",None), ("supervision",None),
        ("matplotlib",None), ("PIL",None), ("yaml",None)]
bad = 0
for name, want in need:
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", "?")
        if want and not str(v).startswith(want):
            print(f"  ⚠️  {name:16} {v}  (기대: {want} — 버전 핀 확인)")
        else:
            print(f"  ✅ {name:16} {v}")
    except Exception as e:
        print(f"  ❌ {name:16} import 실패: {e}")
        bad += 1
sys.exit(1 if bad else 0)
EOF
[[ $? -ne 0 ]] && FAIL=$((FAIL+1))

echo
echo "═══ CUDA ═══"
python - <<'EOF' || true
import torch
print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  ✅ {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB)")
else:
    print("  ❌ CUDA를 못 잡았다")
    raise SystemExit(1)
EOF
[[ $? -ne 0 ]] && FAIL=$((FAIL+1))

echo
echo "═══ groundingdino + 가중치 ═══"
W="${GDINO_ROOT}/weights/groundingdino_swint_ogc.pth"
if [[ -f "$W" ]]; then
  SZ=$(stat -c%s "$W" 2>/dev/null || stat -f%z "$W")
  (( SZ > 500000000 )) && ok "가중치 $(du -h "$W"|cut -f1)" || bad "가중치가 너무 작다 — 받다 말았다"
else
  bad "가중치 없음: $W"
fi

# ms_deform_attn이 실제로 로드되는지. 여기서 깨지면 빌드나 LD_LIBRARY_PATH 문제다.
GDINO_ROOT="$GDINO_ROOT" python - <<'EOF' || true
import os
from pathlib import Path
g = Path(os.environ["GDINO_ROOT"])
from groundingdino.util.inference import load_model
class MC:
    config_path  = str(g / "groundingdino/config/GroundingDINO_SwinT_OGC.py")
    weights_path = str(g / "weights/groundingdino_swint_ogc.pth")
    lora_weights = None
    lora_weigths = None
m = load_model(MC(), use_lora=False)
n = sum(p.numel() for p in m.parameters())
print(f"  ✅ 모델 로드 OK — {n/1e6:.1f}M 파라미터")
EOF
[[ $? -ne 0 ]] && { bad "groundingdino 로드 실패 (빌드 또는 LD_LIBRARY_PATH 확인)"; }

echo
echo "═══ dataset2 무결성 ═══"
DATA_ROOT="$DATA_ROOT" python - <<'EOF' || true
import json, os, re, sys
from pathlib import Path
root = Path(os.environ["DATA_ROOT"])
h = lambda n: re.search(r"rf\.([0-9a-f]+)", Path(n).name).group(1)
EXPECT = {"test": (48, 81), "01/train": (74, 100), "01/val": (32, 42),
          "02/train": (70, 111), "02/val": (30, 46)}
bad = 0
sets = {}
for k, (ni, nb) in EXPECT.items():
    p = root / k / "annotations.json"
    if not p.exists():
        print(f"  ❌ 없음: {p}"); bad += 1; continue
    d = json.loads(p.read_text())
    gi, gb = len(d["images"]), len(d["annotations"])
    miss = [i["file_name"] for i in d["images"]
            if not (root / k / "images" / i["file_name"]).exists()]
    pre  = [i["file_name"] for i in d["images"] if re.match(r"^[0-9a-f]{8}-", i["file_name"])]
    sets[k] = {h(i["file_name"]) for i in d["images"]}
    status = "✅" if (gi, gb) == (ni, nb) and not miss and not pre else "❌"
    if status == "❌": bad += 1
    print(f"  {status} {k:9} {gi:3}장 {gb:3}박스 (기대 {ni}/{nb}) | 파일누락 {len(miss)} | LS prefix {len(pre)}")

if len(sets) == len(EXPECT):
    tr = set().union(*(sets[k] for k in EXPECT if k != "test"))
    leak = sets["test"] & tr
    if leak:
        print(f"  ❌ LEAKAGE! test ∩ train/val = {len(leak)}장"); bad += 1
    else:
        print(f"  ✅ test({len(sets['test'])}) ∩ train/val({len(tr)}) = 0")
sys.exit(1 if bad else 0)
EOF
[[ $? -ne 0 ]] && FAIL=$((FAIL+1))

echo
echo "═══ 규칙 7: attention ═══"
ATT="${EVAL_CODE}/common/attention_map.py"
if [[ ! -f "$ATT" ]]; then
  # 파일이 없는데 grep 실패를 '구현됨'으로 읽으면 거짓 양성이 된다. 실제로 그랬다.
  bad "attention_map.py 없음: $ATT"
elif grep -q "^ATTN_MODULE_PATH = None" "$ATT"; then
  warn "attention hook 미구현 (ATTN_MODULE_PATH = None)"
  warn "RULES 7은 '학습 전에 구현'을 요구한다. 지금 zero-shot을 돌리면"
  warn "attention/ 이 비어 verify_run --kind eval 이 실패하고, 나중에 재추론해야 한다."
else
  ok "attention hook 구현됨"
fi

echo
echo "══════════════════════════════════════════════════════════"
if (( FAIL > 0 )); then
  echo " ❌ 실패 ${FAIL}건 — 위를 해결하고 다시 돌릴 것"
  exit 1
fi
echo " ✅ 점검 통과 — zero-shot 실행 가능"
echo "══════════════════════════════════════════════════════════"
