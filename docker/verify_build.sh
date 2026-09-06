#!/usr/bin/env bash
# =====================================================================
# 빌드 타임 자가검증. 데이터 없이 확인 가능한 것만 본다.
# 하나라도 실패하면 이미지를 만들지 않는다 — 가중치나 확장이 빠진 이미지를
# 폐쇄망에 반입하면 안에서는 고칠 수 없기 때문이다.
# =====================================================================
set -euo pipefail

ROOT=/workspace
fail=0
step() {
  printf '\n--- %s ---\n' "$1"
}

step "sj: 패키지 import + ArcFace 학습 가능성"
cd "$ROOT/implantsj"
/opt/venv/sj/bin/python -c "
import torch, timm, ultralytics
print('torch', torch.__version__, '| timm', timm.__version__, '| ultralytics', ultralytics.__version__)
import src.supervised, src.metric_learning, src.evaluation, src.feature_attributes
import src.arcface, src.arcface_attributes
print('implantsj entrypoints import OK')
"
/opt/venv/sj/bin/python -m src.arcface selftest
/opt/venv/sj/bin/python -m src.arcface_attributes selftest
/opt/venv/sj/bin/python - <<'PY'
from pathlib import Path
import zipfile
names = [l.split('#',1)[0].strip()
         for l in Path('configs/yolo_weights.txt').read_text(encoding='utf-8').splitlines()]
for n in [x for x in names if x]:
    p = Path('weights')/n
    assert p.is_file() and zipfile.is_zipfile(p), f'YOLO 가중치 없음/손상: {p}'
    print(f'  {n}  {p.stat().st_size/1048576:.1f} MB')
PY
bash -n run.sh

step "ys: Grounding DINO CUDA 확장 + 가중치"
cd "$ROOT/implantys"
export LD_LIBRARY_PATH="$(/opt/venv/ys/bin/python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH:-}"
/opt/venv/ys/bin/python -c "
import torch, transformers, timm
print('torch', torch.__version__, '| transformers', transformers.__version__)
import numpy; assert numpy.__version__.startswith('1.26'), numpy.__version__
import cv2; print('opencv', cv2.__version__)
"
# 여기가 이 이미지에서 가장 잘 깨지는 지점이다. ms_deform_attn 은 소스에서
# 컴파일한 CUDA 확장이라, 빌드가 조용히 실패하면 import 할 때 드러난다.
/opt/venv/ys/bin/python -c "
from groundingdino.util.inference import load_model
import groundingdino
print('groundingdino import OK:', groundingdino.__file__)
from groundingdino import _C
print('ms_deform_attn CUDA 확장 로드 OK')
"
/opt/venv/ys/bin/python - <<'PY'
from pathlib import Path
p = Path('weights/groundingdino_swint_ogc.pth')
assert p.is_file(), f'없음: {p}'
mb = p.stat().st_size/1048576
assert mb > 500, f'받다 만 파일: {mb:.1f} MB'
print(f'  groundingdino_swint_ogc.pth  {mb:.1f} MB')
PY

step "ny: SAM2 + band_vit"
cd "$ROOT/implant_ny"
/opt/venv/ny/bin/python -c "
import torch, numpy, cv2
print('torch', torch.__version__, '| numpy', numpy.__version__, '| opencv', cv2.__version__)
assert numpy.__version__.startswith('2.'), numpy.__version__
import sam2, segment_anything
print('sam2 / segment_anything import OK')
"
/opt/venv/ny/bin/python - <<'PY'
from pathlib import Path
import zipfile
p = Path('checkpoints/sam2.1_hiera_tiny.pt')
assert p.is_file(), f'없음: {p}'
mb = p.stat().st_size/1048576
assert mb > 100, f'받다 만 파일: {mb:.1f} MB (저장소 사본은 3MiB 로 잘려 있었다)'
assert zipfile.is_zipfile(p), 'torch 체크포인트가 아니다'
print(f'  sam2.1_hiera_tiny.pt  {mb:.1f} MB')
PY

printf '\n=== 빌드 자가검증 전부 통과 ===\n'
