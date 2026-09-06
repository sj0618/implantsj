#!/usr/bin/env bash
# =====================================================================
# 통합 이미지 진입점. 프로젝트마다 파이썬 환경이 다르므로 여기서 갈라준다.
#
#   docker run ... implantid:latest menu          implantsj 대화형 메뉴 (기본)
#   docker run ... implantid:latest sj <args>     implantsj  venv 에서 실행
#   docker run ... implantid:latest ys <args>     implantys  venv 에서 실행
#   docker run ... implantid:latest ny <args>     implant_ny venv 에서 실행
#   docker run ... implantid:latest bash          셸 (venv 미활성)
#
# 인자 없이 프로젝트만 주면 그 venv 의 python 으로 대화형 셸이 열린다.
# =====================================================================
set -uo pipefail

ROOT=/workspace

usage() {
  cat <<'EOF'
ImplantID 통합 이미지

  menu              implantsj 실험 메뉴 (기본값)
  sj  [명령...]     implantsj  — ArcFace / ViT / metric learning
  ys  [명령...]     implantys  — Grounding DINO / DINOv2 metric learning
  ny  [명령...]     implant_ny — band-guided ViT / SAM2
  bash              셸
  smoke             데이터 없이 도는 자가진단 전체

프로젝트마다 파이썬 환경이 다르다. 핀이 서로 배타적이라 합칠 수 없어서다.
  sj : python3.10 + torch 2.4.1 cu121
  ys : python3.11 + torch 2.4.1 cu124 + Grounding DINO (CUDA 확장)
  ny : python3.11 + torch 2.5.1 cu124 + SAM2

각 venv 의 python 은 /opt/venv/{sj,ys,ny}/bin/python 이다.
문서는 implantsj/README.md, implantys/ENV_SETUP.txt, implant_ny/*/README.md 에 있다.
EOF
}

# ys 는 ms_deform_attn CUDA 확장이 libc10.so 를 찾아야 하므로 torch/lib 를 걸어준다.
ys_env() {
  local torch_lib
  torch_lib="$(/opt/venv/ys/bin/python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))' 2>/dev/null)"
  export LD_LIBRARY_PATH="${torch_lib}:${LD_LIBRARY_PATH:-}"
}

cmd="${1:-menu}"
shift || true

case "$cmd" in
  menu)
    cd "$ROOT/implantsj" && exec env PYTHON=/opt/venv/sj/bin/python ./run.sh "$@"
    ;;
  sj)
    cd "$ROOT/implantsj"
    if [[ $# -eq 0 ]]; then exec /opt/venv/sj/bin/python; fi
    # run.sh 로 시작하는 인자는 메뉴 CLI, 그 외는 파이썬으로 넘긴다.
    if [[ "${1:-}" == "run.sh" || "${1:-}" == "./run.sh" ]]; then
      shift; exec env PYTHON=/opt/venv/sj/bin/python ./run.sh "$@"
    fi
    exec env PYTHON=/opt/venv/sj/bin/python /opt/venv/sj/bin/python "$@"
    ;;
  ys)
    ys_env
    cd "$ROOT/implantys"
    if [[ $# -eq 0 ]]; then exec /opt/venv/ys/bin/python; fi
    exec /opt/venv/ys/bin/python "$@"
    ;;
  ny)
    cd "$ROOT/implant_ny"
    if [[ $# -eq 0 ]]; then exec /opt/venv/ny/bin/python; fi
    exec /opt/venv/ny/bin/python "$@"
    ;;
  smoke)
    exec "$ROOT/docker/smoke_all.sh" "$@"
    ;;
  bash|sh)
    exec /bin/bash "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "알 수 없는 명령: $cmd" >&2
    usage
    exit 2
    ;;
esac
