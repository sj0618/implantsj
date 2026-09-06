# =====================================================================
# ImplantID 통합 이미지: implantsj + implantys + implant_ny
# ---------------------------------------------------------------------
# 세 프로젝트를 한 이미지에 담되, 파이썬 환경은 셋으로 나눈다. 합칠 수 없기
# 때문이다. 각 팀이 문서에 이유까지 적어 고정해 둔 핀들이 서로 배타적이다.
#
#                implantsj        implantys          implant_ny
#   torch        2.4.1 cu121      2.4.1 cu124        2.5.1 cu124
#   numpy        >=1.24,<2.0      1.26.4             2.4.4
#   opencv       (ultralytics)    headless           opencv-python 5.0
#
#   - numpy 2.x 는 implantys 에서 금지다: pycocotools ABI 에러(setup_pod.sh 기록).
#   - torch 2.6+ 는 implantys 의 ms_deform_attn CUDA 확장 컴파일을 깬다.
#   - implantys 는 opencv GUI 빌드와 headless 의 동시 설치를 금지한다.
#
# 그래서 /opt/venv/{sj,ys,ny} 세 개의 venv 를 만들고 어느 핀도 건드리지 않는다.
# 코드와 가중치는 한 이미지에 있고, 실행할 때 해당 venv 로 들어간다.
#
# 베이스가 runtime 이 아니라 devel 인 이유는 nvcc 다. implantys 의 Grounding DINO
# 는 ms_deform_attn CUDA 확장을 소스에서 빌드해야 하고, nvcc 가 없으면 bounding
# box 실험이 통째로 막힌다.
#
# 반입 절차: 인터넷은 되지만 데이터가 없는 곳에서 빌드하고, 데이터는 있지만
# 인터넷이 없는 폐쇄망에서 실행한다. 따라서 런타임에 내려받아야 하는 파일이
# 남으면 안 된다. 모든 의존성과 pretrained 가중치를 빌드 타임에 굽는다.
#
# 빌드(x86_64 호스트에서):
#   ./docker/build_combined.sh
# 검증:
#   docker run --rm implantid:latest /workspace/docker/smoke_all.sh
# =====================================================================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

LABEL org.opencontainers.image.title="ImplantID" \
      org.opencontainers.image.description="Combined implant experiments: implantsj (ArcFace/ViT/metric), implantys (Grounding DINO/DINOv2), implant_ny (band-guided ViT/SAM2)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    DEBIAN_FRONTEND=noninteractive \
    TOKENIZERS_PARALLELISM=false \
    # pretrained 가중치 캐시. 세 venv 가 공유한다.
    TORCH_HOME=/opt/torch \
    HF_HOME=/opt/hf \
    # GPU 없는 빌드 머신에서 CUDA 확장을 컴파일하려면 대상 아키텍처를 직접 준다.
    # 빌드 호스트에 GPU 가 있어도 폐쇄망 서버의 GPU 와 다를 수 있으므로 넓게 잡는다.
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"

# --- 시스템 패키지 ---------------------------------------------------
# python3.10 은 우분투 기본. implantsj 가 그 위에서 검증되어 있어 그대로 쓴다.
# python3.11 은 implantys 가 문서에서 요구하는 버전이고 implant_ny 도 여기에 얹는다.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates gnupg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-venv \
        python3.11 python3.11-dev python3.11-venv \
        python3-pip \
        git curl wget tini ninja-build build-essential \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# =====================================================================
# venv 1/3 — implantsj (python3.10, torch cu121)
# 이 조합이 smoke test 10/10 으로 검증된 상태다. 그대로 둔다.
# =====================================================================
RUN python3.10 -m venv /opt/venv/sj \
 && /opt/venv/sj/bin/pip install --upgrade pip setuptools wheel \
 && /opt/venv/sj/bin/pip install \
      --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.4.1 torchvision==0.19.1
COPY implantsj/requirements.txt /tmp/req.sj.txt
RUN /opt/venv/sj/bin/pip install -r /tmp/req.sj.txt

# =====================================================================
# venv 2/3 — implantys (python3.11, torch cu124, Grounding DINO)
# =====================================================================
RUN python3.11 -m venv /opt/venv/ys \
 && /opt/venv/ys/bin/pip install --upgrade pip setuptools wheel \
 && /opt/venv/ys/bin/pip install \
      --index-url https://download.pytorch.org/whl/cu124 \
      torch==2.4.1 torchvision==0.19.1
COPY implantys/requirements.txt /tmp/req.ys.txt
RUN /opt/venv/ys/bin/pip install -r /tmp/req.ys.txt

# Grounding DINO 는 pip 패키지가 없어 저장소를 받아 소스에서 빌드한다.
# --no-build-isolation 이 필수다. 격리 빌드를 하면 pip 이 자기 torch 를 새로
# 받아와 위에서 고정한 2.4.1 핀이 무시된다(ENV_SETUP.txt 기록).
# LD_LIBRARY_PATH 를 torch/lib 로 잡지 않으면 ms_deform_attn import 시
# libc10.so 를 못 찾고 깨진다.
ENV GDINO_REPO=/opt/Grounding-Dino-FineTuning
RUN git clone https://github.com/Asad-Ismail/Grounding-Dino-FineTuning "$GDINO_REPO" \
 && cd "$GDINO_REPO" \
 && git checkout d241d21801cb2b9a80c75f119e656fcf95750392 \
 && LD_LIBRARY_PATH="$(/opt/venv/ys/bin/python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):$LD_LIBRARY_PATH" \
    /opt/venv/ys/bin/pip install -e . --no-build-isolation

# =====================================================================
# venv 3/3 — implant_ny (python3.11, torch 2.5.1 cu124, SAM2)
# =====================================================================
RUN python3.11 -m venv /opt/venv/ny \
 && /opt/venv/ny/bin/pip install --upgrade pip setuptools wheel
COPY implant_ny/requirements-sam2.txt /tmp/req.ny.txt
RUN /opt/venv/ny/bin/pip install \
      --extra-index-url https://download.pytorch.org/whl/cu124 \
      -r /tmp/req.ny.txt

# =====================================================================
# pretrained 가중치 — 전부 빌드 타임에 굽는다. 여기가 인터넷을 쓰는 마지막 지점.
# =====================================================================
COPY implantsj/configs/ /workspace/implantsj/configs/
COPY docker/ /workspace/docker/
RUN /opt/venv/sj/bin/python /workspace/docker/prefetch_all.py sj \
 && /opt/venv/ys/bin/python /workspace/docker/prefetch_all.py ys \
 && /opt/venv/ny/bin/python /workspace/docker/prefetch_all.py ny

# 오프라인 설정은 prefetch 가 끝난 뒤에 건다. 먼저 걸면 빌드 중 다운로드가 막힌다.
# 실행할 때 이 값이 남아 있어 폐쇄망에서 네트워크를 쓰지 않는다.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# =====================================================================
# 코드 — 자주 바뀌므로 마지막에 복사한다.
# =====================================================================
COPY . /workspace/
RUN chmod +x /workspace/docker/*.sh /workspace/implantsj/run.sh \
             /workspace/implantsj/docker/*.sh 2>/dev/null || true

# --- 빌드 타임 자가검증 ----------------------------------------------
# 폐쇄망 안에서 "돌려보니 import 부터 안 된다"를 만들지 않기 위해, 데이터 없이
# 확인 가능한 것만 여기서 본다. 실패하면 이미지를 만들지 않는다.
RUN /workspace/docker/verify_build.sh

VOLUME ["/workspace/data", "/workspace/outputs"]

# tini 가 PID 1 을 맡아 Ctrl-C 와 좀비 프로세스를 처리한다.
ENTRYPOINT ["/usr/bin/tini", "--", "/workspace/docker/entrypoint.sh"]
CMD ["menu"]
