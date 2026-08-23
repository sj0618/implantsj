# =====================================================================
# implantsj 실험 이미지 (폐쇄망 GPU 서버용)
# ---------------------------------------------------------------------
# 반입 절차상 이 이미지는 "인터넷은 되지만 데이터는 없는" 바깥에서 빌드되고,
# "데이터는 있지만 인터넷은 없는" 폐쇄망 안에서 실행된다. 따라서 런타임에
# 네트워크로 가져와야 하는 것이 하나도 남아 있으면 안 된다:
#   - pip 패키지: 빌드 타임에 설치
#   - timm pretrained weight: 빌드 타임에 prefetch (docker/prefetch_weights.py)
#   - 데이터셋/산출물: 이미지에 넣지 않고 런타임 volume mount
#
# 빌드 (인터넷 되는 곳):
#   docker build -t implantsj:latest .
#   docker save implantsj:latest | gzip > implantsj.tar.gz
# 반입 후 (폐쇄망):
#   gunzip -c implantsj.tar.gz | docker load
#   ./docker/run_container.sh
# =====================================================================

# CUDA 12.1 + cuDNN 런타임. torch cu121 휠과 맞춘다.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    DEBIAN_FRONTEND=noninteractive \
    # 아래 두 경로가 pretrained weight 캐시 위치다. prefetch 가 여기에 채우고
    # 런타임에는 여기서만 읽는다.
    TORCH_HOME=/opt/torch \
    HF_HOME=/opt/hf \
    # 폐쇄망에서 실수로 네트워크를 두드리지 않도록 오프라인 모드를 고정한다.
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=/workspace/implantsj

# OpenCV(ultralytics 의존)와 이미지 디코딩에 필요한 최소 시스템 패키지.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        git ca-certificates tini \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3

WORKDIR /workspace/implantsj

# --- 1) torch 먼저 (CUDA 휠은 별도 인덱스라 레이어를 분리해 캐시를 살린다) ---
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install \
      --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.4.1 torchvision==0.19.1

# --- 2) 나머지 의존성 ---
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt

# --- 3) pretrained weight 를 이미지에 굽는다 (인터넷 있을 때만 가능) ---
COPY configs/ ./configs/
COPY docker/prefetch_weights.py ./docker/prefetch_weights.py
RUN python docker/prefetch_weights.py

# --- 4) 실험 코드 (가장 자주 바뀌므로 마지막) ---
COPY . .
RUN chmod +x run.sh docker/*.sh 2>/dev/null || true

# 데이터/산출물 mount 지점. 이미지에는 비어 있고 런타임에 채워진다.
VOLUME ["/workspace/data", "/workspace/outputs"]

# tini 가 PID 1 을 잡아 Ctrl-C 와 좀비 프로세스를 정리한다.
# (긴 학습을 중단할 때 컨테이너가 안 죽는 문제를 막는다)
ENTRYPOINT ["/usr/bin/tini", "--"]
# 인자 없이 실행하면 대화형 실험 메뉴가 뜬다.
CMD ["./run.sh"]
