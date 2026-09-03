# =====================================================================
# implantsj 실험 이미지: 폐쇄망 GPU 서버용
# ---------------------------------------------------------------------
# 반입 절차상 이 이미지는 "인터넷은 되지만 데이터는 없는" 바깥에서 빌드되고,
# "데이터는 있지만 인터넷은 없는" 폐쇄망 안에서 실행한다. 따라서 런타임에
# 네트워크로 내려받아야 하는 파일이 남지 않도록 준비한다.
#   - pip 패키지: 빌드할 때 설치
#   - timm pretrained weight: 빌드할 때 미리 받기 (docker/prefetch_weights.py)
#   - 데이터셋과 산출물: 이미지에 넣지 않고 실행할 때 volume mount
#
# 담긴 실험: supervised 분류 / SupCon metric / Mahalanobis unseen /
#            ViT+feature 패치 주입 / 클래스->13feature attribute /
#            YOLO 검출·crop / 얼굴인식(ArcFace 갤러리 방식, src/arcface.py)
#
# 인터넷이 되는 곳에서 빌드(폐쇄망 서버가 x86_64 이므로 플랫폼 고정):
#   docker build --platform linux/amd64 -t implantsj:latest .
# 반입 방법 두 가지 — 둘 중 아무거나:
#   (a) Docker Hub 경유:  docker pull yahoexpert/implantsj:latest
#   (b) 파일 반입:        docker save implantsj:latest | gzip > implantsj.tar.gz
#                         gunzip -c implantsj.tar.gz | docker load
# 폐쇄망 안에서 실행:
#   ./docker/run_container.sh
# =====================================================================

# CUDA 12.1과 cuDNN 런타임을 사용한다. torch cu121 wheel과 맞춘다.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL org.opencontainers.image.title="implantsj" \
      org.opencontainers.image.description="Implant system recognition experiments (ViT/metric learning/ArcFace) for an air-gapped GPU server" \
      org.opencontainers.image.source="https://hub.docker.com/r/yahoexpert/implantsj"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    DEBIAN_FRONTEND=noninteractive \
    # 아래 두 경로에 pretrained weight를 저장한다. prefetch 단계에서 채우고
    # 실행할 때는 이 경로만 읽는다.
    TORCH_HOME=/opt/torch \
    HF_HOME=/opt/hf \
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

# --- 1) torch 설치. CUDA wheel은 별도 index를 쓰므로 레이어를 분리한다. ---
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install \
      --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.4.1 torchvision==0.19.1

# --- 2) 나머지 의존성 설치 ---
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt

# --- 3) pretrained weight를 이미지에 저장한다. 인터넷이 필요하다. ---
# 얼굴인식(ArcFace)도 별도 weight 없이 이 백본들을 그대로 쓴다.
# src/arcface.py 의 기본값 vit_base_patch16_224 는 timm 에서
# vit_base_patch16_224.augreg2_in21k_ft_in1k 로 풀리므로 아래 목록에 이미 있다.
COPY configs/ ./configs/
COPY docker/prefetch_weights.py ./docker/prefetch_weights.py
RUN python docker/prefetch_weights.py

# 오프라인 설정은 prefetch가 끝난 뒤에 적용한다. 먼저 설정하면 빌드 중에도
# 다운로드가 막힌다. 실행할 때 이 값이 남아 있어 폐쇄망에서 네트워크를
# 사용하지 않는다.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# --- 4) 실험 코드. 자주 바뀌므로 마지막에 복사한다. ---
COPY . .
RUN chmod +x run.sh docker/*.sh 2>/dev/null || true

# --- 5) 빌드 타임 자가검증 ---
# 폐쇄망 안에서 "돌려보니 import 부터 안 된다"를 만들지 않기 위해, 데이터 없이
# 확인 가능한 것만 여기서 확인한다. 실패하면 이미지를 만들지 않는다.
#   (a) 얼굴인식: ArcFace margin softmax 가 합성 데이터를 실제로 학습하는지
#   (b) 나머지 실험 진입점이 import 되는지 (오타/순환 import 조기 발견)
RUN python -m src.arcface selftest \
 && python -c "import src.supervised, src.metric_learning, src.evaluation, src.patch_feature, src.feature_attributes; print('[build] entrypoints import OK')" \
 && bash -n run.sh

# 데이터와 산출물을 mount할 위치다. 이미지에는 내용이 없고 실행할 때 연결한다.
VOLUME ["/workspace/data", "/workspace/outputs"]

# tini가 PID 1을 맡아 Ctrl-C와 좀비 프로세스를 처리한다.
# 긴 학습을 중단해도 컨테이너가 남지 않게 한다.
ENTRYPOINT ["/usr/bin/tini", "--"]
# 인자 없이 실행하면 대화형 실험 메뉴를 띄운다. 얼굴인식은 메뉴 17번.
CMD ["./run.sh"]
