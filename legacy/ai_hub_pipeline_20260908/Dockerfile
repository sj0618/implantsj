# =====================================================================
# ai_hub_pipeline 실험 이미지: 폐쇄망 GPU 서버용
# ---------------------------------------------------------------------
# "인터넷은 되지만 데이터는 없는" 바깥에서 빌드하고, "데이터는 있지만 인터넷은 없는"
# 폐쇄망 안에서 실행한다. 런타임에 네트워크로 받아야 하는 것이 남지 않도록:
#   - pip 패키지 (sam2, segment_anything 포함) : 빌드 때 설치
#   - timm pretrained 가중치                     : 빌드 때 캐시에 저장 (docker/prefetch_weights.py)
#   - SAM1 vit_h / SAM2 hiera_tiny               : 빌드 때 /opt/weights 에 다운로드
#   - 파인튜닝 adapter (.pth, 팀 자산)           : weights/ 에 두면 /opt/weights 로 복사
#   - 데이터셋·산출물·체크포인트                 : 이미지에 넣지 않고 실행 때 volume mount
#
# 빌드 (x86_64 폐쇄망 서버 기준):
#   ./docker/build_and_export.sh          # = docker build --platform linux/amd64 + docker save
# 실행:
#   WORK=/workspace/ai_hub ./docker/run_container.sh bash scripts/01_backbone/finetune_backbones.sh
# =====================================================================

# SAM2 가 torch>=2.5.1 을 요구하므로 CUDA 12.4 런타임과 torch cu124 wheel 을 맞춘다.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

LABEL org.opencontainers.image.title="ai_hub_pipeline" \
      org.opencontainers.image.description="Implant identification experiment pipeline (backbone / metric / band-vit / masked-vit / attributes / fuse) for an air-gapped GPU server"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    DEBIAN_FRONTEND=noninteractive \
    # timm/HF 가중치 캐시. prefetch 가 채우고 실행 때는 읽기만 한다.
    TORCH_HOME=/opt/torch \
    HF_HOME=/opt/hf \
    # SAM/adapter 가중치. entrypoint 가 checkpoints/ 로 링크한다.
    AI_HUB_WEIGHTS=/opt/weights \
    AI_HUB_ROOT=/workspace/ai_hub_pipeline \
    # sam2 의 CUDA 확장은 비디오 후처리용이라 빌드하지 않는다 (컴파일러 불필요).
    SAM2_BUILD_CUDA=0 \
    SAM2_BUILD_ALLOW_ERRORS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        git ca-certificates tini \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3

WORKDIR /workspace/ai_hub_pipeline

# --- 1) torch. CUDA wheel 은 별도 index 라 레이어를 분리한다. ---
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install \
      --index-url https://download.pytorch.org/whl/cu124 \
      torch==2.5.1 torchvision==0.20.1

# --- 2) 나머지 의존성. sam2 는 pyproject 가 빌드 의존성으로 torch 를 요구하므로
#        그 두 git 패키지만 build isolation 을 끄고 설치한다 (나머지는 정상 wheel). ---
COPY requirements.txt requirements-seg.txt ./
RUN python -m pip install -r requirements.txt \
 && python -m pip install --no-build-isolation -r requirements-seg.txt

# --- 3) 가중치를 이미지에 저장한다. 인터넷이 필요하다. ---
# PREFETCH_SAM1=0 이면 2.5GB 짜리 SAM1 을 생략한다 (adapter 가 이미 있어 Stage 0 ①②를 안 돌릴 때).
# weights/ (팀 adapter .pth) 는 /opt/weights 로만 들어간다 — 아래 코드 COPY 목록에 없으므로 두 번 굽히지 않는다.
ARG PREFETCH_SAM1=1
COPY configs/ ./configs/
COPY docker/prefetch_weights.py ./docker/prefetch_weights.py
COPY weights/ /opt/weights/
RUN python docker/prefetch_weights.py --sam1 "${PREFETCH_SAM1}"

# 오프라인 설정은 prefetch 뒤에 건다. 먼저 걸면 빌드 중에도 다운로드가 막힌다.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# --- 4) 코드. 자주 바뀌므로 마지막에, 폴더를 명시해서 복사한다 (weights/ 제외).
#        새 최상위 폴더를 만들면 여기에 한 줄 추가할 것. ---
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY docker/ ./docker/
COPY tests/ ./tests/
COPY docs/ ./docs/
# verify_pipeline 의 의존성 선언 검사가 Dockerfile 을 읽어 torch/torchvision 을 인정한다 → 같이 넣는다.
COPY Dockerfile ./
RUN chmod +x docker/*.sh scripts/*.sh scripts/*/*.sh 2>/dev/null || true

# --- 5) 빌드 타임 자가검증: 폐쇄망 안에서 "import 부터 안 된다"를 만들지 않는다. ---
#   (a) src/data 패키지 존재 — 루트 .gitignore 의 data/ 규칙에 삼켜져 저장소에 빠진 적이 있다.
#       없으면 run_prepare_data.py(파이프라인 입구)가 폐쇄망 안에서 ImportError 로 죽으므로 빌드에서 막는다.
#   (b) 모든 런타임 패키지 import
#   (c) timm 가중치가 오프라인 캐시에서 풀리는지 + SAM 파일 존재
#   (d) 설정·결합·데이터 로직 점검 (verify_pipeline 전체) + 임베딩/attributes 스모크 테스트
RUN test -f src/data/build.py \
 || (echo "!! src/data/ 가 없습니다. 원 작성자 로컬에서: git add -f ai_hub_pipeline/src/data && commit" && false)
RUN python -c "import torch, torchvision, timm, sam2, segment_anything, cv2, sklearn, scipy, umap, yaml, hydra, iopath, PIL, matplotlib, pandas, numpy, safetensors; print('[build] imports OK', torch.__version__, timm.__version__)" \
 && python docker/prefetch_weights.py --verify-offline \
 && python src/verify_pipeline.py \
 && python tests/test_smoke.py

# 데이터와 산출물을 mount 할 위치. 이미지에는 내용이 없고 실행할 때 연결한다.
VOLUME ["/workspace/ai_hub_pipeline/data", "/workspace/ai_hub_pipeline/outputs", \
        "/workspace/ai_hub_pipeline/checkpoints", "/workspace/ai_hub_pipeline/seg_data"]

# tini 가 PID 1 을 맡아 Ctrl-C 와 좀비 프로세스를 처리한다. entrypoint 는 가중치 링크만 한다.
ENTRYPOINT ["/usr/bin/tini", "--", "/workspace/ai_hub_pipeline/docker/entrypoint.sh"]
CMD ["bash"]
