#!/usr/bin/env bash
# =====================================================================
# [x86_64 리눅스 호스트에서 실행]
# implantsj + implantys + implant_ny 통합 이미지를 빌드한다.
#
#   ./docker/build_combined.sh
#   PUSH=1 HUB_REPO=yahoexpert/implantid ./docker/build_combined.sh
#
# ARM 맥에서는 돌리지 않는다. implantys 의 Grounding DINO 는 nvcc 로 CUDA 확장을
# 컴파일하는데, amd64 에뮬레이션에서 성공한다는 보장이 없다. 네이티브 x86 에서
# 빌드하는 것이 이 이미지의 전제다.
# =====================================================================
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

IMAGE="${IMAGE:-implantid}"
TAG="${TAG:-latest}"

arch="$(uname -m)"
if [[ "$arch" != "x86_64" ]]; then
  echo "!! 이 호스트는 $arch 다. 이 이미지는 x86_64 에서 빌드해야 한다." >&2
  echo "   Grounding DINO 의 CUDA 확장 컴파일이 에뮬레이션에서 깨질 수 있다." >&2
  echo "   그래도 진행하려면 FORCE_ARCH=1 을 준다." >&2
  [[ "${FORCE_ARCH:-0}" == "1" ]] || exit 1
fi

echo "=============================================================="
echo " 1/3  build context 크기"
echo "=============================================================="
echo "저장소 전체: $(du -sh . 2>/dev/null | cut -f1)  (.dockerignore 화이트리스트로 코드만 전송됩니다)"
echo

echo "=============================================================="
echo " 2/3  docker build -> $IMAGE:$TAG"
echo "=============================================================="
# 빌드 중 pretrained 가중치를 받으므로 인터넷이 필요하다.
# 세 venv + gdino CUDA 확장 컴파일이라 시간이 오래 걸린다.
docker build --progress=plain -t "$IMAGE:$TAG" .

echo
echo "=============================================================="
echo " 3/3  자가진단"
echo "=============================================================="
docker run --rm "$IMAGE:$TAG" smoke

if [[ "${PUSH:-0}" == "1" ]]; then
  HUB_REPO="${HUB_REPO:?PUSH=1 이면 HUB_REPO 를 준다}"
  docker tag "$IMAGE:$TAG" "$HUB_REPO:$TAG"
  docker push "$HUB_REPO:$TAG"
  echo "받는 쪽: docker pull $HUB_REPO:$TAG"
fi

echo
echo "완료. 폐쇄망 반입용 파일이 필요하면:"
echo "  docker save $IMAGE:$TAG | gzip -c > implantid_\$(date +%Y%m%d).tar.gz"
