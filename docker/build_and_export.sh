#!/usr/bin/env bash
# =====================================================================
# [폐쇄망 반입 전: 인터넷이 되는 곳에서 실행]
# 이미지를 빌드한 뒤 반입용 tar.gz로 저장한다.
#
#   ./docker/build_and_export.sh
#
# 산출물: implantsj_<날짜>.tar.gz. 폐쇄망에는 이 파일만 반입한다.
# =====================================================================
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

IMAGE="${IMAGE:-implantsj}"
TAG="${TAG:-latest}"
OUT="${OUT:-implantsj_$(date +%Y%m%d).tar.gz}"

echo "=============================================================="
# 전송할 build context의 크기를 먼저 확인한다.
echo " 1/3  build context 크기 확인"
echo "=============================================================="
# 전체 저장소와 비교해 실제 전송량을 가늠한다. Docker는 .dockerignore를 따른다.
echo "저장소 전체:      $(du -sh . 2>/dev/null | cut -f1)"
echo "(.dockerignore 화이트리스트로 코드만 전송됩니다)"
echo

echo "=============================================================="
echo " 2/3  docker build  ->  $IMAGE:$TAG"
echo "=============================================================="
# 폐쇄망 서버가 x86_64이므로 플랫폼을 고정한다. Apple Silicon에서도 같은 이미지를 만든다.
docker build --platform linux/amd64 -t "$IMAGE:$TAG" .

echo
echo "=============================================================="
echo " 3/3  docker save  ->  $OUT"
echo "=============================================================="
docker save "$IMAGE:$TAG" | gzip -c > "$OUT"

echo
echo "완료: $OUT  ($(du -sh "$OUT" | cut -f1))"
echo
echo "폐쇄망 반입 후 실행할 명령:"
echo "  gunzip -c $(basename "$OUT") | docker load"
echo "  ./docker/run_container.sh"
