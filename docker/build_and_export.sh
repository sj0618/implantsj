#!/usr/bin/env bash
# =====================================================================
# [폐쇄 전 · 인터넷 있는 곳에서 실행]
# 이미지를 빌드하고 반입용 tar.gz 로 내보낸다.
#
#   ./docker/build_and_export.sh
#
# 산출물: implantsj_<날짜>.tar.gz  (이 파일만 폐쇄망으로 반입하면 된다)
# =====================================================================
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

IMAGE="${IMAGE:-implantsj}"
TAG="${TAG:-latest}"
OUT="${OUT:-implantsj_$(date +%Y%m%d).tar.gz}"

echo "=============================================================="
echo " 1/3  build context 크기 확인"
echo "=============================================================="
# .dockerignore 를 무시한 전체 크기와 대비해 실제로 보낼 양을 가늠한다.
echo "저장소 전체:      $(du -sh . 2>/dev/null | cut -f1)"
echo "(.dockerignore 화이트리스트로 코드만 전송됩니다)"
echo

echo "=============================================================="
echo " 2/3  docker build  ->  $IMAGE:$TAG"
echo "=============================================================="
# 폐쇄망 서버는 x86_64 이므로 플랫폼을 고정한다(Apple Silicon 에서 빌드해도 안전).
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
