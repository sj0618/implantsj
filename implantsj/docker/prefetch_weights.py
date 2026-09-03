#!/usr/bin/env python3
"""Build-time download of every pretrained backbone used by the image.

The closed network has no internet. A timm or torch weight that is missing from
the image would make training fail when it tries to download the file. This
script runs during ``docker build`` and fills the cache paths used by
``TORCH_HOME`` and ``HF_HOME``. Later, ``pretrained=True`` can load the weights
without network access.

The model list comes from ``configs/backbones.txt``, which is also used by the
``run.sh`` menu. Adding a model there includes it in the next image build.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKBONE_FILE = REPO_ROOT / "configs" / "backbones.txt"
YOLO_FILE = REPO_ROOT / "configs" / "yolo_weights.txt"
YOLO_DIR = REPO_ROOT / "weights"
# ultralytics 는 가중치를 이 릴리스에서 받는다. 런타임에 같은 일을 하려 하므로
# 빌드 타임에 같은 파일을 같은 이름으로 미리 놓아두면 다운로드가 일어나지 않는다.
YOLO_BASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0"


def read_list(path: Path) -> list[str]:
    if not path.is_file():
        print(f"!! {path} not found; nothing to prefetch from it", flush=True)
        return []
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.append(line)
    return names


def read_backbones() -> list[str]:
    return read_list(BACKBONE_FILE)


def prefetch_yolo() -> list[tuple[str, str]]:
    """YOLO 가중치를 weights/ 에 내려놓는다. 실패 목록을 돌려준다.

    timm 과 달리 ultralytics 는 캐시 경로가 아니라 작업 디렉터리 기준으로 파일을
    찾으므로, HF 캐시가 아니라 실제 파일로 놓아야 한다.
    """
    import urllib.request

    names = read_list(YOLO_FILE)
    if not names:
        return []
    YOLO_DIR.mkdir(parents=True, exist_ok=True)
    failed: list[tuple[str, str]] = []
    print(f"prefetching {len(names)} YOLO weights into {YOLO_DIR}", flush=True)
    for name in names:
        dest = YOLO_DIR / name
        url = f"{YOLO_BASE}/{name}"
        try:
            print(f"  -> {name}", flush=True)
            urllib.request.urlretrieve(url, dest)
            size = dest.stat().st_size
            # 릴리스가 사라지면 GitHub 이 HTML 오류 페이지를 200 으로 준다.
            # torch 체크포인트는 zip 이므로 그것으로 판별한다.
            import zipfile
            if not zipfile.is_zipfile(dest):
                raise RuntimeError(f"not a torch checkpoint ({size} bytes) — check {url}")
            print(f"     {size / 1048576:.1f} MB", flush=True)
        except Exception as exc:
            failed.append((name, repr(exc)))
            print(f"  !! FAILED {name}: {exc}", flush=True)
    return failed


def main() -> int:
    import timm

    names = read_backbones()
    if not names:
        return 0

    print(f"prefetching {len(names)} backbones into the image", flush=True)
    failed: list[tuple[str, str]] = []
    for name in names:
        try:
            print(f"  -> {name}", flush=True)
            timm.create_model(name, pretrained=True, num_classes=0)
        except Exception as exc:
            failed.append((name, repr(exc)))
            print(f"  !! FAILED {name}: {exc}", flush=True)
            traceback.print_exc()

    failed += prefetch_yolo()

    if failed:
        # Fail the build loudly: a silently missing weight becomes an
        # un-debuggable training crash inside the air-gapped network.
        print("", flush=True)
        print("Prefetch failed for:", flush=True)
        for name, err in failed:
            print(f"  {name}: {err}", flush=True)
        print(
            "Fix the name in configs/backbones.txt or configs/yolo_weights.txt, or the "
            "build network, then rebuild. Do not ship an image with missing weights.",
            flush=True,
        )
        return 1

    print(f"prefetched {len(names)} backbones", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
