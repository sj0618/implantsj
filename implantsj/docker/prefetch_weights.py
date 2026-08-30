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


def read_backbones() -> list[str]:
    if not BACKBONE_FILE.is_file():
        print(f"!! {BACKBONE_FILE} not found; nothing to prefetch", flush=True)
        return []
    names = []
    for line in BACKBONE_FILE.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.append(line)
    return names


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

    if failed:
        # Fail the build loudly: a silently missing weight becomes an
        # un-debuggable training crash inside the air-gapped network.
        print("", flush=True)
        print("Prefetch failed for:", flush=True)
        for name, err in failed:
            print(f"  {name}: {err}", flush=True)
        print(
            "Fix the model name in configs/backbones.txt or the build network, "
            "then rebuild. Do not ship an image with missing weights.",
            flush=True,
        )
        return 1

    print(f"prefetched {len(names)} backbones", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
