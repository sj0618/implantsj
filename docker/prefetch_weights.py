#!/usr/bin/env python3
"""Download every pretrained backbone into the image at build time.

The closed network has no internet, so any timm/torch weight that is not
already inside the image is unreachable at run time — training would die
partway through with a download error. This runs during `docker build`
(while the network is still up) and populates the torch/timm cache that
`TORCH_HOME`/`HF_HOME` point at, so `pretrained=True` resolves offline later.

The backbone list comes from configs/backbones.txt, the same file the run.sh
menu reads, so adding a model there is enough to get it baked in.
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
