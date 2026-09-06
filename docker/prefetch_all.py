#!/usr/bin/env python3
"""빌드 타임에 세 프로젝트의 pretrained 가중치를 이미지에 굽는다.

폐쇄망에는 인터넷이 없다. 런타임에 내려받아야 하는 가중치가 하나라도 남으면 그
실험은 "돌려보니 안 된다"로 끝나고, 그때는 고칠 방법이 없다. 그래서 빌드가
가중치 하나라도 못 받으면 이미지를 만들지 않고 크게 실패한다.

venv 마다 설치된 패키지가 다르므로 프로젝트별로 따로 호출한다.

    /opt/venv/sj/bin/python prefetch_all.py sj
    /opt/venv/ys/bin/python prefetch_all.py ys
    /opt/venv/ny/bin/python prefetch_all.py ny
"""
from __future__ import annotations

import sys
import traceback
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path("/workspace")

# implantys — Grounding DINO 사전학습 가중치. ENV_SETUP.txt 가 정상 크기를
# "~660MB, 500MB 미만이면 받다 만 것"이라고 적어두었다.
GDINO_URL = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
    "v0.1.0-alpha/groundingdino_swint_ogc.pth"
)
GDINO_MIN_BYTES = 500 * 1024 * 1024

# implant_ny — SAM 2.1 tiny. 저장소에 들어 있던 사본은 정확히 3MiB 에서 잘려
# 있어(zip 헤더만 있고 본문이 없다) 로드가 안 된다. 온전한 것을 새로 받는다.
SAM2_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
)
SAM2_MIN_BYTES = 100 * 1024 * 1024

# implantys metric learning 이 쓰는 DINOv2 백본.
YS_TIMM_MODELS = ["vit_base_patch14_dinov2.lvd142m"]
# Grounding DINO 의 텍스트 인코더.
YS_HF_MODELS = ["bert-base-uncased"]


def read_list(path: Path) -> list[str]:
    if not path.is_file():
        print(f"!! {path} not found; nothing to prefetch from it", flush=True)
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def download(url: str, dest: Path, min_bytes: int) -> None:
    """받아서 크기와 형식을 확인한다.

    릴리스가 옮겨지면 GitHub 이 HTML 오류 페이지를 200 으로 준다. 크기만 보면
    그런 파일도 통과하므로 torch 체크포인트인지(zip 인지)까지 본다.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  -> {dest.name}  ({url})", flush=True)
    urllib.request.urlretrieve(url, dest)
    size = dest.stat().st_size
    if size < min_bytes:
        raise RuntimeError(
            f"{dest} is {size / 1048576:.1f} MB, expected at least "
            f"{min_bytes / 1048576:.0f} MB — download was truncated"
        )
    if not zipfile.is_zipfile(dest):
        raise RuntimeError(f"{dest} is not a torch checkpoint ({size} bytes)")
    print(f"     {size / 1048576:.1f} MB OK", flush=True)


def prefetch_timm(names: list[str]) -> None:
    import timm

    for name in names:
        print(f"  -> timm {name}", flush=True)
        timm.create_model(name, pretrained=True, num_classes=0)


def do_sj() -> None:
    """implantsj: timm 백본 + YOLO 가중치."""
    cfg = ROOT / "implantsj" / "configs"
    backbones = read_list(cfg / "backbones.txt")
    if backbones:
        print(f"[sj] timm 백본 {len(backbones)}개", flush=True)
        prefetch_timm(backbones)

    yolo_names = read_list(cfg / "yolo_weights.txt")
    if yolo_names:
        print(f"[sj] YOLO 가중치 {len(yolo_names)}개", flush=True)
        base = "https://github.com/ultralytics/assets/releases/download/v8.3.0"
        for name in yolo_names:
            download(f"{base}/{name}", ROOT / "implantsj" / "weights" / name, 1 << 20)


def do_ys() -> None:
    """implantys: Grounding DINO 가중치 + BERT 텍스트 인코더 + DINOv2 백본."""
    print("[ys] Grounding DINO 가중치", flush=True)
    download(GDINO_URL, ROOT / "implantys" / "weights" / "groundingdino_swint_ogc.pth",
             GDINO_MIN_BYTES)

    print("[ys] BERT 텍스트 인코더 (HF 캐시)", flush=True)
    from transformers import AutoModel, AutoTokenizer

    for name in YS_HF_MODELS:
        print(f"  -> {name}", flush=True)
        AutoTokenizer.from_pretrained(name)
        AutoModel.from_pretrained(name)

    print("[ys] DINOv2 백본", flush=True)
    prefetch_timm(YS_TIMM_MODELS)


def do_ny() -> None:
    """implant_ny: SAM 2.1 체크포인트."""
    print("[ny] SAM 2.1 tiny", flush=True)
    download(SAM2_URL, ROOT / "implant_ny" / "checkpoints" / "sam2.1_hiera_tiny.pt",
             SAM2_MIN_BYTES)


TASKS = {"sj": do_sj, "ys": do_ys, "ny": do_ny}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in TASKS:
        print(f"usage: prefetch_all.py {{{'|'.join(TASKS)}}}", flush=True)
        return 2
    which = sys.argv[1]
    try:
        TASKS[which]()
    except Exception as exc:
        print(f"\n!! prefetch FAILED for {which}: {exc}", flush=True)
        traceback.print_exc()
        print(
            "\n가중치가 빠진 이미지는 반입하지 않는다. 폐쇄망 안에서는 이걸 고칠 수 "
            "없다. URL 이나 빌드 네트워크를 확인하고 다시 빌드한다.",
            flush=True,
        )
        return 1
    print(f"[{which}] prefetch done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
