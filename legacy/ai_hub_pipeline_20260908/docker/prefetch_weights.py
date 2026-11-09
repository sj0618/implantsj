#!/usr/bin/env python3
"""빌드 타임 가중치 반입 — 폐쇄망에서 네트워크로 받아야 할 파일이 남지 않게 한다.

굽는 것:
  1) timm pretrained  : configs/base.yaml + configs/05_attributes.yaml 의 backbones.*.timm 전부
                        → TORCH_HOME / HF_HOME 캐시 (실행 때 pretrained=True 가 오프라인으로 풀린다)
  2) SAM1 vit_h       : Stage 0 ② 초기 마스크 생성용 (2.5 GB, --sam1 0 으로 생략 가능)
  3) SAM2 hiera_tiny  : Stage 0 ⑤ 학습·⑥ 예측, masked-vit 런타임 seg 의 base
  4) 파인튜닝 adapter : weights/best_sam2.1_tiny_adapter_unet.pth 가 빌드 컨텍스트에 있으면 복사
                        (공개 URL 이 없는 팀 자산이라 없으면 경고만 하고 계속)

2)~4) 는 /opt/weights/ 에 두고, docker/entrypoint.sh 가 컨테이너 시작 때
checkpoints/ 아래 yaml 이 기대하는 경로로 symlink 한다 (checkpoints/ 가 volume 이어도 안전).

    python docker/prefetch_weights.py [--sam1 0|1]      # 빌드 중 (온라인)
    python docker/prefetch_weights.py --verify-offline  # HF_HUB_OFFLINE=1 상태에서 재검증
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import traceback
import urllib.request
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [ROOT / "configs" / "base.yaml", ROOT / "configs" / "05_attributes.yaml"]
WEIGHTS_ROOT = Path(os.environ.get("AI_HUB_WEIGHTS", "/opt/weights"))
STAGED_ADAPTER = ROOT / "weights" / "best_sam2.1_tiny_adapter_unet.pth"

# (checkpoints/ 아래 상대경로 == yaml 이 기대하는 경로, 다운로드 URL)
SAM1 = ("sam/sam_vit_h_4b8939.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
SAM2 = ("sam2/sam2.1_hiera_tiny.pt",
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt")
ADAPTER = "best_sam2.1_tiny_adapter_unet.pth"


def timm_names() -> list[str]:
    names: list[str] = []
    for path in CONFIGS:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for entry in (raw.get("backbones") or {}).values():
            name = (entry or {}).get("timm")
            if name and name not in names:
                names.append(str(name))
    return names


def prefetch_timm(names: list[str]) -> list[tuple[str, str]]:
    import timm

    failed = []
    print(f"prefetching {len(names)} timm backbones", flush=True)
    for name in names:
        try:
            print(f"  -> {name}", flush=True)
            timm.create_model(name, pretrained=True, num_classes=0)
        except Exception as exc:
            failed.append((name, repr(exc)))
            print(f"  !! FAILED {name}: {exc}", flush=True)
            traceback.print_exc()
    return failed


def download(relative: str, url: str) -> list[tuple[str, str]]:
    dest = WEIGHTS_ROOT / relative
    if dest.is_file() and zipfile.is_zipfile(dest):
        print(f"  = {relative} 이미 있음 ({dest.stat().st_size / 1048576:.1f} MB)", flush=True)
        return []
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"  -> {relative}  ({url})", flush=True)
        urllib.request.urlretrieve(url, dest)
        size = dest.stat().st_size
        # 릴리스가 사라지면 HTML 오류 페이지가 200 으로 온다. torch 체크포인트는 zip 이다.
        if not zipfile.is_zipfile(dest):
            raise RuntimeError(f"not a torch checkpoint ({size} bytes) — check {url}")
        print(f"     {size / 1048576:.1f} MB", flush=True)
        return []
    except Exception as exc:
        print(f"  !! FAILED {relative}: {exc}", flush=True)
        return [(relative, repr(exc))]


def stage_adapter() -> None:
    dest = WEIGHTS_ROOT / ADAPTER
    if dest.is_file():
        print(f"  = {ADAPTER} 이미 있음 ({dest.stat().st_size / 1048576:.1f} MB)", flush=True)
        return
    if STAGED_ADAPTER.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(STAGED_ADAPTER, dest)
        print(f"  -> {ADAPTER} 복사 ({dest.stat().st_size / 1048576:.1f} MB)", flush=True)
        return
    print(
        f"  !! WARNING {ADAPTER} 없음 — weights/ 에 넣고 빌드하면 이미지에 굽는다.\n"
        f"     없는 채로 반입하면 폐쇄망 안에서 Stage 0 ①~⑤ 로 먼저 만들어야 "
        f"masked-vit 과 band mask 생성이 된다.",
        flush=True,
    )


def verify_offline(names: list[str]) -> int:
    """HF_HUB_OFFLINE=1 로 timm 가중치가 캐시에서 풀리는지, SAM 파일이 있는지 확인."""
    import timm

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    failed = []
    for name in names:
        try:
            timm.create_model(name, pretrained=True, num_classes=0)
            print(f"  offline OK  {name}", flush=True)
        except Exception as exc:
            failed.append((name, repr(exc)))
            print(f"  !! offline FAIL {name}: {exc}", flush=True)
    for relative in (SAM2[0],):
        path = WEIGHTS_ROOT / relative
        ok = path.is_file() and zipfile.is_zipfile(path)
        print(f"  {'offline OK ' if ok else '!! MISSING '} {relative}", flush=True)
        if not ok:
            failed.append((relative, "missing"))
    sam1 = WEIGHTS_ROOT / SAM1[0]
    print(f"  {'offline OK ' if sam1.is_file() else '   (생략)   '} {SAM1[0]}", flush=True)
    adapter = WEIGHTS_ROOT / ADAPTER
    print(f"  {'offline OK ' if adapter.is_file() else '!! WARNING 없음'} {ADAPTER}", flush=True)
    if failed:
        print("\noffline 검증 실패:", failed, flush=True)
        return 1
    print("offline 검증 통과", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sam1", type=int, default=1, help="SAM1 vit_h(2.5GB) 굽기 (0이면 생략)")
    parser.add_argument("--verify-offline", action="store_true")
    args = parser.parse_args()

    names = timm_names()
    if args.verify_offline:
        return verify_offline(names)

    failed = prefetch_timm(names)
    print("\nprefetching segmentation weights", flush=True)
    failed += download(*SAM2)
    if args.sam1:
        failed += download(*SAM1)
    else:
        print(f"  (생략) {SAM1[0]} — --sam1 0", flush=True)
    stage_adapter()

    if failed:
        print("\nPrefetch failed for:", flush=True)
        for name, err in failed:
            print(f"  {name}: {err}", flush=True)
        print("이름(yaml backbones.*.timm)이나 빌드 네트워크를 고친 뒤 다시 빌드하세요. "
              "가중치가 빠진 이미지는 폐쇄망에서 디버깅할 수 없는 학습 실패가 된다.", flush=True)
        return 1
    print(f"\nprefetched {len(names)} timm backbones + segmentation weights → {WEIGHTS_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
