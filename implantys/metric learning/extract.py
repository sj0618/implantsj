#!/usr/bin/env python3
"""
임베딩 1회 추출 + 디스크 캐싱.

  python extract.py --cond M1                      # pretrained DINOv2 그대로 (Base)
  python extract.py --cond M4 --ckpt ckpt/M4.pt    # 학습된 backbone

산출물
  embeddings/{cond}/{split}.npz   vecs(N,768) / labels(N,) / paths(N,)

중요 1 — 학습 범위와 무관하게 항상 **13개 클래스 전부**를 추출한다.
9개로 학습한 모델(M3/M5/M7)도 E3(13개 분류)를 평가해야 하고,
학습에 안 쓴 closed 4개의 프로토타입은 그 train 임베딩으로 만들기 때문이다.
분류기(head)는 쓰지 않는다 — 9개 학습 모델의 head 는 출력이 9개뿐이라
closed 4개를 표현할 수 없지만, 임베딩 공간에는 그런 제약이 없다.

중요 2 — 임베딩은 **backbone → projection head** 출력이다 (M1 Base 만 예외로
backbone 출력 그대로). frozen 조건은 backbone 이 안 변하므로, proj 를 빼면
임베딩이 M1 과 완전히 같아져 셀이 복사본이 된다. 자세한 이유는 model.py 참조.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from dataset import make_loader, load_split
from model import Embedder, ProjectionHead, build_backbone, pick_device

_ROOT = Path(__file__).parent
OUT_ROOT = _ROOT / "embeddings"


@torch.no_grad()
def extract_split(embedder, split, device, batch_size, num_workers, meta, amp=True):
    ds, loader = make_loader(split, scope="all", batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             train_aug=False, meta=meta)
    vecs, labels, paths = [], [], []
    use_amp = amp and device.type == "cuda"
    for x, y, p in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            emb = embedder(x)
        vecs.append(emb.float().cpu().numpy())
        labels.extend(ds.classes[i] for i in y.tolist())
        paths.extend(p)
    return np.concatenate(vecs, 0), np.array(labels), np.array(paths)


def build_embedder(ckpt_path=None):
    """ckpt 가 없으면 pretrained backbone 그대로(M1), 있으면 backbone + proj."""
    if ckpt_path is None:
        return Embedder(build_backbone(pretrained=True), proj=None)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone = build_backbone(pretrained=False)
    missing, unexpected = backbone.load_state_dict(ckpt["backbone"], strict=False)
    if missing or unexpected:
        print(f"  [주의] backbone state_dict 불일치 — missing {len(missing)}, "
              f"unexpected {len(unexpected)}")
        if missing:
            print(f"    missing 예시: {missing[:3]}")

    if "proj" not in ckpt:
        raise KeyError(
            f"{ckpt_path} 에 projection head 가 없다. "
            "projection head 도입 이전에 학습된 체크포인트다 — train.py 로 재학습할 것.")
    proj = ProjectionHead()
    proj.load_state_dict(ckpt["proj"])

    info = {k: ckpt[k] for k in ("cond", "train_scope", "epoch", "val_macro_f1")
            if k in ckpt}
    print(f"  체크포인트: {info}")
    return Embedder(backbone, proj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True,
                    help="조건 코드 (M1~M7). 출력 폴더명이 된다.")
    ap.add_argument("--ckpt", default=None,
                    help="학습된 backbone 체크포인트. 없으면 pretrained 그대로(=M1).")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = pick_device(args.device)
    meta = load_split()
    out_dir = OUT_ROOT / args.cond
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = [s for s in args.splits
            if args.overwrite or not (out_dir / f"{s}.npz").is_file()]
    if not todo:
        print(f"[{args.cond}] 이미 추출됨. 다시 하려면 --overwrite")
        return

    print(f"[{args.cond}] device={device}  splits={todo}")
    if args.ckpt:
        print(f"  체크포인트 로딩: {args.ckpt}")
    embedder = build_embedder(args.ckpt)
    kind = "backbone only (Base)" if embedder.proj is None else "backbone + projection"
    print(f"  임베딩: {kind}")
    embedder.eval().to(device)

    for split in todo:
        vecs, labels, paths = extract_split(
            embedder, split, device, args.batch_size, args.num_workers, meta)
        np.savez_compressed(out_dir / f"{split}.npz",
                            vecs=vecs, labels=labels, paths=paths)
        n_cls = len(set(labels.tolist()))
        print(f"  {split:<5} {vecs.shape[0]:>4}장  dim={vecs.shape[1]}  "
              f"클래스 {n_cls}개  → {out_dir.name}/{split}.npz")

    print(f"저장 완료: {out_dir}")


if __name__ == "__main__":
    main()
