#!/usr/bin/env python3
"""
데이터 인벤토리 + split.json 생성.

  python make_split.py                                  # 데이터를 자동 탐색
  python make_split.py --data-dir /workspace/data/13label_crop_padding

산출물
  split.json      모든 스크립트가 참조하는 단일 진실 (클래스/그룹/train,val,test 경로)
  inventory.csv   클래스별 장수·크기·종횡비 원자료 (보고서 셋업 절에 사용)

open/closed 정의 — 이 프로젝트 한정, 일반 관례와 반대이므로 주의:
  open   = 데이터량이 많은 제조사 (N >= MIN_OPEN). 항상 학습에 포함.
  closed = 데이터량이 적은 제조사 (N <  MIN_OPEN). 조건에 따라 학습에서 제외.

의존성 없음. Pillow가 있으면 쓰고, 없으면 JPEG 헤더를 직접 읽는다.
"""

import argparse
import csv
import json
import random
import struct
import sys
from pathlib import Path

# ---------------------------------------------------------------- 설정

_ROOT = Path(__file__).parent
DATASET_NAME = "13label_crop_padding"

# 데이터는 코드와 같은 폴더에 있을 수도, 별도 경로에 있을 수도 있다
# (runpod: 코드 /workspace/implantys_dinov2, 데이터 /workspace/data/...).
# 아래 후보를 순서대로 찾고, 없으면 --data-dir 로 직접 지정한다.
DATA_CANDIDATES = [
    _ROOT / DATASET_NAME,
    _ROOT.parent / "data" / DATASET_NAME,
    Path("/workspace/data") / DATASET_NAME,
]

OUT_SPLIT = _ROOT / "split.json"
OUT_CSV = _ROOT / "inventory.csv"

SEED = 42
RATIOS = (0.70, 0.15, 0.15)  # train / val / test
MIN_OPEN = 40  # N >= MIN_OPEN 이면 open, 미만이면 closed
IMG_EXT = {".jpg", ".jpeg", ".png"}


def find_data_dir(explicit=None):
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_dir():
            sys.exit(f"[에러] --data-dir 경로가 없음: {p}")
        return p
    for p in DATA_CANDIDATES:
        if p.is_dir():
            return p.resolve()
    sys.exit("[에러] 데이터 폴더를 찾지 못했다. --data-dir 로 지정할 것.\n"
             "  찾아본 곳:\n" + "\n".join(f"    {p}" for p in DATA_CANDIDATES))

# ---------------------------------------------------------------- 이미지 크기

try:
    from PIL import Image  # noqa

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


def _jpeg_size(fh):
    """JPEG SOF 마커에서 (w, h)를 읽는다. 실패하면 None."""
    if fh.read(2) != b"\xff\xd8":
        return None
    while True:
        b = fh.read(1)
        while b and b != b"\xff":  # 마커 시작까지 스킵
            b = fh.read(1)
        if not b:
            return None
        marker = fh.read(1)
        while marker == b"\xff":  # fill bytes
            marker = fh.read(1)
        if not marker:
            return None
        m = marker[0]
        if m in (0xD8, 0xD9) or 0xD0 <= m <= 0xD7:
            continue
        seg = fh.read(2)
        if len(seg) < 2:
            return None
        (length,) = struct.unpack(">H", seg)
        # SOF0..SOF15, DHT(C4)/JPG(C8)/DAC(CC) 제외
        if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
            body = fh.read(5)
            if len(body) < 5:
                return None
            h, w = struct.unpack(">HH", body[1:5])
            return w, h
        fh.seek(length - 2, 1)


def _png_size(fh):
    head = fh.read(24)
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    w, h = struct.unpack(">II", head[16:24])
    return w, h


def image_size(path):
    """(w, h) 또는 None(손상/판독불가)."""
    if _HAS_PIL:
        try:
            with Image.open(path) as im:
                return im.size
        except Exception:
            return None
    try:
        with open(path, "rb") as fh:
            if path.suffix.lower() == ".png":
                return _png_size(fh)
            return _jpeg_size(fh)
    except Exception:
        return None


# ---------------------------------------------------------------- 스캔


def scan(data_dir):
    """{class: [Path, ...]} 를 정렬된 순서로. 정렬은 재현성을 위해 필수."""
    if not data_dir.is_dir():
        sys.exit(f"[에러] 데이터 폴더 없음: {data_dir}")
    classes = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    if not classes:
        sys.exit(f"[에러] 하위 클래스 폴더가 없음: {data_dir}")
    out = {}
    for c in classes:
        files = sorted(
            p for p in (data_dir / c).iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXT
        )
        out[c] = files
    return out


# ---------------------------------------------------------------- 분할


def split_counts(n, ratios):
    """n장을 train/val/test로. val/test는 가능하면 최소 1장씩 확보."""
    tr, va, te = ratios
    n_val = max(1, round(n * va)) if n >= 3 else 0
    n_test = max(1, round(n * te)) if n >= 2 else 0
    n_train = n - n_val - n_test
    if n_train < 1:  # 너무 작은 클래스: train을 우선 확보
        n_train = 1
        n_test = min(n_test, n - 1)
        n_val = n - n_train - n_test
    return n_train, n_val, n_test


def make_split(files_by_class, seed, ratios):
    rng = random.Random(seed)
    split = {}
    for c in sorted(files_by_class):
        files = list(files_by_class[c])
        rng.shuffle(files)
        n_tr, n_va, n_te = split_counts(len(files), ratios)
        split[c] = {
            "train": files[:n_tr],
            "val": files[n_tr:n_tr + n_va],
            "test": files[n_tr + n_va:n_tr + n_va + n_te],
        }
    return split


# ---------------------------------------------------------------- 메인


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None,
                    help="이미지 루트 (제조사별 하위 폴더). 생략 시 자동 탐색.")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--min-open", type=int, default=MIN_OPEN,
                    help="N 이 이 값 이상이면 open, 미만이면 closed")
    args = ap.parse_args()

    data_dir = find_data_dir(args.data_dir)
    seed, min_open = args.seed, args.min_open
    print(f"데이터 경로: {data_dir}")

    files_by_class = scan(data_dir)
    counts = {c: len(f) for c, f in files_by_class.items()}
    total = sum(counts.values())

    groups = {c: ("open" if n >= min_open else "closed") for c, n in counts.items()}
    open_classes = sorted(c for c, g in groups.items() if g == "open")
    closed_classes = sorted(c for c, g in groups.items() if g == "closed")

    # --- 인벤토리 (크기/종횡비/손상 확인)
    rows, corrupt = [], []
    for c, files in files_by_class.items():
        for p in files:
            wh = image_size(p)
            if wh is None:
                corrupt.append(p)
                w = h = ar = ""
            else:
                w, h = wh
                ar = round(w / h, 4)
            rows.append({
                "path": str(p.relative_to(data_dir)),
                "class": c,
                "group": groups[c],
                "width": w,
                "height": h,
                "aspect_ratio": ar,
                "bytes": p.stat().st_size,
            })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # --- 분할
    split = make_split(files_by_class, seed, RATIOS)

    payload = {
        "_comment": (
            "이 프로젝트에서 open = 데이터량이 많은 제조사(항상 학습 포함), "
            "closed = 데이터량이 적은 제조사(조건에 따라 학습 제외). "
            "일반적인 metric learning 용어와 반대이므로 주의."
        ),
        "config": {
            "data_dir": str(data_dir),
            "dataset_name": data_dir.name,
            "seed": seed,
            "ratios": {"train": RATIOS[0], "val": RATIOS[1], "test": RATIOS[2]},
            "min_open": min_open,
        },
        "classes": sorted(counts),
        "open_classes": open_classes,
        "closed_classes": closed_classes,
        "groups": groups,
        "counts": counts,
        "splits": {
            c: {s: [str(p.relative_to(data_dir)) for p in paths]
                for s, paths in d.items()}
            for c, d in split.items()
        },
    }
    with open(OUT_SPLIT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    # --- 리포트
    ars = sorted(r["aspect_ratio"] for r in rows if r["aspect_ratio"] != "")
    print(f"데이터: {data_dir.name}  |  클래스 {len(counts)}개  |  총 {total}장")
    print(f"크기 판독: {'Pillow' if _HAS_PIL else '헤더 직접 파싱'}")
    print(f"open(N>={min_open}) {len(open_classes)}개 / closed {len(closed_classes)}개")
    print()
    print(f"{'클래스':<10} {'그룹':<7} {'전체':>5} {'train':>6} {'val':>4} {'test':>5}")
    print("-" * 43)
    tot = [0, 0, 0]
    for c in sorted(counts, key=lambda x: -counts[x]):
        d = split[c]
        n_tr, n_va, n_te = len(d["train"]), len(d["val"]), len(d["test"])
        tot[0] += n_tr
        tot[1] += n_va
        tot[2] += n_te
        print(f"{c:<10} {groups[c]:<7} {counts[c]:>5} {n_tr:>6} {n_va:>4} {n_te:>5}")
    print("-" * 43)
    print(f"{'합계':<10} {'':<7} {total:>5} {tot[0]:>6} {tot[1]:>4} {tot[2]:>5}")
    print()
    if ars:
        print(f"종횡비(w/h): min {ars[0]}  중앙값 {ars[len(ars)//2]}  max {ars[-1]}")
    print(f"불균형 비율: {max(counts.values())}:{min(counts.values())} "
          f"= {max(counts.values())/min(counts.values()):.0f}:1")

    # --- 경고
    warn = []
    for c in sorted(counts):
        n_te = len(split[c]["test"])
        if n_te <= 2:
            warn.append(f"  {c}: test {n_te}장 — 이 클래스의 F1이 거의 이산값이라 "
                        f"macro-F1을 크게 흔든다. 클래스별 성능을 반드시 함께 볼 것.")
    if corrupt:
        warn.append(f"  판독 실패 {len(corrupt)}건: "
                    + ", ".join(str(p.name) for p in corrupt[:5]))
    if warn:
        print("\n[경고]")
        print("\n".join(warn))

    print(f"\n저장: {OUT_SPLIT.name}, {OUT_CSV.name}")


if __name__ == "__main__":
    main()
