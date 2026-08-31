#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_test2.py

목적:
  학습(train/val)·기존 test에 한 번도 쓰이지 않은 '깨끗한' 원천 이미지에서
  152장을 브랜드 비율에 맞춰 뽑아 test2/images 에 넣는다.
  (기존 test 48장은 test2에 복사하지 않는다. 기존 label을 그대로 재사용하고
   최종 GT 만들 때 dataset2/test 의 48장과 합칠 예정.)

원칙:
  - 원천 풀:  원본 데이터/7라벨  (train + valid + test 전체)
  - 사용 제외: dataset2/01~06 의 train/val + dataset2/test(48장)
  - 파일 식별: 파일명 앞부분 base-ID (예: adin434) 기준. roboflow 해시 접미사는
    재-export 시 달라질 수 있으므로 매칭 키에서 제외한다.
  - 재현성: seed 42 고정 (기존 실험과 동일).

산출물:
  gdino finetuning/test2/images/   -> 152장 (신규만)
  gdino finetuning/test2/manifest.csv -> base_id, brand, source(new), src_path
"""

import os
import re
import csv
import random
import shutil

# ---- 경로 (이 스크립트 위치 기준) ---------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))          # .../gdino finetuning
REPO = os.path.dirname(HERE)                                # .../implant
POOL_DIR = os.path.join(REPO, "원본 데이터", "7라벨")       # 원천 풀
DS_DIR   = os.path.join(HERE, "dataset2")                   # 학습/기존test 위치
USED_ROUNDS = ["01", "02", "03", "04", "05", "06"]         # train/val로 사용된 라운드
TEST48_DIR  = os.path.join(DS_DIR, "test", "images")        # 기존 고정 test 48

OUT_DIR   = os.path.join(HERE, "test2")
OUT_IMG   = os.path.join(OUT_DIR, "images")
MANIFEST  = os.path.join(OUT_DIR, "manifest.csv")

SEED = 42
N_NEW = 152                                                 # 추가로 뽑을 장수
BRANDS = ["adin", "dionavi", "dentium", "mis", "noris", "nobel", "osstem"]
IMG_EXT = (".jpg", ".jpeg", ".png")

# ---- 유틸 ---------------------------------------------------------------
HASH_RE = re.compile(r"_jpg\.rf\.[a-f0-9]+\.(jpg|jpeg|png)$", re.IGNORECASE)
EXT_RE  = re.compile(r"\.(jpg|jpeg|png)$", re.IGNORECASE)

def base_id(fname):
    """파일명 -> base-ID (소문자). 예: adin434_jpg.rf.<hash>.jpg -> adin434"""
    s = HASH_RE.sub("", fname)
    s = EXT_RE.sub("", s)
    return s.lower()

def brand_of(bid):
    m = re.match(r"^([a-z]+)\d", bid)
    return m.group(1) if m else "unknown"

def walk_images(root):
    out = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXT):
                out.append(os.path.join(dirpath, f))
    return out

# ---- 1) 원천 풀 스캔: base_id -> 대표 파일경로 --------------------------
pool = {}
for p in walk_images(POOL_DIR):
    bid = base_id(os.path.basename(p))
    pool.setdefault(bid, p)          # 첫 번째 등장 경로만 보관

# ---- 2) 사용된(train/val + test48) base_id 수집 ------------------------
used = set()
for r in USED_ROUNDS:
    for p in walk_images(os.path.join(DS_DIR, r)):
        used.add(base_id(os.path.basename(p)))
test48 = {}   # base_id -> 실제 파일경로 (그대로 복사하려고 경로 보관)
for p in walk_images(TEST48_DIR):
    test48[base_id(os.path.basename(p))] = p
used |= set(test48.keys())

# ---- 3) 깨끗한 available 계산 ------------------------------------------
available = {bid: path for bid, path in pool.items() if bid not in used}

by_brand = {b: [] for b in BRANDS}
for bid in available:
    b = brand_of(bid)
    if b in by_brand:
        by_brand[b].append(bid)

print(f"[pool] unique base-id      : {len(pool)}")
print(f"[used] train/val+test48    : {len(used)}")
print(f"[available] clean          : {len(available)}")
for b in BRANDS:
    print(f"    {b:8s}: {len(by_brand[b])}")

# ---- 4) 브랜드 비율에 맞춰 152장 배분 (largest-remainder) -------------
rng = random.Random(SEED)
avail_total = len(available)
raw = {b: (len(by_brand[b]) / avail_total * N_NEW) for b in BRANDS}
alloc = {b: int(raw[b]) for b in BRANDS}                    # 내림
alloc = {b: min(alloc[b], len(by_brand[b])) for b in BRANDS}
# 남은 자리를 소수부 큰 순서로 채움 (여유 있는 브랜드에만)
remaining = N_NEW - sum(alloc.values())
frac_order = sorted(BRANDS, key=lambda b: (raw[b] - int(raw[b])), reverse=True)
i = 0
while remaining > 0:
    b = frac_order[i % len(BRANDS)]
    if alloc[b] < len(by_brand[b]):
        alloc[b] += 1
        remaining -= 1
    i += 1
    if i > 10000:
        break

print("\n[allocation] new 152 by brand:")
for b in BRANDS:
    print(f"    {b:8s}: {alloc[b]}  (avail {len(by_brand[b])})")
print(f"    TOTAL new: {sum(alloc.values())}")

# ---- 5) 샘플링 (seed 고정) --------------------------------------------
picked_new = []   # (bid, src_path, brand)
for b in BRANDS:
    ids = sorted(by_brand[b])          # 정렬로 결정성 확보
    rng.shuffle(ids)
    chosen = ids[:alloc[b]]
    for bid in chosen:
        picked_new.append((bid, available[bid], b))

# ---- 6) 복사: 신규 152장만 -> test2/images ---------------------------
#   (기존 test48은 used 제외에는 쓰이지만 test2로 복사하지 않는다.)
if os.path.isdir(OUT_IMG):
    for f in os.listdir(OUT_IMG):
        if f.lower().endswith(IMG_EXT):
            os.remove(os.path.join(OUT_IMG, f))   # 재실행 시 기존 파일 정리
os.makedirs(OUT_IMG, exist_ok=True)

manifest_rows = []
for bid, src, b in sorted(picked_new):
    dst = os.path.join(OUT_IMG, os.path.basename(src))
    shutil.copy2(src, dst)
    manifest_rows.append([bid, b, "new", src])

with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["base_id", "brand", "source", "src_path"])
    w.writerows(manifest_rows)

n_files = len([f for f in os.listdir(OUT_IMG) if f.lower().endswith(IMG_EXT)])
print(f"\n[done] copied files in test2/images : {n_files}  (신규만)")
print(f"       new rows: {len(manifest_rows)}")
print(f"       manifest: {MANIFEST}")
