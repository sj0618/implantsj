"""
pseudo-label 라운드 3~6 이미지 풀 구성 스크립트

research_note 계획표 (누적 목표 1,000장):
  라운드3 +100 → 306   라운드4 +200 → 506
  라운드5 +200 → 706   라운드6 +294 → 1,000
  (현재 206 = 01+02. 신규 합계 794 = 100+200+200+294)

목적:
  7라벨 원본 풀(라벨 없음)에서 아직 안 쓴 이미지를 뽑아
    dataset2/03  ← 100장
    dataset2/04  ← 200장
    dataset2/05  ← 200장
    dataset2/06  ← 294장
  를 만들고, 신규 전부를 합친
    dataset3     ← 794장  (FT-206으로 한 번 추론해 pseudo-label 뽑을 용도)
  을 만든다.

원칙:
  - 소스는 '기존 데이터/7라벨' 뿐. (train/valid/test 브랜드 폴더 전체를 한 풀로 봄)
  - 이미 쓴 dataset2/01, 02, test 의 이미지는 rf 해시로 전부 제외 (leakage 0)
  - 모든 라운드끼리 ∩ = ∅, (모든 신규) ∩ (01∪02∪test) = ∅  ← "안겹치게" 강제
  - 브랜드 비율대로 층화 추출(최대잔여법). seed 42 고정 → 항상 같은 선택
  - 라벨이 없는 단계이므로 train/val 분할·annotations.json 없음. 이미지만 배치.
    (추론→검수 후 resplit_73 방식으로 브랜드별 7:3 분할하는 건 다음 단계)

산출물:
  dataset2/03..06/images/*.jpg
  dataset3/images/*.jpg              (794, 신규 전체 합본)
  dataset3/manifest.csv              (파일명, 브랜드, 라운드, 원본경로)

안전장치:
  staging에 먼저 만들고 전체 검증 통과 후에만 최종 폴더로 교체.
"""

import csv
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

SEED = 42
BASE = Path("/Users/gim-yesong/Desktop/implant")
POOL_DIR = BASE / "기존 데이터" / "7라벨"
DS2 = BASE / "dataset2"
DS3 = BASE / "dataset3"

# 라운드별 추가 장수 (순서 고정)
ROUNDS = [("03", 100), ("04", 200), ("05", 200), ("06", 294)]
N_TOTAL = sum(n for _, n in ROUNDS)  # 794
IMG_EXT = (".jpg", ".jpeg", ".png")


def rf_hash(name: str) -> str:
    if ".rf." in name:
        return name.split(".rf.")[1].rsplit(".", 1)[0]
    return name


def brand_of(name: str) -> str:
    m = re.match(r"^([a-zA-Z]+)", Path(name).name)
    return m.group(1).lower() if m else "?"


def largest_remainder(counts: dict, total: int) -> dict:
    """브랜드별 개수를 전체=total 에 맞춰 최대잔여법으로 배분."""
    pool_total = sum(counts.values())
    exact = {b: n / pool_total * total for b, n in counts.items()}
    alloc = {b: int(v) for b, v in exact.items()}
    alloc = {b: min(v, counts[b]) for b, v in alloc.items()}
    deficit = total - sum(alloc.values())
    order = sorted(counts, key=lambda b: (-(exact[b] - int(exact[b])), b))
    i = 0
    while deficit > 0:
        b = order[i % len(order)]
        if alloc[b] < counts[b]:
            alloc[b] += 1
            deficit -= 1
        i += 1
        assert i < 1000000, "배분 실패"
    return alloc


# ── 1. 풀 수집 & 제외 ──────────────────────────────────────────────────────────
pool = {}  # hash -> Path
for p in POOL_DIR.rglob("*"):
    if p.suffix.lower() in IMG_EXT:
        pool.setdefault(rf_hash(p.name), p)

used = set()
for sub in ["01/train", "01/val", "02/train", "02/val", "test"]:
    for f in (DS2 / sub / "images").glob("*.jpg"):
        used.add(rf_hash(f.name))

avail = {h: p for h, p in pool.items() if h not in used}
print(f"7라벨 고유: {len(pool)} / 이미 쓴(01+02+test): {len(used)} / 가용: {len(avail)}")
assert len(avail) >= N_TOTAL, f"가용 이미지 부족: {len(avail)} < {N_TOTAL}"

# ── 2. 브랜드 층화 추출 (신규 전체 794장) ─────────────────────────────────────
by_brand = defaultdict(list)
for h, p in avail.items():
    by_brand[brand_of(p.name)].append((h, p))

counts = {b: len(v) for b, v in by_brand.items()}
alloc_all = largest_remainder(counts, N_TOTAL)
print(f"\n브랜드별 추출 배분 (총 {N_TOTAL}):")
for b in sorted(alloc_all):
    print(f"  {b:10} 가용 {counts[b]:4} → 추출 {alloc_all[b]}")

rng = random.Random(SEED)
pick_by_brand = {}  # brand -> shuffled list, 앞에서부터 라운드에 순차 배정
for b in sorted(by_brand):
    items = sorted(by_brand[b], key=lambda x: x[1].name)  # 결정적 정렬
    rng.shuffle(items)
    pick_by_brand[b] = items[: alloc_all[b]]

selected = [(h, p, b) for b, lst in pick_by_brand.items() for (h, p) in lst]
assert len(selected) == N_TOTAL, len(selected)

# ── 3. 라운드 분배 (각 라운드도 브랜드 비율 유지) ─────────────────────────────
# 브랜드별로 각 라운드에 최대잔여법으로 나눠 담는다.
round_records = {rid: [] for rid, _ in ROUNDS}
print("\n라운드별 분배:")
header = "  브랜드      " + " ".join(f"{rid:>4}" for rid, _ in ROUNDS)
print(header)
for b in sorted(pick_by_brand):
    items = pick_by_brand[b]  # 이미 셔플됨
    # 이 브랜드 몫을 라운드 비율(라운드 크기)대로 배분
    per_round = largest_remainder({rid: n for rid, n in ROUNDS}, len(items))
    idx = 0
    row = []
    for rid, _ in ROUNDS:
        k = per_round[rid]
        for h, p in items[idx: idx + k]:
            round_records[rid].append((h, p, b))
        idx += k
        row.append(k)
    assert idx == len(items)
    print(f"  {b:10} " + " ".join(f"{x:>4}" for x in row))

# 라운드 크기 검증
for rid, n in ROUNDS:
    assert len(round_records[rid]) == n, f"{rid}: {len(round_records[rid])} != {n}"

# ── 4. 검증 (겹침·풀출처) ─────────────────────────────────────────────────────
hashes_per_round = {rid: {h for h, _, _ in recs} for rid, recs in round_records.items()}
all_new = set()
for rid, hs in hashes_per_round.items():
    assert len(hs) == dict(ROUNDS)[rid], f"{rid} 내부 중복"
    assert not (hs & all_new), f"{rid} 이 이전 라운드와 겹침!"
    all_new |= hs
assert len(all_new) == N_TOTAL, "라운드 합 중복"
assert not (all_new & used), "기존(01/02/test)과 겹침!"
assert all_new <= set(pool), "7라벨 밖 이미지 존재!"
print(f"\n검증 통과: 라운드 상호 ∩=∅, 신규 {N_TOTAL}장 ∩ (01/02/test)=∅, 전부 7라벨 출처")

# ── 5. staging 생성 ───────────────────────────────────────────────────────────
STAGE = BASE / ".staging_round"
if STAGE.exists():
    shutil.rmtree(STAGE)

manifest = []  # (round, brand, file_name, src)
for rid, _ in ROUNDS:
    out = STAGE / f"ds2_{rid}" / "images"
    out.mkdir(parents=True)
    for h, p, b in sorted(round_records[rid], key=lambda x: x[1].name):
        shutil.copy2(p, out / p.name)
        manifest.append((rid, b, p.name, str(p.relative_to(BASE))))

# dataset3 = 신규 전체 합본
ds3_img = STAGE / "ds3" / "images"
ds3_img.mkdir(parents=True)
for rid, _ in ROUNDS:
    for h, p, b in round_records[rid]:
        shutil.copy2(p, ds3_img / p.name)

# ── 6. 파일 수/유일성 검증 ────────────────────────────────────────────────────
for rid, n in ROUNDS:
    cnt = len(list((STAGE / f"ds2_{rid}" / "images").glob("*")))
    assert cnt == n, f"{rid} 파일수 {cnt} != {n}"
ds3_hashes = [rf_hash(f.name) for f in ds3_img.glob("*")]
assert len(ds3_hashes) == len(set(ds3_hashes)) == N_TOTAL, "dataset3 중복/개수 이상"
print(f"파일 배치 검증: " + ", ".join(f"{rid}={n}" for rid, n in ROUNDS) + f", dataset3={N_TOTAL}")

# ── 7. 최종 교체 ──────────────────────────────────────────────────────────────
for rid, _ in ROUNDS:
    dst = DS2 / rid
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(STAGE / f"ds2_{rid}"), str(dst))

if DS3.exists():
    shutil.rmtree(DS3)
DS3.mkdir(parents=True)
shutil.move(str(ds3_img), str(DS3 / "images"))

with open(DS3 / "manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["round", "brand", "file_name", "src_path"])
    w.writerows(sorted(manifest))

shutil.rmtree(STAGE)

# ── 8. 요약 ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("완료")
cum = 206
for rid, n in ROUNDS:
    cum += n
    got = len(list((DS2 / rid / "images").glob("*")))
    print(f"  dataset2/{rid}/images : {got:3}장  (누적 {cum})")
print(f"  dataset3/images    : {len(list((DS3/'images').glob('*')))}장 (신규 전체 합본)")
print(f"  dataset3/manifest.csv 기록")
print("=" * 55)
