"""
test 세트 추가 37장 샘플링 스크립트

- 소스: 기존 데이터/7라벨 (train+valid+test 폴더 통합)
- 제외: dataset2에 이미 포함된 219장
- 필수 포함: noris773_jpg.rf.1876193f43dddec8b105a021be40e3fc.jpg
- 목표: 37장, 제조사 균형 (소수 제조사는 있는 만큼)
- 주의: test 48장은 이미 확정됨(merge_test48.py). 이 스크립트는 다음 라운드
         샘플링용이며, test를 다시 만들기 위해 재실행하면 안 된다.
- 출력: dataset2/test_candidates/ 에 이미지 복사
         data split script/test37_list.csv 에 목록 저장
"""

import csv
import math
import random
import re
from pathlib import Path

SEED = 42
random.seed(SEED)

BASE        = Path("/Users/gim-yesong/Desktop/implant")
SRC_7LABEL  = BASE / "기존 데이터" / "7라벨"
DS2         = BASE / "dataset2"
OUT_DIR     = DS2 / "test_candidates"
OUT_CSV     = BASE / "data split script" / "test37_list.csv"

TOTAL_SAMPLE = 37
BRANDS       = ["ADIN", "Dentium", "DIONAVI", "MIS", "nobel", "NORIS", "osstem"]
MANDATORY    = "noris773_jpg.rf.1876193f43dddec8b105a021be40e3fc.jpg"


# ── 1. 이미 사용된 이미지 수집 ────────────────────────────────────────────────
# 파일명 문자열로 비교하면 안 된다. dataset2/01 과 test 의 이미지는 Label Studio를
# 거치며 업로드 prefix(예: 73d1c242-)가 붙어 원본 파일명과 절대 일치하지 않는다.
# 이 때문에 최초 실행 시 01/train 의 2장(dionavi8-copy-10, nobel7-copy-10)이
# 후보 풀에 남아 test로 뽑히는 leakage가 발생했다. rf.<hash>는 이미지 콘텐츠
# 고유값이라 prefix 유무와 무관하게 동일하므로 이것으로 비교한다.
def rf_hash(name: str) -> str:
    m = re.search(r"rf\.([0-9a-f]+)", Path(name).name)
    assert m, f"rf 해시를 찾을 수 없음: {name}"
    return m.group(1)


used = {rf_hash(img.name) for img in DS2.rglob("*.jpg")
        if OUT_DIR not in img.parents}
print(f"기존 사용 이미지: {len(used)}장")


# ── 2. 브랜드별 후보 풀 구성 (7라벨 전체 - 사용 이미지) ─────────────────────
pool: dict[str, list[Path]] = {}
for brand in BRANDS:
    files = []
    for split in ["train", "valid", "test"]:
        files += list((SRC_7LABEL / split / brand).glob("*.jpg"))
    remaining = [f for f in files if rf_hash(f.name) not in used]
    pool[brand] = sorted(remaining)

print("\n브랜드별 잔여 후보:")
for brand, files in pool.items():
    print(f"  {brand}: {len(files)}장")
total_pool = sum(len(v) for v in pool.values())
print(f"  합계: {total_pool}장")


# ── 3. 필수 이미지 확인 ───────────────────────────────────────────────────────
mandatory_path = next(
    (f for files in pool.values() for f in files if f.name == MANDATORY), None
)
assert mandatory_path is not None, f"필수 이미지를 후보 풀에서 찾을 수 없음: {MANDATORY}"
mandatory_brand = mandatory_path.parent.name  # 브랜드 폴더명


# ── 4. 브랜드별 할당 수 계산 ──────────────────────────────────────────────────
# 전략:
#   - 소수 브랜드(DIONAVI, nobel)는 보유량 전체 사용 (단, 상한 3)
#   - 나머지 N 슬롯을 대형 브랜드에 비례 배분
#   - 필수 이미지가 속한 브랜드(NORIS)는 그 슬롯을 채우면서 mandatory 포함 보장

SMALL_CAP = 3  # 소수 브랜드 최대 할당

small_brands  = {b for b in BRANDS if len(pool[b]) <= SMALL_CAP}
large_brands  = [b for b in BRANDS if b not in small_brands]

# 소수 브랜드 할당 (보유량 전부)
allocation: dict[str, int] = {}
for b in small_brands:
    allocation[b] = len(pool[b])

remaining_slots = TOTAL_SAMPLE - sum(allocation.values())
large_pool_total = sum(len(pool[b]) for b in large_brands)

# 비례 배분 (반올림 후 합계 맞춤)
raw = {b: remaining_slots * len(pool[b]) / large_pool_total for b in large_brands}
alloc_large = {b: max(1, math.floor(v)) for b, v in raw.items()}

# 남은 슬롯 보정 (반올림 오차)
deficit = remaining_slots - sum(alloc_large.values())
fracs = sorted(large_brands, key=lambda b: raw[b] - math.floor(raw[b]), reverse=True)
for i in range(deficit):
    alloc_large[fracs[i % len(fracs)]] += 1

allocation.update(alloc_large)

# 각 브랜드 가용량 초과 방지
for b in BRANDS:
    allocation[b] = min(allocation[b], len(pool[b]))

print(f"\n브랜드별 목표 할당 (합계={sum(allocation.values())}):")
for b in BRANDS:
    print(f"  {b}: {allocation[b]}")


# ── 5. 샘플링 ─────────────────────────────────────────────────────────────────
selected: list[Path] = []

for brand in BRANDS:
    candidates = pool[brand].copy()
    n = allocation[brand]

    # mandatory 이미지가 이 브랜드에 속하면 먼저 고정
    if brand == mandatory_brand:
        candidates.remove(mandatory_path)
        selected.append(mandatory_path)
        n -= 1  # 나머지 슬롯에서 1 차감

    random.shuffle(candidates)
    selected.extend(candidates[:n])

assert len(selected) == TOTAL_SAMPLE, f"샘플 수 불일치: {len(selected)} != {TOTAL_SAMPLE}"
assert any(f.name == MANDATORY for f in selected), "필수 이미지 누락!"

# 중복 검사
selected_names = [f.name for f in selected]
assert len(selected_names) == len(set(selected_names)), "선택 목록에 중복 있음!"

# 기존 사용 이미지와 교집합 없는지 재확인
overlap = {rf_hash(n) for n in selected_names} & used
assert len(overlap) == 0, f"leakage! 교집합: {overlap}"


# ── 6. 출력: 이미지 복사 + CSV ────────────────────────────────────────────────
import shutil

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

with open(OUT_CSV, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["file_name", "brand", "src_split"])
    for f in sorted(selected, key=lambda x: x.name):
        brand    = f.parent.name
        src_split = f.parent.parent.name  # train / valid / test
        shutil.copy2(f, OUT_DIR / f.name)
        writer.writerow([f.name, brand, src_split])


# ── 7. 최종 요약 ──────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"샘플링 완료: {len(selected)}장")
print(f"출력 폴더  : {OUT_DIR}")
print(f"목록 CSV   : {OUT_CSV}")
print("\n브랜드별 실제 선택:")
from collections import Counter
brand_counts = Counter(f.parent.name for f in selected)
for b in BRANDS:
    print(f"  {b}: {brand_counts.get(b, 0)}장")
print(f"  필수 포함 확인: {MANDATORY} ✓")
print("=" * 55)
