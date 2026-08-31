"""
dataset2/01, dataset2/02 를 브랜드별 7:3으로 재분할한다. (RULES.md 3번)

- 기존 8:2 분할을 해제하고 각 그룹(01, 02)의 전체 이미지를 브랜드별 7:3으로 다시 나눔
- 브랜드별 val 개수는 최대잔여법으로 배분 → 전체가 정확히 7:3
- seed 고정. 같은 데이터 + 같은 seed면 항상 같은 분할
- Label Studio prefix 제거 (RULES.md 1번). 01의 이미지는 prefix가 붙어 있음
- 분할 결과를 폴더에 실제 파일로 고정

안전장치:
  staging에 먼저 만들고 전체 검증을 통과한 뒤에만 기존 폴더와 교체한다.
  기존 annotations.json은 .bak8020 으로 백업.

결과:
  01: 106장 → train 74 / val 32
  02: 100장 → train 70 / val 30
"""

import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

SEED = 42
VAL_RATIO = 0.3

BASE = Path("/Users/gim-yesong/Desktop/implant")
DS2 = BASE / "dataset2"
GROUPS = ["01", "02"]


def rf_hash(name: str) -> str:
    m = re.search(r"rf\.([0-9a-f]+)", Path(name).name)
    assert m, f"rf 해시를 찾을 수 없음: {name}"
    return m.group(1)


def clean_name(name: str) -> str:
    """Label Studio 업로드 prefix(8자리 hex + '-') 제거. RULES.md 1번."""
    return re.sub(r"^[0-9a-f]{8}-", "", Path(name).name)


def brand_of(name: str) -> str:
    return re.match(r"^([a-zA-Z]+)", clean_name(name)).group(1).lower()


def largest_remainder(counts: dict[str, int], ratio: float) -> dict[str, int]:
    """브랜드별 val 개수를 최대잔여법으로 배분해 전체 비율에 정확히 맞춘다."""
    total = sum(counts.values())
    target = round(total * ratio)

    exact = {b: n * ratio for b, n in counts.items()}
    alloc = {b: int(v) for b, v in exact.items()}          # floor
    alloc = {b: min(v, counts[b]) for b, v in alloc.items()}

    deficit = target - sum(alloc.values())
    # 잔여분이 큰 브랜드부터. 동률이면 브랜드명으로 결정적 정렬(재현성)
    order = sorted(counts, key=lambda b: (-(exact[b] - int(exact[b])), b))
    i = 0
    while deficit > 0:
        b = order[i % len(order)]
        if alloc[b] < counts[b]:
            alloc[b] += 1
            deficit -= 1
        i += 1
        assert i < 10000, "배분 실패"
    return alloc


# ── 1. 수집 ───────────────────────────────────────────────────────────────────
records: dict[str, list[dict]] = {}

for g in GROUPS:
    recs = []
    seen_hash = set()
    for split in ["train", "val"]:
        d = json.loads(DS2.joinpath(g, split, "annotations.json").read_text())
        by_img = defaultdict(list)
        for a in d["annotations"]:
            by_img[a["image_id"]].append(a)
        for img in d["images"]:
            fn = clean_name(img["file_name"])
            src = DS2 / g / split / "images" / Path(img["file_name"]).name
            assert src.exists(), f"원본 이미지 없음: {src}"
            h = rf_hash(fn)
            assert h not in seen_hash, f"{g} 내부 중복 이미지: {fn}"
            seen_hash.add(h)
            recs.append({
                "file_name": fn,
                "hash": h,
                "brand": brand_of(fn),
                "width": img["width"],
                "height": img["height"],
                "src": src,
                "boxes": by_img[img["id"]],
            })
    records[g] = recs
    print(f"{g}: 수집 {len(recs)}장 / {sum(len(r['boxes']) for r in recs)}박스")

# clean_name 충돌 검사 (prefix 제거 후 이름이 겹치면 안 됨)
for g in GROUPS:
    names = [r["file_name"] for r in records[g]]
    assert len(names) == len(set(names)), f"{g}: prefix 제거 후 파일명 충돌"


# ── 2. 브랜드별 7:3 분할 ──────────────────────────────────────────────────────
plan: dict[str, dict[str, list[dict]]] = {}

for g in GROUPS:
    by_brand = defaultdict(list)
    for r in records[g]:
        by_brand[r["brand"]].append(r)

    counts = {b: len(v) for b, v in by_brand.items()}
    val_alloc = largest_remainder(counts, VAL_RATIO)

    rng = random.Random(SEED)
    tr, va = [], []
    print(f"\n{g} 브랜드별 배분 (seed={SEED}):")
    for b in sorted(by_brand):                    # 브랜드 순회 순서 고정
        items = sorted(by_brand[b], key=lambda r: r["file_name"])  # 정렬 후
        rng.shuffle(items)                        # 셔플 → 재현 가능
        nv = val_alloc[b]
        va += items[:nv]
        tr += items[nv:]
        print(f"  {b:8} {counts[b]:3} → train {counts[b] - nv:3} / val {nv:2}")

    n = len(records[g])
    print(f"  {'합계':8} {n:3} → train {len(tr):3} / val {len(va):2}"
          f"  ({len(tr) / n * 100:.0f}:{len(va) / n * 100:.0f})")
    assert len(va) == round(n * VAL_RATIO), f"{g}: val {len(va)} != {round(n * VAL_RATIO)}"
    plan[g] = {"train": tr, "val": va}


# ── 3. staging 에 생성 ────────────────────────────────────────────────────────
STAGE = DS2 / ".staging_73"
if STAGE.exists():
    shutil.rmtree(STAGE)

for g in GROUPS:
    for split in ["train", "val"]:
        out_img = STAGE / g / split / "images"
        out_img.mkdir(parents=True)

        imgs, anns = [], []
        next_ann = 1
        for iid, r in enumerate(sorted(plan[g][split], key=lambda r: r["file_name"]), 1):
            shutil.copy2(r["src"], out_img / r["file_name"])
            imgs.append({"width": r["width"], "height": r["height"],
                         "id": iid, "file_name": r["file_name"]})
            for a in r["boxes"]:
                anns.append({
                    "id": next_ann,
                    "image_id": iid,
                    "category_id": a["category_id"],
                    "segmentation": a.get("segmentation", []),
                    "bbox": a["bbox"],
                    "ignore": a.get("ignore", 0),
                    "iscrowd": a.get("iscrowd", 0),
                    "area": a.get("area", a["bbox"][2] * a["bbox"][3]),
                })
                next_ann += 1

        (STAGE / g / split / "annotations.json").write_text(json.dumps(
            {"images": imgs, "annotations": anns,
             "categories": [{"id": 0, "name": "implant"}]}, indent=2))


# ── 4. 검증 ───────────────────────────────────────────────────────────────────
test_hashes = {rf_hash(i["file_name"])
               for i in json.loads(DS2.joinpath("test", "annotations.json").read_text())["images"]}

EXPECT = {"01": (74, 32), "02": (70, 30)}
all_hashes = []

for g in GROUPS:
    hs = {}
    for split in ["train", "val"]:
        d = json.loads(STAGE.joinpath(g, split, "annotations.json").read_text())
        n_exp = EXPECT[g][0 if split == "train" else 1]
        assert len(d["images"]) == n_exp, f"{g}/{split}: {len(d['images'])} != {n_exp}"

        files = list((STAGE / g / split / "images").glob("*.jpg"))
        assert len(files) == n_exp, f"{g}/{split}: 파일 {len(files)} != {n_exp}"
        for i in d["images"]:
            p = STAGE / g / split / "images" / i["file_name"]
            assert p.exists(), f"파일 없음: {p}"
            assert not re.match(r"^[0-9a-f]{8}-", i["file_name"]), \
                f"prefix 잔존(RULES 1 위반): {i['file_name']}"

        # 박스 무결성
        assert {a["image_id"] for a in d["annotations"]} <= {i["id"] for i in d["images"]}, "고아 ann"
        no_box = {i["id"] for i in d["images"]} - {a["image_id"] for a in d["annotations"]}
        assert not no_box, f"{g}/{split}: 박스 없는 이미지 {len(no_box)}장"

        hs[split] = {rf_hash(i["file_name"]) for i in d["images"]}
        all_hashes += list(hs[split])

    assert not (hs["train"] & hs["val"]), f"{g}: train ∩ val 겹침"

    # 박스 총합 보존
    before = sum(len(r["boxes"]) for r in records[g])
    after = sum(len(json.loads(STAGE.joinpath(g, s, "annotations.json").read_text())["annotations"])
                for s in ["train", "val"])
    assert before == after, f"{g}: 박스 수 손실 {before} → {after}"

# 이미지 손실·중복 없음
assert len(all_hashes) == len(set(all_hashes)) == 206, f"이미지 수 이상: {len(all_hashes)}"
# test leakage
overlap = set(all_hashes) & test_hashes
assert not overlap, f"LEAKAGE! train/val ∩ test = {overlap}"

print("\n검증 통과: 206장 보존, train ∩ val = ∅, ∩ test = ∅, prefix 0, 박스 손실 0")


# ── 5. 교체 ───────────────────────────────────────────────────────────────────
for g in GROUPS:
    for split in ["train", "val"]:
        old = DS2 / g / split
        bak = old / "annotations.json.bak8020"
        if not bak.exists():
            shutil.copy2(old / "annotations.json", bak)
        keep = bak.read_bytes()
        shutil.rmtree(old)
        shutil.move(str(STAGE / g / split), str(old))
        (old / "annotations.json.bak8020").write_bytes(keep)

shutil.rmtree(STAGE)

print("\n최종:")
for g in GROUPS:
    for split in ["train", "val"]:
        d = json.loads(DS2.joinpath(g, split, "annotations.json").read_text())
        n = len(list((DS2 / g / split / "images").glob("*.jpg")))
        print(f"  {g}/{split:5} {len(d['images']):3}장 {len(d['annotations']):3}박스  (jpg {n})")
print("\n기존 8:2 라벨은 각 폴더의 annotations.json.bak8020 에 백업됨")
