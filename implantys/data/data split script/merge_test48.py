"""
Step A. test 세트 병합 스크립트 (13장 + 35장 = 48장)

- 기존 test 13장 + Label Studio 신규 라벨 37장에서 leakage 2장 제외한 35장
- leakage 판정 기준: Roboflow 콘텐츠 해시 (rf.<hash>)
  * 파일명 문자열 비교는 Label Studio 업로드 prefix(예: 73d1c242-) 때문에 실패함.
    sample_test37.py가 이 이유로 01/train의 2장을 후보 풀에 남겨 뽑아버렸음.
- 병합 후 dataset2/test/ 최종 확정 (이후 변경 금지)

출력:
  dataset2/test/annotations.json  (덮어씀, 기존은 .bak13 으로 백업)
  dataset2/test/images/           (신규 35장 복사)
"""

import json
import re
import shutil
from pathlib import Path

BASE   = Path("/Users/gim-yesong/Desktop/implant")
DS2    = BASE / "dataset2"
TEST   = DS2 / "test"
CAND   = DS2 / "test_candidates"
NEW_JS = DS2 / "project-5-at-2026-07-16-15-08-80f1538f" / "result.json"

TRAIN_SETS = ["01/train", "01/val", "02/train", "02/val"]

# 01/train 과 중복되어 제외하는 2장 (해시 기준)
LEAK_HASHES = {
    "092fb34d1bca7453c329b9993d7c15ca",  # nobel7-copy-10
    "5b89cefe036ad13aacd793bf12c257ab",  # dionavi8-copy-10
}

MANDATORY = "noris773_jpg.rf.1876193f43dddec8b105a021be40e3fc.jpg"


def rf_hash(name: str) -> str:
    """Roboflow 콘텐츠 해시 추출. 이미지 고유 식별자."""
    m = re.search(r"rf\.([0-9a-f]+)", Path(name).name)
    assert m, f"rf 해시를 찾을 수 없음: {name}"
    return m.group(1)


def clean_name(name: str) -> str:
    """Label Studio 업로드 prefix(8자리 hex + '-') 제거."""
    b = Path(name).name
    return re.sub(r"^[0-9a-f]{8}-", "", b)


# ── 1. 로드 ───────────────────────────────────────────────────────────────────
old = json.loads(TEST.joinpath("annotations.json").read_text())
new = json.loads(NEW_JS.read_text())

print(f"기존 test : {len(old['images'])}장 / {len(old['annotations'])}박스")
print(f"신규 라벨 : {len(new['images'])}장 / {len(new['annotations'])}박스")

assert old["categories"] == new["categories"] == [{"id": 0, "name": "implant"}], \
    "categories 불일치"


# ── 2. leakage 2장 제외 ───────────────────────────────────────────────────────
drop_ids = {i["id"] for i in new["images"] if rf_hash(i["file_name"]) in LEAK_HASHES}
assert len(drop_ids) == len(LEAK_HASHES), \
    f"제외 대상 {len(LEAK_HASHES)}장 중 {len(drop_ids)}장만 발견됨"

keep_imgs = [i for i in new["images"] if i["id"] not in drop_ids]
keep_anns = [a for a in new["annotations"] if a["image_id"] not in drop_ids]
print(f"leakage 제외 후: {len(keep_imgs)}장 / {len(keep_anns)}박스")


# ── 3. id 재부여 후 병합 ──────────────────────────────────────────────────────
next_img_id = max(i["id"] for i in old["images"]) + 1
next_ann_id = max(a["id"] for a in old["annotations"]) + 1

merged_imgs = list(old["images"])
merged_anns = list(old["annotations"])
id_map: dict[int, int] = {}

for img in sorted(keep_imgs, key=lambda x: clean_name(x["file_name"])):
    fn = clean_name(img["file_name"])
    assert (CAND / fn).exists(), f"test_candidates에 이미지 없음: {fn}"
    id_map[img["id"]] = next_img_id
    merged_imgs.append({
        "width":     img["width"],
        "height":    img["height"],
        "id":        next_img_id,
        "file_name": fn,
    })
    next_img_id += 1

for ann in sorted(keep_anns, key=lambda a: (a["image_id"], a["id"])):
    merged_anns.append({
        "id":           next_ann_id,
        "image_id":     id_map[ann["image_id"]],
        "category_id":  ann["category_id"],
        "segmentation": ann.get("segmentation", []),
        "bbox":         ann["bbox"],
        "ignore":       ann.get("ignore", 0),
        "iscrowd":      ann.get("iscrowd", 0),
        "area":         ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
    })
    next_ann_id += 1

merged = {"images": merged_imgs, "annotations": merged_anns,
          "categories": old["categories"]}


# ── 4. 검증 ───────────────────────────────────────────────────────────────────
assert len(merged_imgs) == 48, f"이미지 수 {len(merged_imgs)} != 48"
assert len(merged_anns) == 81, f"박스 수 {len(merged_anns)} != 81"  # 16 + 65

ids = [i["id"] for i in merged_imgs]
assert len(ids) == len(set(ids)), "image id 중복"
aids = [a["id"] for a in merged_anns]
assert len(aids) == len(set(aids)), "annotation id 중복"

hashes = [rf_hash(i["file_name"]) for i in merged_imgs]
assert len(hashes) == len(set(hashes)), "test 내부에 중복 이미지 존재"

# 핵심: test ∩ train = ∅ (해시 기준)
train_hashes = set()
for sub in TRAIN_SETS:
    d = json.loads(DS2.joinpath(sub, "annotations.json").read_text())
    train_hashes |= {rf_hash(i["file_name"]) for i in d["images"]}
overlap = set(hashes) & train_hashes
assert not overlap, f"LEAKAGE! test ∩ train = {overlap}"

# 모든 이미지에 박스가 최소 1개
img_ids_with_ann = {a["image_id"] for a in merged_anns}
no_box = set(ids) - img_ids_with_ann
assert not no_box, f"박스 없는 이미지: {no_box}"

# bbox가 이미지 경계 안에 있는지
dims = {i["id"]: (i["width"], i["height"]) for i in merged_imgs}
for a in merged_anns:
    x, y, w, h = a["bbox"]
    W, H = dims[a["image_id"]]
    assert w > 0 and h > 0, f"bbox 크기 이상: ann {a['id']}"
    assert -1 <= x <= W + 1 and -1 <= y <= H + 1 and x + w <= W + 1 and y + h <= H + 1, \
        f"bbox 경계 초과: ann {a['id']} bbox={a['bbox']} img={W}x{H}"

assert any(clean_name(i["file_name"]) == MANDATORY for i in merged_imgs), \
    f"필수 이미지 누락: {MANDATORY}"

print("검증 통과: 48장 / 81박스, leakage 0, 박스 없는 이미지 0")


# ── 5. 쓰기 ───────────────────────────────────────────────────────────────────
bak = TEST / "annotations.json.bak13"
if not bak.exists():
    shutil.copy2(TEST / "annotations.json", bak)
    print(f"기존 annotations 백업: {bak.name}")

copied = 0
for img in merged_imgs[len(old["images"]):]:
    dst = TEST / "images" / img["file_name"]
    if not dst.exists():
        shutil.copy2(CAND / img["file_name"], dst)
        copied += 1

TEST.joinpath("annotations.json").write_text(json.dumps(merged, indent=2))

n_files = len(list((TEST / "images").glob("*.jpg")))
assert n_files == 48, f"test/images 파일 수 {n_files} != 48"
for i in merged_imgs:
    assert (TEST / "images" / i["file_name"]).exists(), f"파일 없음: {i['file_name']}"

print(f"이미지 복사: {copied}장 → test/images ({n_files}장)")
print(f"저장 완료: {TEST / 'annotations.json'}")
print("=" * 55)
print("test 48장 확정. 이후 변경 금지.")
