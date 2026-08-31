"""
dataset2/test/ 의 Label Studio prefix를 제거한다. (RULES.md 1번)

기존 13장은 Label Studio를 거쳐 들어와 파일명에 8자리 hex prefix가 붙어 있고
(예: 642983cb-osstem407_jpg.rf.….jpg), 신규 35장은 prefix가 없어 섞여 있었다.

파일명만 바꾼다. 이미지 내용·박스 좌표·이미지 수는 전혀 건드리지 않으므로
"test 확정 후 변경 금지" 원칙과 충돌하지 않는다.
"""

import hashlib
import json
import re
from pathlib import Path

BASE = Path("/Users/gim-yesong/Desktop/implant")
TEST = BASE / "dataset2" / "test"
IMGS = TEST / "images"

clean = lambda n: re.sub(r"^[0-9a-f]{8}-", "", Path(n).name)
md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()

d = json.loads((TEST / "annotations.json").read_text())

before_n = len(d["images"])
before_boxes = len(d["annotations"])
before_md5 = sorted(md5(IMGS / i["file_name"]) for i in d["images"])
before_geom = sorted((i["id"], tuple(a["bbox"]))
                     for i in d["images"] for a in d["annotations"] if a["image_id"] == i["id"])

targets = [i for i in d["images"] if re.match(r"^[0-9a-f]{8}-", i["file_name"])]
print(f"prefix 있는 이미지: {len(targets)}장 / 전체 {before_n}장")

# 이름 충돌 검사
new_names = [clean(i["file_name"]) for i in d["images"]]
assert len(new_names) == len(set(new_names)), "prefix 제거 시 파일명 충돌"

# 파일 rename + file_name 갱신
for img in targets:
    src = IMGS / img["file_name"]
    dst = IMGS / clean(img["file_name"])
    assert src.exists(), f"원본 없음: {src}"
    assert not dst.exists(), f"대상이 이미 존재: {dst}"
    src.rename(dst)
    img["file_name"] = dst.name

(TEST / "annotations.json").write_text(json.dumps(d, indent=2))

# ── 검증: 파일명만 바뀌고 나머지는 그대로인가 ──────────────────────────────────
d2 = json.loads((TEST / "annotations.json").read_text())
assert len(d2["images"]) == before_n, "이미지 수 변경됨"
assert len(d2["annotations"]) == before_boxes, "박스 수 변경됨"
assert sorted(md5(IMGS / i["file_name"]) for i in d2["images"]) == before_md5, \
    "이미지 내용 변경됨"
assert sorted((i["id"], tuple(a["bbox"]))
              for i in d2["images"] for a in d2["annotations"]
              if a["image_id"] == i["id"]) == before_geom, "박스 좌표 변경됨"

for i in d2["images"]:
    assert not re.match(r"^[0-9a-f]{8}-", i["file_name"]), f"prefix 잔존: {i['file_name']}"
    assert (IMGS / i["file_name"]).exists(), f"파일 없음: {i['file_name']}"

n_jpg = len(list(IMGS.glob("*.jpg")))
assert n_jpg == before_n, f"디스크 파일 수 {n_jpg} != {before_n}"

print(f"완료: {len(targets)}장 rename, prefix 0")
print(f"검증: {before_n}장 / {before_boxes}박스 / md5·bbox 동일")
