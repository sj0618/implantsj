#!/usr/bin/env python3
"""
4차 사이클 데이터셋 구성. (4th_plan.md ⑧)

고정 val = FT-1000 run의 val 299장 (3rd/results/FT-1000/data/val_annotations.csv).
각 학습 세트 train_N = (해당 라운드까지 누적 이미지) − 고정 val.

생성:
  4th/dataset/
  ├── split_manifest.json          고정 val 299 파일명+md5 (분할의 정본)
  ├── val/        {images/, annotations.json}   299장
  ├── test200/    {images/, annotations.json}   200장 (3rd/dataset/test200 복사)
  ├── train_206/  {images/, annotations.json}   01+02 − val
  ├── train_306/  {images/, annotations.json}   01~03 − val
  └── train_506/  {images/, annotations.json}   01~04 − val

pure stdlib. repo 코드 수정 없음. 실행: python make_dataset.py
"""
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

R = Path("/workspace/Grounding-Dino-FineTuning")
ROUND_ROOT = R / "3rd/dataset"                    # 01,02는 2nd/dataset2로 가는 symlink
VAL_CSV = R / "3rd/results/FT-1000/data/val_annotations.csv"
TEST200 = R / "3rd/dataset/test200"
OUT = R / "4th/dataset"

ROUNDS = ["01", "02", "03", "04", "05", "06"]
CUM = {"train_206": ["01", "02"],
       "train_306": ["01", "02", "03"],
       "train_506": ["01", "02", "03", "04"]}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    # ── 1. 고정 val 목록 ────────────────────────────────────────────────────
    with open(VAL_CSV) as f:
        val_names = sorted({row["image_name"] for row in csv.DictReader(f)})
    assert len(val_names) == 299, f"val {len(val_names)}장 — 299장이어야 함"
    val_set = set(val_names)

    # ── 2. 라운드 COCO 전부 로드 (file_name 기준 dedup) ─────────────────────
    registry = {}          # file_name -> {img, anns, src_dir, rounds}
    categories = None
    for r in ROUNDS:
        for split in ["train", "val"]:
            d = ROUND_ROOT / r / split
            coco = json.loads((d / "annotations.json").read_text())
            if categories is None:
                categories = coco["categories"]
            anns_by_img = {}
            for a in coco["annotations"]:
                anns_by_img.setdefault(a["image_id"], []).append(a)
            for im in coco["images"]:
                fn = im["file_name"]
                boxes = sorted(tuple(a["bbox"]) for a in anns_by_img.get(im["id"], []))
                if fn in registry:
                    registry[fn]["rounds"].append(r)
                    if boxes != registry[fn]["boxes"]:
                        print(f"[WARN] 라운드 간 bbox 불일치: {fn} "
                              f"({registry[fn]['rounds']} vs {r}) — 먼저 등장한 것 사용")
                    continue
                registry[fn] = {"img": im, "anns": anns_by_img.get(im["id"], []),
                                "boxes": boxes, "src": d / "images" / fn, "rounds": [r]}

    union_all = set(registry)
    dupes = sorted(fn for fn, e in registry.items() if len(e["rounds"]) > 1)
    print(f"라운드 합집합 {len(union_all)}장 (라운드 간 중복 {len(dupes)}건)")
    for fn in dupes:
        print(f"  중복: {fn}  <- 라운드 {registry[fn]['rounds']}")

    # ── 3. 무결성 검사 ──────────────────────────────────────────────────────
    missing_val = val_set - union_all
    assert not missing_val, f"val인데 라운드에 없음: {sorted(missing_val)[:5]}"
    test_names = {im["file_name"] for im in
                  json.loads((TEST200 / "annotations.json").read_text())["images"]}
    leak = union_all & test_names
    assert not leak, f"라운드-test200 leakage: {sorted(leak)[:5]}"
    for fn, e in registry.items():
        if not e["src"].exists():
            sys.exit(f"[ERROR] 이미지 파일 없음: {e['src']}")

    # ── 4. 세트 구성 ────────────────────────────────────────────────────────
    def cum_names(rounds):
        return {fn for fn, e in registry.items() if e["rounds"][0] in rounds}

    sets = {"val": val_names}
    for name, rounds in CUM.items():
        members = sorted(cum_names(rounds) - val_set)
        sets[name] = members
    # train_1000은 학습 안 하지만 검증용으로 크기만 확인
    full_train = cum_names(ROUNDS) - val_set
    print(f"검증: 전체 train(1000셋 기준) {len(full_train)}장 + val 299장 "
          f"= {len(full_train) + 299}장")

    # ── 5. 출력 ─────────────────────────────────────────────────────────────
    if OUT.exists():
        sys.exit(f"[ERROR] {OUT} 이미 존재 — 지우고 다시 실행할 것 (덮어쓰기 안 함)")
    for name, members in sets.items():
        d = OUT / name
        (d / "images").mkdir(parents=True)
        images, anns = [], []
        img_id = ann_id = 1
        for fn in members:
            e = registry[fn]
            shutil.copy2(e["src"], d / "images" / fn)
            im = dict(e["img"]); im["id"] = img_id
            images.append(im)
            for a in e["anns"]:
                a2 = dict(a); a2["id"] = ann_id; a2["image_id"] = img_id
                anns.append(a2); ann_id += 1
            img_id += 1
        (d / "annotations.json").write_text(json.dumps(
            {"images": images, "annotations": anns, "categories": categories}))
        print(f"{name}: {len(images)}장 / {len(anns)}박스")

    shutil.copytree(TEST200, OUT / "test200")
    print(f"test200: {len(list((OUT/'test200/images').glob('*.jpg')))}장 복사")

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": "3rd/results/FT-1000/data/val_annotations.csv (FT-1000 run val, seed 42)",
        "fixed_val": [{"file_name": fn, "md5": md5(OUT / "val/images" / fn)}
                      for fn in val_names],
        "sets": {k: len(v) for k, v in sets.items()},
        "round_duplicates": dupes,
    }
    (OUT / "split_manifest.json").write_text(json.dumps(manifest, indent=2))

    # 최종 leakage 재확인 (실제 산출 폴더 기준)
    for name in CUM:
        tr = {p.name for p in (OUT / name / "images").iterdir()}
        assert not tr & val_set, f"{name} ∩ val leakage!"
        assert not tr & test_names, f"{name} ∩ test200 leakage!"
    print("leakage 최종 검사 통과 (train ∩ val = 0, train ∩ test200 = 0)")
    print("완료:", OUT)


if __name__ == "__main__":
    main()
