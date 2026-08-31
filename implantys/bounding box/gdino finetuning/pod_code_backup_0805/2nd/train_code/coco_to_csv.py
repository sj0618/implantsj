#!/usr/bin/env python3
"""
dataset2의 COCO 라벨을 repo(Asad-Ismail/Grounding-Dino-FineTuning) CSV 형식으로 변환.

여러 라운드 폴더를 읽어 합칠 수 있다. dataset2에는 라운드별 원본만 두고,
병합은 이 변환 시점에만 일어난다. 03·04 라운드가 늘어도 인자만 추가하면 된다.

  FT-106 :  --data 01
  FT-206 :  --data 01 02
  FT-xxx :  --data 01 02 03 ...

repo CSV 컬럼: image_name, bbox_x, bbox_y, bbox_width, bbox_height, label_name
box는 COCO와 동일한 [x, y, w, h] 픽셀 좌표라 변환이 필요 없다.

출력 (repo의 config.yaml이 기대하는 구조):
  <out-root>/images/train/*.jpg  +  <out-root>/train_annotations.csv
  <out-root>/images/val/*.jpg    +  <out-root>/val_annotations.csv
  <out-root>/images/test/*.jpg   +  <out-root>/test_annotations.csv

사용:
  python coco_to_csv.py --data 01 02 \
  (경로 기본값은 2nd 기준으로 잡혀 있어 그냥 --data만 주면 된다)
"""

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path("/workspace/Grounding-Dino-FineTuning")
BASE = REPO / "2nd"

LABEL = "implant"


def rf_hash(name: str) -> str:
    m = re.search(r"rf\.([0-9a-f]+)", Path(name).name)
    assert m, f"rf 해시를 찾을 수 없음: {name}"
    return m.group(1)


def load_split(data_root: Path, rounds: list[str], split: str) -> list[dict]:
    """여러 라운드의 같은 split을 읽어 합친다. test는 라운드와 무관하게 단일."""
    recs = []
    dirs = [data_root / "test"] if split == "test" else [data_root / r / split for r in rounds]

    for d in dirs:
        ann = d / "annotations.json"
        if not ann.exists():
            print(f"  [skip] {d} — annotations.json 없음")
            continue
        coco = json.loads(ann.read_text())
        id2name = {im["id"]: im["file_name"] for im in coco["images"]}
        boxes_by_img: dict[str, list] = {im["file_name"]: [] for im in coco["images"]}
        for a in coco["annotations"]:
            boxes_by_img[id2name[a["image_id"]]].append(a["bbox"])
        for im in coco["images"]:
            src = d / "images" / im["file_name"]
            if not src.exists():
                sys.exit(f"[ERROR] 이미지 없음: {src}")
            recs.append({
                "file_name": im["file_name"],
                "hash": rf_hash(im["file_name"]),
                "src": src,
                "boxes": boxes_by_img[im["file_name"]],
            })
    return recs


def write_split(recs: list[dict], out_root: Path, split: str) -> tuple[int, int]:
    img_out = out_root / "images" / split
    img_out.mkdir(parents=True, exist_ok=True)

    n_box = 0
    csv_path = out_root / f"{split}_annotations.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_name", "bbox_x", "bbox_y",
                    "bbox_width", "bbox_height", "label_name"])
        for r in sorted(recs, key=lambda r: r["file_name"]):
            shutil.copy2(r["src"], img_out / r["file_name"])
            for x, y, bw, bh in r["boxes"]:
                w.writerow([r["file_name"], int(round(x)), int(round(y)),
                            int(round(bw)), int(round(bh)), LABEL])
                n_box += 1
    return len(recs), n_box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True,
                    help="합칠 라운드 폴더명. 예: --data 01 02")
    ap.add_argument("--data-root", default=str(BASE / "dataset2"))
    ap.add_argument("--out-root", default=str(REPO / "multimodal-data/implant_dataset"))
    ap.add_argument("--include-test", action="store_true",
                    help="test도 변환한다. 학습 시에는 불필요 (RULES.md 2번)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)

    for r in args.data:
        if not (data_root / r).is_dir():
            sys.exit(f"[ERROR] 라운드 폴더 없음: {data_root / r}")

    if out_root.exists():
        shutil.rmtree(out_root)      # 이전 라운드 잔여물이 섞이면 안 된다
    out_root.mkdir(parents=True)

    print(f"라운드: {' + '.join(args.data)}")
    splits = ["train", "val"] + (["test"] if args.include_test else [])
    loaded = {s: load_split(data_root, args.data, s) for s in splits}

    # ── 검증 ─────────────────────────────────────────────────────────────────
    tr_h = {r["hash"] for r in loaded["train"]}
    va_h = {r["hash"] for r in loaded["val"]}
    assert len(tr_h) == len(loaded["train"]), "train 내부 중복 이미지"
    assert len(va_h) == len(loaded["val"]), "val 내부 중복 이미지"
    assert not (tr_h & va_h), f"train ∩ val 겹침: {len(tr_h & va_h)}장"

    # test leakage: test 폴더를 변환하지 않더라도 항상 검사한다 (RULES.md 2번)
    test_ann = data_root / "test" / "annotations.json"
    if test_ann.exists():
        te_h = {rf_hash(i["file_name"])
                for i in json.loads(test_ann.read_text())["images"]}
        leak = (tr_h | va_h) & te_h
        assert not leak, f"LEAKAGE! train/val ∩ test = {len(leak)}장 {sorted(leak)[:3]}"
        print(f"leakage 검사: train/val({len(tr_h | va_h)}) ∩ test({len(te_h)}) = 0 ✓")
    else:
        print("[warn] test/annotations.json 없음 — leakage 검사 건너뜀")

    # 파일명 충돌 (라운드가 달라도 같은 이름이면 복사 시 덮어써짐)
    for s in splits:
        names = [r["file_name"] for r in loaded[s]]
        assert len(names) == len(set(names)), f"{s}: 파일명 충돌"

    # ── 쓰기 ─────────────────────────────────────────────────────────────────
    total = {}
    for s in splits:
        n_img, n_box = write_split(loaded[s], out_root, s)
        total[s] = (n_img, n_box)
        n_jpg = len(list((out_root / "images" / s).glob("*.jpg")))
        assert n_jpg == n_img, f"{s}: 복사된 jpg {n_jpg} != {n_img}"
        print(f"  [{s:5}] images={n_img:3} boxes={n_box:3} -> {s}_annotations.csv")

    n = sum(v[0] for k, v in total.items() if k != "test")
    print(f"\n[done] train+val {n}장 -> {out_root}")


if __name__ == "__main__":
    main()
