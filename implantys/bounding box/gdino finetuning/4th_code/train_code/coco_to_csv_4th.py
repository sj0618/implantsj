#!/usr/bin/env python3
"""
4차 전용 COCO -> repo CSV 변환. 2nd/train_code/coco_to_csv.py의 4차판.

원본과 다른 점은 **데이터를 지정하는 방식뿐**이다:
  원본 : --data 01 02      (라운드 폴더를 병합. train/val 모두 라운드별로 합침)
  4차  : --train-dir / --val-dir  (폴더를 그대로 읽음)

라운드 병합은 make_dataset.py에서 이미 끝났고, val은 세 run이 공유하는 고정
299장이라 "라운드 병합" 개념 자체가 필요 없다. 출력 형식·검증 항목은 원본과 동일.

leakage 검사는 원본이 test48을 봤던 것과 달리 **test200(평가에 실제로 쓰는 세트)**
을 본다. --test-json으로 지정한다.

출력 (repo config.yaml이 기대하는 구조):
  <out-root>/images/train/*.jpg  +  <out-root>/train_annotations.csv
  <out-root>/images/val/*.jpg    +  <out-root>/val_annotations.csv

사용:
  python coco_to_csv_4th.py --train-dir <...>/train_206 --val-dir <...>/val \
      --test-json <...>/test200/annotations.json --out-root <run>/data
"""

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

LABEL = "implant"


def rf_hash(name: str) -> str:
    """Roboflow 파일명의 rf.<hex>. 같은 이미지를 이름 변형과 무관하게 식별한다."""
    m = re.search(r"rf\.([0-9a-f]+)", Path(name).name)
    assert m, f"rf 해시를 찾을 수 없음: {name}"
    return m.group(1)


def load_dir(d: Path) -> list[dict]:
    """<dir>/{annotations.json, images/} 를 읽는다."""
    ann = d / "annotations.json"
    if not ann.exists():
        sys.exit(f"[ERROR] annotations.json 없음: {d}")
    coco = json.loads(ann.read_text())
    id2name = {im["id"]: im["file_name"] for im in coco["images"]}
    boxes_by_img: dict[str, list] = {im["file_name"]: [] for im in coco["images"]}
    for a in coco["annotations"]:
        boxes_by_img[id2name[a["image_id"]]].append(a["bbox"])

    recs = []
    for im in coco["images"]:
        src = d / "images" / im["file_name"]
        if not src.exists():
            sys.exit(f"[ERROR] 이미지 없음: {src}")
        recs.append({"file_name": im["file_name"], "hash": rf_hash(im["file_name"]),
                     "src": src, "boxes": boxes_by_img[im["file_name"]]})
    return recs


def write_split(recs: list[dict], out_root: Path, split: str) -> tuple[int, int]:
    img_out = out_root / "images" / split
    img_out.mkdir(parents=True, exist_ok=True)

    n_box = 0
    with open(out_root / f"{split}_annotations.csv", "w", newline="",
              encoding="utf-8") as f:
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
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--test-json", default=None,
                    help="leakage 검사용 test GT. 4차는 test200/annotations.json")
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()

    train_dir, val_dir, out_root = Path(a.train_dir), Path(a.val_dir), Path(a.out_root)

    if out_root.exists():
        shutil.rmtree(out_root)      # 이전 run 잔여물이 섞이면 안 된다
    out_root.mkdir(parents=True)

    print(f"train: {train_dir}")
    print(f"val  : {val_dir}")
    loaded = {"train": load_dir(train_dir), "val": load_dir(val_dir)}

    # ── 검증 (원본과 동일 항목) ──────────────────────────────────────────────
    tr_h = {r["hash"] for r in loaded["train"]}
    va_h = {r["hash"] for r in loaded["val"]}
    assert len(tr_h) == len(loaded["train"]), "train 내부 중복 이미지"
    assert len(va_h) == len(loaded["val"]), "val 내부 중복 이미지"
    assert not (tr_h & va_h), f"train ∩ val 겹침: {len(tr_h & va_h)}장"

    if a.test_json and Path(a.test_json).exists():
        te_h = {rf_hash(i["file_name"])
                for i in json.loads(Path(a.test_json).read_text())["images"]}
        leak = (tr_h | va_h) & te_h
        assert not leak, f"LEAKAGE! train/val ∩ test = {len(leak)}장 {sorted(leak)[:3]}"
        print(f"leakage 검사: train/val({len(tr_h | va_h)}) ∩ test({len(te_h)}) = 0 ✓")
    else:
        print("[warn] test GT 없음 — leakage 검사 건너뜀")

    for s, recs in loaded.items():
        names = [r["file_name"] for r in recs]
        assert len(names) == len(set(names)), f"{s}: 파일명 충돌"

    # ── 쓰기 ────────────────────────────────────────────────────────────────
    n = 0
    for s in ["train", "val"]:
        n_img, n_box = write_split(loaded[s], out_root, s)
        n += n_img
        n_jpg = len(list((out_root / "images" / s).glob("*.jpg")))
        assert n_jpg == n_img, f"{s}: 복사된 이미지 수 불일치 {n_jpg} != {n_img}"
        print(f"  [{s:5s}] images={n_img} boxes={n_box} -> {s}_annotations.csv")

    # 고정 val 확인용 해시 — 세 run이 정말 같은 val을 썼는지 나중에 대조한다
    val_sig = hashlib.sha256(
        "\n".join(sorted(r["file_name"] for r in loaded["val"])).encode()
    ).hexdigest()[:16]
    print(f"\n[done] train+val {n}장 -> {out_root}")
    print(f"[val 고정 확인] val {len(loaded['val'])}장 · split_hash {val_sig}")


if __name__ == "__main__":
    main()
