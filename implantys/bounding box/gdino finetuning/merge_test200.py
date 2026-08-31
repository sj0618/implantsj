#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_test200.py

검수 완료본(test2.json, Label Studio export 152장)을 COCO GT로 역변환하고,
기존 test 48장(dataset2/test/annotations.json)과 합쳐
200장짜리 단일 정답셋(test200/)을 만든다.

출력:
  test200/images/            -> 200장 (48 + 152)
  test200/annotations.json   -> 단일 COCO GT (image id 1..200 재부여)

COCO 포맷은 기존 dataset2/test/annotations.json 과 동일:
  categories [{id:0,name:implant}]
  images {width,height,id,file_name}
  annotations {id,image_id,category_id:0,segmentation:[],bbox:[x,y,w,h]px,ignore:0,iscrowd:0,area}
"""
import json, os, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
TEST48_JSON = os.path.join(HERE, "dataset2", "test", "annotations.json")
TEST48_IMG  = os.path.join(HERE, "dataset2", "test", "images")
TEST2_LS    = os.path.join(HERE, "test2", "test2.json")           # 검수 완료 export
TEST2_IMG   = os.path.join(HERE, "test2", "images")

OUT_DIR = os.path.join(HERE, "test200")
OUT_IMG = os.path.join(OUT_DIR, "images")
OUT_JSON = os.path.join(OUT_DIR, "annotations.json")

LABEL = "implant"


def ls_task_boxes(task):
    """LS export task -> [(x,y,w,h)px, ...]. 사람이 검수한 annotations[].result 사용."""
    anns = task.get("annotations") or []
    if not anns:
        return [], None, None
    # 여러 annotation이면 최신(updated_at) 하나 선택
    ann = sorted(anns, key=lambda a: a.get("updated_at", ""))[-1]
    # 좌표가 비어있는(깨진) 박스는 같은 id의 prediction 좌표로 복구한다.
    pred_by_id = {}
    for r in (ann.get("prediction") or {}).get("result", []):
        pred_by_id[r.get("id")] = r.get("value", {})
    boxes = []
    W = H = None
    skipped = 0
    recovered = 0
    for r in ann.get("result", []):
        if r.get("type") != "rectanglelabels":
            continue
        W = int(r["original_width"]); H = int(r["original_height"])
        v = r["value"]
        if any(v.get(k) is None for k in ("x", "y", "width", "height")):
            pv = pred_by_id.get(r.get("id"), {})
            if all(pv.get(k) is not None for k in ("x", "y", "width", "height")):
                v = pv               # prediction 좌표로 복구
                recovered += 1
            else:
                skipped += 1         # prediction도 없으면 제외
                continue
        x = v["x"] / 100.0 * W
        y = v["y"] / 100.0 * H
        w = v["width"] / 100.0 * W
        h = v["height"] / 100.0 * H
        # 이미지 경계로 클램프
        x = max(0.0, min(x, W)); y = max(0.0, min(y, H))
        w = max(0.0, min(w, W - x)); h = max(0.0, min(h, H - y))
        boxes.append((x, y, w, h))
    return boxes, W, H, skipped, recovered


def task_filename(task):
    url = task["data"]["image"]
    return url.split("?d=images/")[-1].split("/")[-1]


def main():
    os.makedirs(OUT_IMG, exist_ok=True)

    images, annotations = [], []
    next_img_id = 1
    next_ann_id = 1
    seen_files = {}

    def add_image(fname, W, H, src_path):
        nonlocal next_img_id
        if fname in seen_files:
            raise SystemExit(f"[중복 파일명] {fname} — 두 세트에 같은 파일명 존재")
        iid = next_img_id; next_img_id += 1
        seen_files[fname] = iid
        images.append({"width": int(W), "height": int(H), "id": iid, "file_name": fname})
        dst = os.path.join(OUT_IMG, fname)
        if not os.path.exists(dst):
            shutil.copy2(src_path, dst)
        return iid

    def add_ann(iid, box):
        nonlocal next_ann_id
        x, y, w, h = box
        annotations.append({
            "id": next_ann_id, "image_id": iid, "category_id": 0,
            "segmentation": [], "bbox": [x, y, w, h],
            "ignore": 0, "iscrowd": 0, "area": w * h,
        })
        next_ann_id += 1

    # ---- 1) 기존 test 48 ----------------------------------------------
    t48 = json.load(open(TEST48_JSON, encoding="utf-8"))
    old2new = {}
    for im in t48["images"]:
        src = os.path.join(TEST48_IMG, im["file_name"])
        if not os.path.exists(src):
            raise SystemExit(f"[test48 이미지 없음] {src}")
        old2new[im["id"]] = add_image(im["file_name"], im["width"], im["height"], src)
    n48_box = 0
    for a in t48["annotations"]:
        add_ann(old2new[a["image_id"]], a["bbox"]); n48_box += 1

    # ---- 2) 검수 완료 152 (test2.json) --------------------------------
    ls = json.load(open(TEST2_LS, encoding="utf-8"))
    n152_box = 0
    skipped_files = []
    recovered_files = []
    for task in ls:
        fname = task_filename(task)
        src = os.path.join(TEST2_IMG, fname)
        if not os.path.exists(src):
            raise SystemExit(f"[test2 이미지 없음] {src}")
        boxes, W, H, skipped, recovered = ls_task_boxes(task)
        if skipped:
            skipped_files.append((fname, skipped))
        if recovered:
            recovered_files.append((fname, recovered))
        if W is None:  # 박스 0개인 이미지 -> 원본에서 크기 읽기
            from PIL import Image
            with Image.open(src) as im:
                W, H = im.size
        iid = add_image(fname, W, H, src)
        for b in boxes:
            add_ann(iid, b); n152_box += 1

    # ---- 3) 저장 ------------------------------------------------------
    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": 0, "name": LABEL}]}
    json.dump(coco, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False)

    n_files = len([f for f in os.listdir(OUT_IMG)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    print(f"[test48]  images 48  boxes {n48_box}")
    print(f"[test152] images {len(ls)}  boxes {n152_box}")
    print(f"[merged]  images {len(images)}  boxes {len(annotations)}")
    print(f"[files]   test200/images 실제 파일 {n_files}")
    print(f"[out]     {OUT_JSON}")
    if recovered_files:
        print("\n[복구] 좌표가 비어있던 박스를 같은 id의 prediction 좌표로 되살림:")
        for fn, k in recovered_files:
            print(f"   {fn}: {k}개 복구")
    if skipped_files:
        print("\n[경고] prediction도 없어 제외한 깨진 박스 — LS에서 재확인 필요:")
        for fn, k in skipped_files:
            print(f"   {fn}: {k}개 제외")


if __name__ == "__main__":
    main()
