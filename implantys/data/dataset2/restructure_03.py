"""
04/05/06 폴더를 01/02 구조(train/val + COCO annotations.json)로 재구성.

- GT: 3rd_total/result.json (사람 검수본, COCO)
- split: 70/30 (train = round(0.7*n)), seed 고정
- 이미지: images/ -> train/images, val/images 로 이동 후 빈 images/ 제거
- adin974(라벨 없음) 은 06/_excluded_unlabeled/ 로 빼고 학습에서 제외
- 중복 basename(adin1050/osstem763)은 박스 많은 image_id 하나만 채택
"""
import json, os, random, shutil
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(BASE, "3rd_total", "result.json")
FOLDERS = ["03"]
EXCLUDE = {"adin974_jpg.rf.a392000e0c752ae953f9c87ff86c5f41.jpg"}
SEED = 42
TRAIN_RATIO = 0.7
IMG_EXT = (".jpg", ".jpeg", ".png")

# ---- GT 로드 (basename -> (w,h,[bbox...])) ----
res = json.load(open(RESULT, encoding="utf-8"))
name_ids = defaultdict(list)      # basename -> [image_id...]
imgmeta = {}                      # image_id -> (w,h,basename)
for im in res["images"]:
    bn = os.path.basename(im["file_name"])
    name_ids[bn].append(im["id"])
    imgmeta[im["id"]] = (im["width"], im["height"], bn)
anns_by_id = defaultdict(list)
for a in res["annotations"]:
    anns_by_id[a["image_id"]].append(a)

# 중복 basename -> 박스 많은 image_id 하나만 (동수면 낮은 id)
chosen_id = {}
for bn, ids in name_ids.items():
    chosen_id[bn] = sorted(ids, key=lambda i: (-len(anns_by_id[i]), i))[0]

gt = {}   # basename -> (w, h, [ann_dict...])
for bn, iid in chosen_id.items():
    w, h, _ = imgmeta[iid]
    gt[bn] = (w, h, anns_by_id[iid])

# ---- 폴더별 재구성 ----
def build_coco(names):
    images, annotations = [], []
    aid = 1
    for img_id, bn in enumerate(names, start=1):
        w, h, anns = gt[bn]
        images.append({"width": w, "height": h, "id": img_id, "file_name": bn})
        for a in anns:
            x, y, bw, bh = a["bbox"]
            annotations.append({
                "id": aid, "image_id": img_id, "category_id": 0,
                "segmentation": [], "bbox": [x, y, bw, bh],
                "ignore": 0, "iscrowd": 0,
                "area": a.get("area", bw * bh),
            })
            aid += 1
    return {"images": images, "annotations": annotations,
            "categories": [{"id": 0, "name": "implant"}]}

rnd = random.Random(SEED)
print(f"{'folder':<8}{'total':>7}{'excl':>6}{'train':>7}{'val':>6}{'tr_box':>8}{'val_box':>8}")
for folder in FOLDERS:
    fdir = os.path.join(BASE, folder)
    idir = os.path.join(fdir, "images")
    files = sorted(f for f in os.listdir(idir) if f.lower().endswith(IMG_EXT))

    excl = [f for f in files if f in EXCLUDE]
    use = [f for f in files if f not in EXCLUDE]
    # GT 없는 것도 방어적으로 제외 (이론상 없음)
    no_gt = [f for f in use if f not in gt]
    use = [f for f in use if f in gt]

    rnd.shuffle(use)
    n_train = round(len(use) * TRAIN_RATIO)
    split = {"train": use[:n_train], "val": use[n_train:]}

    box = {}
    for sp, names in split.items():
        sdir = os.path.join(fdir, sp, "images")
        os.makedirs(sdir, exist_ok=True)
        for bn in names:
            shutil.move(os.path.join(idir, bn), os.path.join(sdir, bn))
        coco = build_coco(names)
        json.dump(coco, open(os.path.join(fdir, sp, "annotations.json"), "w", encoding="utf-8"),
                  ensure_ascii=False)
        box[sp] = len(coco["annotations"])

    # 제외 이미지 보존
    for f in excl + no_gt:
        xdir = os.path.join(fdir, "_excluded_unlabeled")
        os.makedirs(xdir, exist_ok=True)
        src = os.path.join(idir, f)
        if os.path.exists(src):
            shutil.move(src, os.path.join(xdir, f))

    # 빈 images/ 제거
    if os.path.isdir(idir) and not os.listdir(idir):
        os.rmdir(idir)

    print(f"{folder:<8}{len(files):>7}{len(excl)+len(no_gt):>6}"
          f"{len(split['train']):>7}{len(split['val']):>6}{box['train']:>8}{box['val']:>8}")

print("\n완료.")
