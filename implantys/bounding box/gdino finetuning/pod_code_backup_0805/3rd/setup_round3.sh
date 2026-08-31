#!/usr/bin/env bash
# 3rd 라운드 1회 셋업: 코드복제·BASE패치·01/02/test 심링크·03 train-only 빌드
set -euo pipefail
R=/workspace/Grounding-Dino-FineTuning; D=$R/3rd/dataset; PY=$R/implant-env/bin/python

# 1) 코드 복제 + 패치 (2nd 원본은 안 건드림)
cp -rn $R/2nd/train_code $R/3rd/train_code
sed -i 's#REPO / "2nd"#REPO / "3rd"#g' $R/3rd/train_code/train.py $R/3rd/train_code/coco_to_csv.py
sed -i 's#BASE / "dataset2"#BASE / "dataset"#g' $R/3rd/train_code/coco_to_csv.py

# 2) 01/02/test 심링크 (복사 X)
ln -sfn $R/2nd/dataset2/01 $D/01
ln -sfn $R/2nd/dataset2/02 $D/02
ln -sfn $R/2nd/dataset2/test $D/test

# 3) 03 train-only 빌드 (result.json 라벨, 중복 basename은 박스 많은 id 하나만)
$PY - <<'PYEOF'
import json,os,shutil
from collections import defaultdict
D="/workspace/Grounding-Dino-FineTuning/3rd/dataset"
res=json.load(open(f"{D}/3rd_total/result.json"))
nid=defaultdict(list); sz={}
for im in res["images"]:
    bn=os.path.basename(im["file_name"]); nid[bn].append(im["id"]); sz[im["id"]]=(im["width"],im["height"])
ab=defaultdict(list)
for a in res["annotations"]: ab[a["image_id"]].append(a)
pick={bn:sorted(i,key=lambda x:(-len(ab[x]),x))[0] for bn,i in nid.items()}
files=sorted(f for f in os.listdir(f"{D}/03/images") if f.lower().endswith((".jpg",".jpeg",".png")))
os.makedirs(f"{D}/03/train/images",exist_ok=True)
IMG=[];ANN=[];aid=1
for iid,bn in enumerate(files,1):
    w,h=sz[pick[bn]]; IMG.append({"width":w,"height":h,"id":iid,"file_name":bn})
    for a in ab[pick[bn]]:
        x,y,bw,bh=a["bbox"]
        ANN.append({"id":aid,"image_id":iid,"category_id":0,"segmentation":[],"bbox":[x,y,bw,bh],"ignore":0,"iscrowd":0,"area":a.get("area",bw*bh)}); aid+=1
    shutil.copy(f"{D}/03/images/{bn}",f"{D}/03/train/images/{bn}")
json.dump({"images":IMG,"annotations":ANN,"categories":[{"id":0,"name":"implant"}]},open(f"{D}/03/train/annotations.json","w"),ensure_ascii=False)
print(f"03/train: img={len(IMG)} box={len(ANN)} (val 없음)")
PYEOF

echo "=== 폴더 확인 ==="
for f in 01 02 03 04 05 06 test; do echo "$f: $(ls $D/$f | tr '\n' ' ')"; done
echo "=== setup 완료 ==="
