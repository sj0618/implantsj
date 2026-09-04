
''' 
원본 train 데이터셋에서 세그멘테이션 용 데이터 1000장 뽑아내는 코드 
-각 클래스에서 1장씩 번걸아 뽑음-> 클래스 불균형 해소

'''

from pathlib import Path
import random,shutil

src=Path("/home/ubuntu/workspace/data/datas/test")
dst=Path("/home/ubuntu/workspace/seg_test_sample_2")
N=50
exts={".jpg"}

random.seed(42)

classes= sorted([d for d in src.iterdir() if d.is_dir()])
files={}

for c in classes:
    files[c.name]=[
        f for f in (c/"images").rglob("*")
        if f.is_file() and f.suffix.lower() in exts
    ]
    random.shuffle(files[c.name])

selected= {c.name: [] for c in classes}

while sum(map(len, selected.values())) <N:
    added=False

    for c in classes:
        name=c.name

        if files[name] and sum(map(len,selected.values()))< N:
            selected[name].append(files[name].pop())
            added=True
    
    if not added:
        break

for name,imgs in selected.items():
    out=dst/name
    out.mkdir(parents=True,exist_ok=True)

    for img in imgs:
        shutil.copy2(img,out/img.name)
    
    print(name,len(imgs))

print("총 선택:",sum(map(len,selected.values())) )