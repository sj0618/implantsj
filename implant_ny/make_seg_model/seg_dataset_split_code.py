'''
seg_dataset을 7:2:1로 split하는 코드 
'''


import os, shutil,random,glob
from pathlib import Path

random.seed(42)
SRC="seg_dataset_new"
OUT="seg_dataset_split"

for cls in os.listdir(f"{SRC}/images"):
    files=[f for f in os.listdir(f"{SRC}/images/{cls}") if f.endswith(".jpg")]
    if not files:
        continue
    random.shuffle(files)
    n=len(files)
    splits={"train":files[:int(n*0.7)],"val":files[int(n*0.7):int(n*0.9)],"test":files[int(n*0.9):]}

    for split, names in splits.items():
        for folder in ["images","masks"]:
            Path(f"{OUT}/{split}/{folder}/{cls}").mkdir(parents=True,exist_ok=True)
        for f in names:
            shutil.copy2(f"{SRC}/images/{cls}/{f}",f"{OUT}/{split}/images/{cls}/{f}")
            stem=Path(f).stem
            masks=glob.glob(f"{SRC}/masks/{cls}/{stem}_mask_*.png")
            if masks:
                mask_name=os.path.basename(masks[0])
                shutil.copy2(masks[0],f"{OUT}/{split}/masks/{cls}/{mask_name}")
            else:
                print(f"마스크없음:{cls}/{f}")

                
           

print("Done")