import argparse,cv2
import numpy as np, pandas as pd
from pathlib import Path

p= argparse.ArgumentParser()
p.add_argument('--manifest', required=True)
p.add_argument('--mask-root', required=True)
p.add_argument('--out-root', required=True)
p.add_argument('--out-manifest', required=True)
p.add_argument('--size',type=int, default=224)
p.add_argument('--outer',type=int, default=10)
p.add_argument('--inner', type=int,default=10)
a=p.parse_args()

idx= {q.name[: -len('_pred.png')]: q for q in Path(a.mask_root).rglob('*_pred.png')}
out= Path(a.out_root); out.mkdir(parents=True,exist_ok=True)
ko=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*a.outer+1, 2*a.outer+1))
ki=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*a.inner+1, 2*a.inner+1))
df=pd.read_csv(a.manifest); paths=[]

for _, r in df.iterrows():
    q=idx.get(Path(r['path']).stem)
    if q is None:
        paths.append(''); continue
    m=cv2.imread(str(q),0)
    m=cv2.resize(m,(a.size, a.size), interpolation=cv2.INTER_NEAREST)
    m=(m>127).astype(np.uint8)*255
    band=cv2.subtract(cv2.dilate(m,ko), cv2.erode(m,ki))
    op=out/ f"{r.get('split','')}_{r['label']}_{Path(r['path']).stem}_band.png"
    cv2.imwrite(str(op),band); paths.append(str(op.resolve()))
df['mask_path']=paths
df.to_csv(a.out_manifest,index=False)
print('done.missing mask:',paths.count(''))