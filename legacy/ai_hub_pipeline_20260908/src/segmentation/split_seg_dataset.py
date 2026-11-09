"""
④ (이미지, 마스크) 쌍을 train/val/test 7:2:1로 분할
   - 사람이 선별을 마친 image:mask 1:1 폴더(src_dir)를 입력으로 받음
     구조: src_dir/images/<cls>/*.jpg, src_dir/masks/<cls>/<stem>_mask_*.png
   (원본: implant_ny/make_seg_model/seg_dataset_split_code.py)

실행: python split_seg_dataset.py --config configs/02_segmentation.yaml
"""
import glob
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_utils import load_config_from_cli


def main():
    cfg = load_config_from_cli(__doc__)
    scfg = cfg["split"]

    random.seed(scfg["seed"])
    src = scfg["src_dir"]
    out = scfg["out_dir"]
    r = scfg["ratios"]

    missing = 0
    for cls in os.listdir(f"{src}/images"):
        files = [f for f in os.listdir(f"{src}/images/{cls}") if f.endswith(".jpg")]
        if not files:
            continue
        random.shuffle(files)
        n = len(files)
        n_train = int(n * r["train"])
        n_val = int(n * (r["train"] + r["val"]))
        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_val],
            "test": files[n_val:],
        }

        for split, names in splits.items():
            for folder in ["images", "masks"]:
                Path(f"{out}/{split}/{folder}/{cls}").mkdir(parents=True, exist_ok=True)
            for f in names:
                shutil.copy2(f"{src}/images/{cls}/{f}", f"{out}/{split}/images/{cls}/{f}")
                stem = Path(f).stem
                masks = glob.glob(f"{src}/masks/{cls}/{stem}_mask_*.png")
                if masks:
                    mask_name = os.path.basename(masks[0])
                    shutil.copy2(masks[0], f"{out}/{split}/masks/{cls}/{mask_name}")
                else:
                    print(f"마스크없음:{cls}/{f}")
                    missing += 1

    print(f"Done (마스크 없는 이미지: {missing})")


if __name__ == "__main__":
    main()
