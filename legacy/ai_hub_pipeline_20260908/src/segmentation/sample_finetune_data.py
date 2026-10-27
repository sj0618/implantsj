"""
① 원본 train 데이터셋에서 세그멘테이션 finetuning용 데이터 n장 샘플링
   - 각 클래스에서 1장씩 번갈아 뽑음 → 클래스 불균형 해소
   (원본: implant_ny/make_seg_model/seg_train_split.py)

실행: python sample_finetune_data.py --config configs/02_segmentation.yaml
"""
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_utils import load_config_from_cli


def main():
    cfg = load_config_from_cli(__doc__)
    scfg = cfg["sample"]

    src = Path(scfg["src_dir"])
    dst = Path(scfg["dst_dir"])
    n = scfg["n"]
    exts = {e.lower() for e in scfg["exts"]}

    random.seed(scfg["seed"])

    classes = sorted([d for d in src.iterdir() if d.is_dir()])
    files = {}

    for c in classes:
        # 클래스 폴더 바로 아래 or images/ 하위 모두 지원
        img_root = c / "images" if (c / "images").is_dir() else c
        files[c.name] = [
            f for f in img_root.rglob("*")
            if f.is_file() and f.suffix.lower() in exts
        ]
        random.shuffle(files[c.name])

    selected = {c.name: [] for c in classes}

    # 클래스별 라운드로빈으로 n장 채움
    while sum(map(len, selected.values())) < n:
        added = False
        for c in classes:
            name = c.name
            if files[name] and sum(map(len, selected.values())) < n:
                selected[name].append(files[name].pop())
                added = True
        if not added:
            break

    for name, imgs in selected.items():
        out = dst / name
        out.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy2(img, out / img.name)
        print(name, len(imgs))

    print("총 선택:", sum(map(len, selected.values())))


if __name__ == "__main__":
    main()
