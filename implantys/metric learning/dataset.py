#!/usr/bin/env python3
"""
split.json 로딩 + 전처리 파이프라인.

전처리 (모든 조건에 동일 적용 — 철칙)
  왜곡 리사이즈 224x224  : Resize((224,224)) 는 종횡비를 무시하고 강제로 맞춘다.
                           실측 종횡비가 0.20 ~ 4.23 (21배 범위)이라 letterbox 는
                           패딩 비율 편차가 임베딩을 오염시킬 위험이 더 크다고 보고
                           왜곡을 택했다. 대신 모든 이미지에 일관 적용된다.
  정규화                 : DINOv2 사전학습 통계 (ImageNet mean/std)

augmentation 은 full FT 조건에서만 선택적으로 켠다 (§9 미결정 5번).
좌우 반전은 기본 제외 — 나사산이 나선이라 투영 상에서 기울기가 뒤집히고,
그게 제조사 신호일 수 있어 라벨을 훼손할 가능성이 있다.
"""

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_ROOT = Path(__file__).parent
SPLIT_JSON = _ROOT / "split.json"


# ---------------------------------------------------------------- split.json


def load_split(path=SPLIT_JSON):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} 없음. 먼저 `python make_split.py` 를 실행할 것.")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def resolve_data_dir(meta, split_json=SPLIT_JSON, override=None):
    """이미지 루트를 찾는다.

    split.json 에는 만들 때의 절대경로가 박혀 있다. 그 경로가 그대로 있으면 쓰고,
    없으면(=다른 머신으로 옮겼으면) 이름으로 후보를 뒤진다.
    """
    if override:
        return Path(override)
    cfg = meta["config"]
    name = cfg.get("dataset_name") or Path(cfg["data_dir"]).name

    candidates = [
        Path(cfg["data_dir"]),                  # split.json 을 만든 경로
        Path(split_json).parent / name,         # 코드와 같은 폴더
        Path(split_json).parent.parent / "data" / name,
        Path("/workspace/data") / name,
    ]
    for p in candidates:
        if p.is_dir():
            return p
    raise FileNotFoundError(
        "이미지 폴더를 찾지 못했다. 찾아본 곳:\n"
        + "\n".join(f"  {p}" for p in candidates)
        + "\n--data-dir 로 make_split.py 를 다시 실행하거나 데이터를 옮길 것.")


def resolve_classes(meta, scope):
    """scope: 'all'(13개) | 'open'(9개) | 'closed'(4개) | 리스트."""
    if isinstance(scope, (list, tuple)):
        return sorted(scope)
    if scope == "all":
        return sorted(meta["classes"])
    if scope == "open":
        return sorted(meta["open_classes"])
    if scope == "closed":
        return sorted(meta["closed_classes"])
    raise ValueError(f"알 수 없는 scope: {scope}")


# ---------------------------------------------------------------- transform


def build_transform(train_aug=False):
    ops = [transforms.Resize((IMG_SIZE, IMG_SIZE))]  # 튜플 = 왜곡 리사이즈
    if train_aug:
        ops += [
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)


# ---------------------------------------------------------------- dataset


class ImplantDataset(Dataset):
    """(image_tensor, label_idx, rel_path) 를 돌려준다.

    label_idx 는 `classes` 리스트에서의 위치. 9개로 학습한 모델이라도
    임베딩 추출 때는 13개 전부를 훑어야 하므로 classes 를 밖에서 주입한다.
    """

    def __init__(self, split, scope="all", meta=None, split_json=SPLIT_JSON,
                 train_aug=False, data_dir=None):
        self.meta = meta if meta is not None else load_split(split_json)
        self.split = split
        self.classes = resolve_classes(self.meta, scope)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.data_dir = resolve_data_dir(self.meta, split_json, data_dir)
        self.transform = build_transform(train_aug)

        self.samples = []
        for c in self.classes:
            for rel in self.meta["splits"][c][split]:
                self.samples.append((rel, self.class_to_idx[c]))
        if not self.samples:
            raise ValueError(f"샘플 없음: split={split}, scope={scope}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        rel, label = self.samples[i]
        img = Image.open(self.data_dir / rel).convert("RGB")
        return self.transform(img), label, rel

    def counts(self):
        out = {c: 0 for c in self.classes}
        for rel, label in self.samples:
            out[self.classes[label]] += 1
        return out


def make_loader(split, scope="all", batch_size=32, shuffle=False, num_workers=4,
                train_aug=False, meta=None, split_json=SPLIT_JSON):
    ds = ImplantDataset(split, scope=scope, meta=meta, split_json=split_json,
                        train_aug=train_aug)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, drop_last=False,
    )
    return ds, loader


if __name__ == "__main__":
    meta = load_split()
    for scope in ("all", "open", "closed"):
        cls = resolve_classes(meta, scope)
        n = {s: sum(len(meta["splits"][c][s]) for c in cls)
             for s in ("train", "val", "test")}
        print(f"{scope:<7} 클래스 {len(cls):>2}개  "
              f"train {n['train']:>4}  val {n['val']:>3}  test {n['test']:>3}")
    print(f"\n이미지 경로: {resolve_data_dir(meta)}")
    print("샘플 로딩 테스트:")
    ds = ImplantDataset("test", scope="all")
    x, y, p = ds[0]
    print(f"  tensor {tuple(x.shape)}  label {y} ({ds.classes[y]})  path {p}")
