"""데이터셋 폴더 → manifest CSV. 파이프라인의 입구.

모든 스테이지가 `path,label,split` CSV를 입력으로 받는다.
이 모듈이 그 CSV를 만들고, closed/open 분리와 band mask 연결까지 처리한다.

받아들이는 폴더 구조:
  class_folder        <root>/<클래스>/*.jpg              (split은 여기서 나눈다)
  split_class_folder  <root>/{train,valid,test}/<클래스>/*.jpg  (split을 그대로 쓴다)

출처: implantsj pipelines/data/{make_manifest,make_splits}.py 의 로직
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# split 폴더 이름의 흔한 변형들 (소문자로 비교한다).
# AI Hub 한국어 데이터셋은 Training / Validation 표기를 자주 쓴다.
SPLIT_DIRS = {
    "train": "train", "training": "train", "tr": "train",
    "valid": "valid", "val": "valid", "validation": "valid", "dev": "valid",
    "test": "test", "testing": "test", "eval": "test",
}


def _split_of(name: str) -> str | None:
    """폴더 이름이 split을 뜻하면 정규화된 이름(train/valid/test), 아니면 None."""
    return SPLIT_DIRS.get(name.strip().lower())


def detect_layout(root: Path) -> str:
    if any(_split_of(p.name) for p in root.iterdir() if p.is_dir()):
        return "split_class_folder"
    return "class_folder"


def scan_folder(root: str | Path, layout: str = "auto") -> pd.DataFrame:
    """이미지를 훑어 path,label,split DataFrame을 만든다 (split은 비어 있을 수 있음)."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"데이터 폴더가 없습니다: {root}")
    if layout == "auto":
        layout = detect_layout(root)

    rows: List[dict] = []
    if layout == "class_folder":
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if _split_of(class_dir.name):
                continue
            for image in sorted(class_dir.rglob("*")):
                if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                    rows.append({"path": str(image), "label": class_dir.name, "split": ""})
    elif layout == "split_class_folder":
        for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            split = _split_of(split_dir.name)
            if split is None:
                continue
            for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                for image in sorted(class_dir.rglob("*")):
                    if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                        rows.append({"path": str(image), "label": class_dir.name, "split": split})
    else:
        raise ValueError(f"unsupported layout: {layout}")

    if not rows:
        raise RuntimeError(f"이미지를 찾지 못했습니다: {root}")
    return pd.DataFrame(rows)


def assign_splits(frame: pd.DataFrame, *, ratios: Sequence[float] = (8, 1, 1),
                  seed: int = 42) -> pd.DataFrame:
    """클래스 비율을 유지하며 train/valid/test를 나눈다.

    클래스마다 최소 1장씩은 각 split에 들어가게 해서, 어떤 클래스가
    valid나 test에서 통째로 빠지는 일이 없도록 한다.
    """
    weights = np.asarray(ratios, dtype=float)
    if weights.sum() <= 0 or (weights < 0).any():
        raise ValueError("split 비율은 음수가 아니고 합이 0보다 커야 합니다")
    weights = weights / weights.sum()
    active = np.flatnonzero(weights > 0)
    names = ["train", "valid", "test"]

    rng = np.random.default_rng(seed)
    out = frame.copy()
    out["split"] = ""
    too_small = {}
    for label, group in out.groupby("label", sort=True):
        # copy=True 필수 — pandas 3.x의 to_numpy()는 읽기 전용 뷰를 줄 수 있고
        # 그대로 shuffle하면 "array is read-only"로 죽는다
        index = group.index.to_numpy(copy=True)
        rng.shuffle(index)
        if len(index) < len(active):
            too_small[str(label)] = len(index)
            continue
        raw = weights * len(index)
        counts = np.floor(raw).astype(int)
        counts[active] = np.maximum(counts[active], 1)      # 각 split에 최소 1장
        while counts.sum() > len(index):
            candidates = [i for i in active if counts[i] > 1]
            counts[max(candidates, key=lambda i: counts[i] - raw[i])] -= 1
        while counts.sum() < len(index):
            counts[max(active, key=lambda i: raw[i] - counts[i])] += 1
        start = 0
        for position, count in enumerate(counts):
            out.loc[index[start:start + count], "split"] = names[position]
            start += count

    if too_small:
        raise ValueError(
            f"split 수({len(active)})보다 이미지가 적은 클래스가 있습니다: {too_small}. "
            f"해당 클래스를 빼거나 비율을 조정하세요."
        )
    return out


def build_kfold_manifests(frame: pd.DataFrame, out_dir: str | Path, base_name: str,
                          *, k: int = 5, seed: int = 42) -> List[Path]:
    """train+valid를 stratified k-fold로 다시 나눈 manifest를 k개 만든다.

    test split은 건드리지 않는다 — 최종 평가 기준은 fold마다 같아야 하기 때문이다.
    각 fold 파일에는 원래 split이 source_split 컬럼으로 남는다.
    출처: implantsj src/kfold.py
    """
    if k < 2:
        raise ValueError(f"k는 2 이상이어야 합니다: {k}")
    from sklearn.model_selection import StratifiedKFold

    work = frame.copy().reset_index(drop=True)
    work["source_split"] = work["split"].astype(str)
    pool = work[work["source_split"] != "test"].reset_index(drop=True)
    held_out = work[work["source_split"] == "test"].reset_index(drop=True)

    counts = pool["label"].value_counts()
    too_small = counts[counts < k]
    if not too_small.empty:
        raise ValueError(
            f"stratified {k}-fold를 하려면 클래스마다 최소 {k}장이 필요합니다. "
            f"부족: {too_small.to_dict()}"
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    labels = pool["label"].to_numpy()

    paths: List[Path] = []
    rows = []
    for fold, (train_index, valid_index) in enumerate(splitter.split(pool, labels), start=1):
        part = pool.copy()
        part["split"] = "train"
        part.loc[valid_index, "split"] = "valid"
        part["fold"] = fold
        combined = pd.concat([part, held_out.assign(fold=fold)], ignore_index=True)
        path = out / f"{base_name}_fold{fold:02d}.csv"
        combined.to_csv(path, index=False)
        paths.append(path)
        for split in ("train", "valid", "test"):
            block = combined[combined["split"] == split]
            rows.append({"fold": fold, "split": split, "images": len(block),
                         "classes": block["label"].nunique()})

    pd.DataFrame(rows).to_csv(out / f"{base_name}_fold_summary.csv", index=False)
    return paths


def attach_band_masks(frame: pd.DataFrame, mask_root: str | Path,
                      *, suffix: str = "_pred.png", column: str = "mask_path") -> pd.DataFrame:
    """Stage 0이 만든 예측 마스크를 파일 stem으로 이어 붙인다."""
    index = {p.name[: -len(suffix)]: str(p.resolve())
             for p in Path(mask_root).expanduser().rglob(f"*{suffix}")}
    out = frame.copy()
    out[column] = [index.get(Path(p).stem, "") for p in out["path"]]
    missing = int((out[column] == "").sum())
    if missing:
        print(f"  경고: band mask를 못 찾은 이미지 {missing}/{len(out)}장 "
              f"(band-* 백본과 band-vit은 이 행을 쓸 수 없습니다)")
    return out


def _normalize(name: str) -> str:
    """폴더명 표기 차이를 흡수 (대소문자·공백·구분자·앞 일련번호)."""
    s = re.sub(r"^\d+[_\-\s]+", "", str(name).strip().lower())
    return re.sub(r"[^a-z0-9]+", "", s)


def resolve_open_labels(frame: pd.DataFrame, wanted: Sequence[str]) -> List[str]:
    """지정한 open 클래스 이름을 데이터셋의 실제 라벨에 맞춘다.

    정확히 일치하면 그대로 쓰고, 아니면 표기 차이를 무시하고 한 번 더 찾는다
    (`03_Osstem_SSII` ↔ `Osstem SSII`). 그래도 못 찾으면 비슷한 후보를 보여주며 중단한다 —
    open 클래스가 하나라도 틀리면 실험 전체가 어긋나므로 조용히 넘어가지 않는다.
    """
    labels = sorted(frame["label"].astype(str).unique().tolist())
    exact = {l: l for l in labels}
    loose: Dict[str, List[str]] = {}
    for l in labels:
        loose.setdefault(_normalize(l), []).append(l)

    resolved, missing = [], []
    for name in wanted:
        name = str(name).strip()
        if not name:
            continue
        if name in exact:
            resolved.append(name)
            continue
        candidates = loose.get(_normalize(name), [])
        if len(candidates) == 1:
            print(f"  open 클래스 이름 보정: {name!r} → {candidates[0]!r}")
            resolved.append(candidates[0])
        elif len(candidates) > 1:
            raise ValueError(f"open 클래스 {name!r}이 여러 라벨과 겹칩니다: {candidates}")
        else:
            missing.append(name)

    if missing:
        lines = [f"데이터셋에서 찾지 못한 open 클래스 {len(missing)}개:"]
        for name in missing:
            key = _normalize(name)
            near = [l for l in labels if key[:6] and key[:6] in _normalize(l)][:3]
            lines.append(f"    {name!r}" + (f"   비슷한 라벨: {near}" if near else ""))
        lines.append(f"  데이터셋 라벨 {len(labels)}개 중 예시: {labels[:5]}")
        raise ValueError("\n".join(lines))

    duplicates = sorted({l for l in resolved if resolved.count(l) > 1})
    if duplicates:
        raise ValueError(f"open 클래스 목록에 중복이 있습니다: {duplicates}")
    return sorted(resolved)


def split_closed_open(frame: pd.DataFrame, open_labels: Sequence[str]) -> Dict[str, pd.DataFrame]:
    """closed / open 두 갈래로 나눈다.

    open 클래스는 임베딩 학습에 쓰지 않고, train split은 갤러리 등록용 support가 된다.
    """
    open_set = {str(l) for l in open_labels}
    is_open = frame["label"].astype(str).isin(open_set)
    closed = frame[~is_open].reset_index(drop=True)
    if closed.empty:
        raise ValueError("closed 클래스가 하나도 남지 않았습니다")
    return {"closed": closed, "open": frame[is_open].reset_index(drop=True)}


def choose_open_labels(frame: pd.DataFrame, count: int, *, seed: int = 42,
                       strategy: str = "rarest") -> List[str]:
    """open으로 뺄 클래스를 자동으로 고른다 (실제 목록을 아직 모를 때만).

    rarest : 이미지가 적은 클래스부터 (희귀 시스템이 unseen이 되는 현실적 상황)
    random : 무작위
    """
    counts = frame["label"].astype(str).value_counts()
    if count >= len(counts):
        raise ValueError(f"open 클래스 수({count})가 전체 클래스 수({len(counts)}) 이상입니다")
    if strategy == "rarest":
        return sorted(counts.tail(count).index.tolist())
    if strategy == "random":
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(counts.index.to_numpy(), count, replace=False).tolist())
    raise ValueError("strategy must be rarest or random")



def summarize(frame: pd.DataFrame, name: str) -> str:
    counts = frame.groupby("split")["label"].agg(["count", "nunique"])
    parts = [f"{s}: {int(r['count'])}장/{int(r['nunique'])}클래스" for s, r in counts.iterrows()]
    return f"{name}: 총 {len(frame)}장, {frame['label'].nunique()}클래스  ({', '.join(parts)})"


def inspect_root(root: str | Path, max_depth: int = 4, sample: int = 12) -> str:
    """데이터 폴더 구조를 훑어 사람이 읽을 보고서를 만든다.

    서버의 폴더 구조를 모른 채 들어갈 때, manifest를 만들기 **전에** 무엇이 보이는지
    확인하는 용도. 아무것도 쓰지 않고 읽기만 한다.
    """
    root = Path(root).expanduser()
    lines = [f"데이터 폴더: {root}"]
    if not root.is_dir():
        return "\n".join(lines + ["  ✗ 폴더가 없습니다."])
    root = root.resolve()

    # ---- 깊이별 폴더/파일 분포 ----
    lines.append("")
    lines.append("깊이별 구조 (이미지가 어느 깊이에 있는지 보면 layout을 알 수 있다)")
    for depth in range(1, max_depth + 1):
        pattern = "/".join(["*"] * depth)
        entries = list(root.glob(pattern))
        dirs = [e for e in entries if e.is_dir()]
        imgs = [e for e in entries if e.is_file() and e.suffix.lower() in IMAGE_EXTENSIONS]
        others = [e for e in entries if e.is_file() and e.suffix.lower() not in IMAGE_EXTENSIONS]
        if not entries:
            break
        names = ", ".join(sorted(d.name for d in dirs)[:6])
        more = f" …(+{len(dirs) - 6})" if len(dirs) > 6 else ""
        lines.append(f"  깊이 {depth}: 폴더 {len(dirs):5d}  이미지 {len(imgs):7d}  기타 {len(others):5d}"
                     + (f"   예: {names}{more}" if dirs else ""))

    # ---- 확장자 분포 ----
    exts: Dict[str, int] = {}
    for f in root.rglob("*"):
        if f.is_file():
            exts[f.suffix.lower() or "(없음)"] = exts.get(f.suffix.lower() or "(없음)", 0) + 1
    known = {e: n for e, n in exts.items() if e in IMAGE_EXTENSIONS}
    unknown = {e: n for e, n in exts.items() if e not in IMAGE_EXTENSIONS}
    lines.append("")
    lines.append(f"확장자: 인식 {known}")
    if unknown:
        lines.append(f"        무시 {dict(sorted(unknown.items(), key=lambda kv: -kv[1])[:8])}")
        if any(e in (".json", ".xml", ".csv", ".txt") for e in unknown):
            lines.append("        ⚠ 라벨이 파일(json/xml/csv)에 들어 있는 구조일 수 있습니다 —"
                         " 폴더명이 클래스가 아니면 별도 변환이 필요합니다.")

    # ---- 두 layout으로 각각 스캔해 보기 ----
    lines.append("")
    lines.append("layout별 스캔 결과")
    detected = detect_layout(root)
    for layout in ("class_folder", "split_class_folder"):
        mark = "← auto가 고른 것" if layout == detected else ""
        try:
            frame = scan_folder(root, layout)
            counts = frame["label"].value_counts()
            splits = (frame["split"].replace("", "(없음)").value_counts().to_dict())
            lines.append(f"  {layout:19s} 이미지 {len(frame):7d}  클래스 {frame['label'].nunique():4d}"
                         f"  split {splits}  {mark}")
            lines.append(f"      클래스 예: {list(counts.index[:sample])}")
            lines.append(f"      장수 최소 {counts.min()} / 중앙값 {int(counts.median())} / 최대 {counts.max()}")
            if counts.min() < 3:
                thin = counts[counts < 3].to_dict()
                lines.append(f"      ⚠ 3장 미만 클래스: {thin} — split 배분이 어렵습니다.")
        except Exception as exc:
            lines.append(f"  {layout:19s} ✗ {type(exc).__name__}: {exc}  {mark}")

    lines.append("")
    lines.append("판단 기준")
    lines.append("  · 클래스 수가 예상과 다르면 layout을 직접 지정하세요 (source.layout).")
    lines.append("  · 클래스 폴더 위에 병원/기관 같은 단계가 하나 더 있으면 그 단계가 클래스로 잡힙니다.")
    lines.append("  · 라벨이 json/xml에 있으면 path,label CSV를 따로 만들어 두면 됩니다"
                 " (manifest 형식이 그대로 입력입니다).")
    return "\n".join(lines)
