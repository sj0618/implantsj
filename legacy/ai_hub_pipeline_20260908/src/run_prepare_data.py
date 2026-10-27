"""
Stage 0-a. 데이터 준비 — 파이프라인이 쓰는 manifest를 한 번에 만든다.

데이터셋 폴더를 훑어 `path,label,split` CSV를 만들고, band mask를 이어 붙인 뒤
closed / open으로 나눠 저장한다. 이후 모든 스테이지가 이 파일들을 참조한다.

만들어지는 파일 (configs/00_data.yaml 의 이름 그대로):
  manifest_full42.csv    42클래스 전체            ← full42 프로토콜, masked-vit
  manifest_closed27.csv  open을 제외한 클래스     ← closed27_open15 프로토콜의 학습
  manifest_open15.csv    open으로 뺀 클래스       ← 갤러리 등록 support + open query
  manifest_labels.json   어떤 클래스를 open으로 뺐는지 기록

band mask 경로(mask_path)는 세 파일 모두에 들어간다 — band-* 백본과 band-vit이 쓴다.

실행:
  # 0) 데이터 폴더 구조부터 확인 (아무것도 쓰지 않는다 — 구조를 모를 때 가장 먼저)
  python src/run_prepare_data.py --config configs/00_data.yaml --inspect
  python src/run_prepare_data.py --config configs/00_data.yaml --inspect \
      --set source.root=/mnt/aihub/데이터

  # 1) manifest 생성
  python src/run_prepare_data.py --config configs/00_data.yaml
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_utils import load_config_from_cli


def main():
    # --inspect: manifest를 만들기 전에 폴더 구조만 훑어 본다 (아무것도 쓰지 않는다).
    # 서버의 데이터 구조를 모른 채 들어갈 때 가장 먼저 실행할 것.
    inspect_only = "--inspect" in sys.argv
    if inspect_only:
        sys.argv = [a for a in sys.argv if a != "--inspect"]

    cfg = load_config_from_cli(__doc__)

    if inspect_only:
        from data.build import inspect_root
        print(inspect_root(cfg["source"]["root"]))
        return 0

    from data.build import (
        assign_splits,
        attach_band_masks,
        build_kfold_manifests,
        choose_open_labels,
        resolve_open_labels,
        scan_folder,
        split_closed_open,
        summarize,
    )

    source = cfg["source"]
    split_cfg = cfg["split"]
    open_cfg = cfg["openset"]
    out_dir = Path(cfg["paths"]["data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) 폴더 스캔 ----
    frame = scan_folder(source["root"], source.get("layout", "auto"))
    print(f"스캔: {len(frame)}장 / {frame['label'].nunique()}클래스  ({source['root']})")

    # ---- 2) split 배정 ----
    has_split = frame["split"].astype(str).str.len().gt(0).all()
    if has_split and split_cfg.get("keep_existing", True):
        print("폴더 구조의 train/valid/test를 그대로 사용합니다")
    else:
        ratios = [split_cfg["train"], split_cfg["valid"], split_cfg["test"]]
        frame = assign_splits(frame, ratios=ratios, seed=int(split_cfg.get("seed", 42)))
        print(f"split 배정: {ratios} (seed={split_cfg.get('seed', 42)}), 클래스마다 최소 1장씩 보장")

    # ---- 3) band mask 연결 ----
    mask_root = cfg.get("band", {}).get("mask_root")
    if mask_root and Path(mask_root).is_dir():
        frame = attach_band_masks(frame, mask_root,
                                  suffix=cfg["band"].get("suffix", "_pred.png"))
        found = int((frame["mask_path"] != "").sum())
        print(f"band mask 연결: {found}/{len(frame)}장  ({mask_root})")
    else:
        frame["mask_path"] = ""
        print("band mask 없음 — Stage 0을 먼저 돌리면 band-* 백본과 band-vit을 쓸 수 있습니다")

    # ---- 4) closed / open 분리 ----
    # 실제 open 클래스는 정해져 있으므로 명시 지정이 원칙이다.
    # 자동 선정은 auto를 명시했을 때만 동작한다 (실수로 엉뚱한 클래스가 open이 되는 걸 막는다).
    wanted = [str(l).strip() for l in (open_cfg.get("labels") or []) if str(l).strip()]
    if wanted:
        open_labels = resolve_open_labels(frame, wanted)
        expected = open_cfg.get("count")
        if expected and len(open_labels) != int(expected):
            print(f"  주의: count={expected}인데 지정된 open 클래스는 {len(open_labels)}개입니다")
    else:
        auto = open_cfg.get("auto")
        if not auto:
            raise ValueError(
                "open 클래스가 지정되지 않았습니다.\n"
                "  configs/00_data.yaml 의 openset.labels 에 클래스 이름을 적으세요.\n"
                "  전체 라벨 보기: python src/check_open_labels.py "
                "--config configs/00_data.yaml --list\n"
                "  (아직 목록을 모른다면 openset.auto: rarest 로 임시 선정할 수 있습니다)"
            )
        open_labels = choose_open_labels(
            frame, int(open_cfg["count"]), seed=int(open_cfg.get("seed", 42)), strategy=auto,
        )
        print(f"⚠ open 클래스 {len(open_labels)}개를 '{auto}' 기준으로 임시 선정했습니다 "
              f"— 실제 목록이 정해지면 openset.labels 에 적으세요")
    parts = split_closed_open(frame, open_labels)

    names = cfg["outputs"]
    written = {}
    for key, data in (("full", frame), ("closed", parts["closed"]), ("open", parts["open"])):
        path = out_dir / names[key]
        data.to_csv(path, index=False)
        written[key] = str(path)
        print("  " + summarize(data, names[key]))

    (out_dir / names["labels_json"]).write_text(json.dumps({
        "open_labels": list(open_labels),
        "closed_labels": sorted(parts["closed"]["label"].astype(str).unique().tolist()),
        "counts": {k: len(v) for k, v in
                   (("full", frame), ("closed", parts["closed"]), ("open", parts["open"]))},
        "files": written,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---- 5) k-fold (옵션) ----
    kfold_cfg = cfg.get("kfold", {}) or {}
    if kfold_cfg.get("enabled", False):
        k = int(kfold_cfg.get("k", 5))
        fold_dir = out_dir / kfold_cfg.get("subdir", "folds")
        for key, data in (("full", frame), ("closed", parts["closed"])):
            if not kfold_cfg.get(key, True):
                continue
            base = Path(names[key]).stem
            made = build_kfold_manifests(data, fold_dir, base,
                                         k=k, seed=int(kfold_cfg.get("seed", 42)))
            print(f"  {base}: {k}-fold manifest {len(made)}개 → {fold_dir}")
        print(f"  (test split은 fold마다 동일하게 유지됩니다)")

    print(f"\n완료 → {out_dir}")
    print(f"open 클래스: {open_labels}")
    # open의 train split이 비면 갤러리에 등록할 support가 없다
    open_train = parts["open"][parts["open"]["split"] == "train"]
    if open_train.empty:
        print("⚠ open 클래스에 train split이 없습니다 — metric 갤러리 등록용 support가 필요합니다")


if __name__ == "__main__":
    main()
