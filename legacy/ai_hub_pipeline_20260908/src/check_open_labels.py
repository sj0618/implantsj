"""
open 클래스 목록 점검 — 데이터 준비를 돌리기 전에 확인용.

configs/00_data.yaml에 적은 open 클래스가 실제 데이터셋 라벨과 맞는지 본다.
이름이 조금 달라도 맞춰주지만, 못 찾는 게 있으면 비슷한 후보를 알려준다.

실행:
  python src/check_open_labels.py --config configs/00_data.yaml
  python src/check_open_labels.py --config configs/00_data.yaml --list   # 전체 라벨만 출력
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_utils import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--list", action="store_true", help="데이터셋 라벨 전체만 출력")
    args = parser.parse_args()

    from data.build import resolve_open_labels, scan_folder

    cfg = load_config(args.config)
    frame = scan_folder(cfg["source"]["root"], cfg["source"].get("layout", "auto"))
    counts = frame["label"].astype(str).value_counts().sort_index()
    print(f"데이터셋: {len(frame)}장 / {len(counts)}클래스  ({cfg['source']['root']})")

    if args.list:
        print("\n라벨 목록 (이미지 수):")
        for label, n in counts.items():
            print(f"  {label:44s} {n:>5d}")
        return 0

    open_cfg = cfg["openset"]
    wanted = [str(l).strip() for l in (open_cfg.get("labels") or []) if str(l).strip()]

    if not wanted:
        print("\n⚠ open 클래스가 지정되지 않았습니다.")
        print(f"  {args.config} 의 openset.labels 에 목록을 적으세요.")
        print(f"  전체 라벨을 보려면: python {Path(__file__).name} --config {args.config} --list")
        return 1

    print(f"open 클래스 {len(wanted)}개 지정됨 (openset.labels)")
    resolved = resolve_open_labels(frame, wanted)      # 실패하면 여기서 후보와 함께 중단

    open_set = set(resolved)
    closed = [l for l in counts.index if l not in open_set]
    open_images = int(counts[resolved].sum())
    print()
    print(f"{'open 클래스':44s} {'이미지':>6s}  {'split 존재':>12s}")
    print("-" * 68)
    for label in resolved:
        part = frame[frame["label"].astype(str) == label]
        splits = sorted(s for s in part["split"].astype(str).unique() if s)
        note = ",".join(splits) if splits else "(준비 단계에서 배정)"
        print(f"  {label:42s} {counts[label]:>6d}  {note:>12s}")
    print()
    print(f"closed {len(closed)}클래스 / {len(frame) - open_images}장")
    print(f"open   {len(resolved)}클래스 / {open_images}장")

    expected = open_cfg.get("count")
    if expected and len(resolved) != int(expected):
        print(f"\n⚠ count={expected}로 적혀 있는데 실제 지정은 {len(resolved)}개입니다")
    print("\n확인 완료 — bash scripts/00_data/prepare_data.sh 를 실행하면 됩니다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
