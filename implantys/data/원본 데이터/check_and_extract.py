#!/usr/bin/env python3
"""
1) SAM_finetuning_dataset_1_final(학습에 쓴 119장)과 7라벨 폴더가 겹치는지 검사
2) 겹치지 않는 '남은 이미지'를 pseudo-label 후보로 추출

Roboflow 파일명(xxx.rf.<HASH>.jpg)의 HASH로 동일 이미지를 식별.
같은 이미지는 어느 폴더에 있든 HASH가 같으므로 정확히 매칭됨.

사용법 (경로는 실제에 맞게 수정):
  python check_and_extract.py \
      --used-dir  "SAM_finetuning_dataset_1_final_7라벨 기준" \
      --pool-dir  "7라벨" \
      --out-list  pseudo_candidates.txt
"""
import argparse, shutil
from pathlib import Path


def img_hash(name: str) -> str:
    """Roboflow 파일명에서 이미지 해시 추출. 없으면 파일명 자체."""
    if ".rf." in name:
        return name.split(".rf.")[1].rsplit(".", 1)[0]
    return name


def collect(folder):
    """폴더(하위 포함) 내 모든 이미지의 {hash: path}"""
    d = {}
    for p in Path(folder).rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            d[img_hash(p.name)] = p
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--used-dir", required=True, help="학습에 쓴 119장 폴더")
    ap.add_argument("--pool-dir", required=True, help="전체 원본(7라벨) 폴더")
    ap.add_argument("--out-list", default="pseudo_candidates.txt")
    ap.add_argument("--copy-to", default=None,
                    help="지정 시 남은 이미지를 이 폴더로 복사")
    args = ap.parse_args()

    used = collect(args.used_dir)
    pool = collect(args.pool_dir)

    print(f"학습에 쓴(119장): {len(used)}개")
    print(f"원본 pool(7라벨): {len(pool)}개")

    # 겹치는 것 = pool 안에 있으면서 used에도 있는 해시
    overlap = set(used) & set(pool)
    print(f"\n[중복] 119장과 7라벨에 동시 존재: {len(overlap)}개")
    # (참고: 119장이 7라벨의 부분집합이면 이 값이 119에 가까움 = 정상)

    # pseudo-label 후보 = pool에는 있고 used에는 없는 것 (아직 안 쓴 것)
    remaining = [pool[h] for h in pool if h not in used]
    print(f"[남은 이미지] pseudo-label 후보(119장 제외): {len(remaining)}개")

    # 후보 목록 저장
    with open(args.out_list, "w", encoding="utf-8") as f:
        for p in sorted(remaining, key=str):
            f.write(str(p) + "\n")
    print(f"\n후보 목록 저장: {args.out_list}")

    # 검증: used 중 pool에 없는 게 있으면 경고 (119장이 7라벨 밖에서 왔다는 뜻)
    used_not_in_pool = set(used) - set(pool)
    if used_not_in_pool:
        print(f"\n[주의] 119장 중 {len(used_not_in_pool)}개는 7라벨에서 안 보임 "
              f"(파일명이 다르거나 다른 출처일 수 있음)")

    if args.copy_to:
        out = Path(args.copy_to)
        out.mkdir(parents=True, exist_ok=True)
        for p in remaining:
            shutil.copy2(p, out / p.name)
        print(f"\n남은 이미지 {len(remaining)}개를 {out}/ 로 복사 완료")


if __name__ == "__main__":
    main()
