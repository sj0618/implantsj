"""브랜치 산출물 경로 해석 — run 디렉토리나 best.pt를 주면 scores CSV를 찾아낸다.

fuse가 읽는 것은 체크포인트가 아니라 각 브랜치가 평가하며 남긴 [N, C] 점수 CSV다.
(fuse가 모델을 직접 로드하려면 세 브랜치의 구조와 전처리를 전부 알아야 해서
 패키지 독립성이 깨진다. 그래서 각 브랜치가 평가 때 점수를 남기고 fuse는 그것만 읽는다.)

받아들이는 입력 — 셋 다 같은 결과:
  outputs/metric/<tag>/                                run 디렉토리
  outputs/metric/<tag>/train/checkpoints/best.pt       체크포인트 (위로 거슬러 올라감)
  outputs/metric/<tag>/eval/scores_mean_cosine.csv     scores CSV 직접 지정
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import re

import pandas as pd

# 브랜치별 scores 파일이 놓이는 자리 (split → glob 패턴)
LAYOUTS: Dict[str, Dict[str, List[str]]] = {
    "metric": {
        "valid": ["eval/scores_valid_*.csv"],
        "test": ["eval/scores_test_*.csv"],
    },
    # band-vit  : <run>/evaluation/{valid,test}/scores.csv
    # masked-vit: <run>/eval_valid/, <run>/eval_{original,band}/  +  <run>/evaluation/{valid,test}/
    #
    # ★ 순서가 중요하다. masked-vit은 두 벌을 모두 남긴다:
    #     evaluation/{valid,test}/  학습 루프가 남긴 것 = **band 입력** (학습 분포)
    #     eval_valid/, eval_original/  학습 뒤 재평가 = **원본 입력** (배포 조건)
    #   masked-vit의 핵심 실험은 원본 입력이므로 eval_* 를 먼저 찾는다.
    #   band-vit에는 eval_* 가 없으므로 자동으로 evaluation/ 이 쓰인다.
    #
    #   eval_band/ 는 test를 band 입력으로 본 것이라 valid 자리에 절대 넣지 않는다
    #   (넣으면 fuse가 test로 가중치를 학습하게 된다).
    "classifier": {
        "valid": ["eval_valid/scores.csv", "evaluation/valid/scores.csv"],
        "test": ["eval_original/scores.csv", "evaluation/test/scores.csv",
                 "eval_band/scores.csv"],
    },
    "attributes": {
        "valid": ["scores_valid.csv"],
        "test": ["scores_test.csv"],
    },
}



def _has_split_token(path: Path, split: str) -> bool:
    """경로에 split 이름이 독립된 조각으로 들어 있는지 (test/scores.csv, scores_test_*.csv …)."""
    if split in path.parent.parts:
        return True
    return bool(re.search(rf"(^|[_\-.]){split}([_\-.]|$)", path.stem))


def _swap_split_token(path: Path, old: str, new: str) -> Path | None:
    """test → valid 처럼 경로의 split 조각만 바꾼 경로."""
    parts = list(path.parts)
    if old in parts[:-1]:                      # 폴더 이름이 split인 경우
        parts[parts.index(old)] = new
        return Path(*parts)
    swapped = re.sub(rf"(^|[_\-.]){old}([_\-.]|$)", rf"\g<1>{new}\g<2>", path.stem)
    if swapped != path.stem:
        return path.with_name(swapped + path.suffix)
    return None


def _to_run_dir(path: Path) -> Path:
    """best.pt / scores.csv 등을 받아 브랜치 run 디렉토리로 거슬러 올라간다."""
    if path.is_dir():
        return path
    # 체크포인트: <run>/train/checkpoints/best.pt 또는 <run>/checkpoints/best.pt
    parts = path.parts
    for marker in ("checkpoints",):
        if marker in parts:
            index = parts.index(marker)
            run = Path(*parts[:index])
            # metric은 <run>/train/checkpoints 구조라 train도 벗겨낸다
            return run.parent if run.name == "train" else run
    # scores CSV: <run>/eval/... 또는 <run>/evaluation/<split>/...
    for marker in ("eval", "evaluation"):
        if marker in parts:
            return Path(*parts[: parts.index(marker)])
    return path.parent


def _pick_metric_variant(run_dir: Path, variant: str | None) -> str:
    """metric은 prototype×distance 조합이 여러 개다. 지정이 없으면 summary.csv에서 best를 고른다."""
    if variant:
        return variant
    summary = run_dir / "eval" / "summary.csv"
    if not summary.is_file():
        raise FileNotFoundError(
            f"metric 조합을 고를 수 없습니다. {summary}가 없으니 "
            f"--metric-variant mean_cosine 처럼 직접 지정하세요."
        )
    frame = pd.read_csv(summary)
    # overall_macro_f1 → macro_f1 → accuracy 순으로 있는 지표를 쓴다
    for column in ("overall_macro_f1", "macro_f1", "overall_accuracy", "accuracy"):
        if column in frame.columns:
            best = frame.loc[frame[column].idxmax()]
            # variant 열이 파일 이름과 정확히 같다 (구면 평균처럼 접미사가 붙는 조합이 있다).
            # 없는 예전 결과만 prototype_distance로 조립한다.
            if "variant" in frame.columns and isinstance(best.get("variant"), str):
                return str(best["variant"])
            return f"{best['prototype']}_{best['distance']}"
    raise ValueError(f"{summary}에서 성능 컬럼을 찾을 수 없습니다")


def resolve_branch(branch: str, given: str | Path, split: str,
                   *, metric_variant: str | None = None) -> tuple[Path, str]:
    """(scores CSV 경로, 어떻게 찾았는지 설명) 반환."""
    given = Path(given).expanduser()

    # scores CSV를 직접 준 경우.
    # 같은 파일을 valid와 test에 함께 쓰면 test로 가중치·임계값을 학습하게 되므로
    # (= test 성능이 부풀려짐) 절대 그대로 돌려주지 않는다.
    # 파일 이름/경로에서 split을 읽어, 요청한 split의 짝을 찾아 준다.
    if given.is_file() and given.suffix == ".csv" and "score" in given.name:
        other = {"valid": "test", "test": "valid"}[split]
        if _has_split_token(given, split):
            return given, "직접 지정"
        if _has_split_token(given, other):
            counterpart = _swap_split_token(given, other, split)
            if counterpart is not None and counterpart.is_file():
                return counterpart, f"직접 지정({other} 경로에서 {split} 짝을 찾음)"
            raise FileNotFoundError(
                f"[{branch}] {split} 점수 파일을 찾지 못했습니다.\n"
                f"  준 파일은 {other} 것입니다: {given}\n"
                f"  기대한 짝: {counterpart}\n"
                f"  valid와 test를 같은 파일로 쓰면 test로 가중치를 학습하게 되므로 허용하지 않습니다.\n"
                f"  yaml에 run 대신 scores 블록으로 두 split을 각각 적으세요:\n"
                f"    scores:\n      valid: <valid scores.csv>\n      test:  <test scores.csv>"
            )
        raise ValueError(
            f"[{branch}] 이 CSV가 valid인지 test인지 알 수 없습니다: {given}\n"
            f"  경로나 파일명에 valid/test가 들어 있어야 자동으로 구분됩니다.\n"
            f"  yaml에 run 대신 scores 블록으로 두 split을 각각 적으세요:\n"
            f"    scores:\n      valid: <valid scores.csv>\n      test:  <test scores.csv>"
        )

    run_dir = _to_run_dir(given)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"[{branch}] 디렉토리를 찾을 수 없습니다: {run_dir}")

    if branch == "metric":
        variant = _pick_metric_variant(run_dir, metric_variant)
        candidate = run_dir / "eval" / f"scores_{split}_{variant}.csv"
        if not candidate.is_file():
            available = sorted(
                p.name[len(f"scores_{split}_"):-4]
                for p in (run_dir / "eval").glob(f"scores_{split}_*.csv")
            )
            raise FileNotFoundError(
                f"[metric] {candidate.name}이 없습니다. {split}에 있는 조합: {available}"
            )
        how = f"{run_dir.name} / {variant}"
        return candidate, how + ("" if metric_variant else " (summary.csv에서 자동 선택)")

    for pattern in LAYOUTS[branch][split]:
        matches = sorted(run_dir.glob(pattern))
        if matches:
            return matches[0], f"{run_dir.name} / {matches[0].relative_to(run_dir)}"

    tried = ", ".join(LAYOUTS[branch][split])
    raise FileNotFoundError(
        f"[{branch}] {run_dir} 아래에서 {split} scores를 못 찾았습니다. 찾아본 곳: {tried}\n"
        f"  fuse는 체크포인트(.pt)가 아니라 각 브랜치가 평가하며 남긴 점수 CSV를 읽습니다.\n"
        f"  .pt를 줘도 되지만 그건 run 폴더를 찾는 힌트일 뿐이라, "
        f"평가 산출물이 함께 있는 원래 run 폴더 안의 .pt여야 합니다.\n"
        f"  (체크포인트만 딴 곳으로 복사해 두면 여기서 걸립니다. "
        f"그 브랜치를 아직 평가하지 않았다면 먼저 평가부터 하세요.)"
    )
