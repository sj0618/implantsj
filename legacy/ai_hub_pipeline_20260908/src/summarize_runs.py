"""
실행 결과 자동 집계 — outputs 폴더를 훑어 모든 run의 지표를 한 표로.

스테이지마다 결과 파일 이름과 형식이 달라서, 각각을 알아보고 공통 컬럼으로 모은다.
실험을 여러 번 돌린 뒤 무엇이 제일 좋았는지 한눈에 보려고 쓴다.

실행:
  python src/summarize_runs.py                          # outputs 전체
  python src/summarize_runs.py --stage metric           # 특정 스테이지만
  python src/summarize_runs.py --sort macro_f1 --top 20 # 상위 20개만
출처: implantsj pipelines/report/summarize_runs.py 의 아이디어
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_utils import PROJECT_ROOT


def _display_path(path: Path) -> str:
    """가능하면 프로젝트 기준 상대경로, 아니면 절대경로 그대로.

    paths.output_dir 는 프로젝트 밖(별도 데이터 디스크 등)을 가리킬 수 있어서
    relative_to를 그대로 쓰면 ValueError로 죽는다.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# 스테이지별로 읽을 파일과 그 안에서 뽑을 것
COMMON = ["accuracy", "macro_f1", "weighted_f1"]


def _rows_from_summary_csv(path: Path, stage: str) -> list[dict]:
    """스테이지가 이미 만들어 둔 summary.csv를 그대로 흡수."""
    frame = pd.read_csv(path)
    rows = []
    for record in frame.to_dict("records"):
        rows.append({"stage": stage, "run": path.parent.name,
                     "source": _display_path(path), **record})
    return rows


# metrics.json이 놓이는 하위 폴더 이름들 — run 이름이 아니라 split/평가 종류다
_SUBDIRS = {"valid", "test", "train", "eval", "evaluation"}


def _run_name(path: Path, stage_dir_name: str) -> tuple[str, str]:
    """metrics.json 경로에서 (run 이름, 그 안의 split/변형) 뽑기.

    classifier는 <run>/evaluation/test/metrics.json 처럼 두세 단계 아래에 있어서
    parent.name을 그대로 쓰면 run 열에 'test'가 찍힌다.
    """
    parts = list(path.parent.parts)
    tail = []
    while parts and (parts[-1] in _SUBDIRS
                     or parts[-1].startswith(("eval_", "rollout_"))):
        tail.insert(0, parts.pop())
    run = parts[-1] if parts and parts[-1] != stage_dir_name else stage_dir_name
    return run, "/".join(tail)


def _rows_from_metrics_json(path: Path, stage: str) -> list[dict]:
    """metrics.json 하나를 한 행으로."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    run, variant = _run_name(path, stage)
    row = {"stage": stage, "run": run, "source": _display_path(path)}
    if variant:
        row["eval"] = variant
    for key in COMMON + ["samples", "n", "split", "prototype", "distance", "unknown_rate"]:
        if key in payload:
            row[key] = payload[key]
    # 중첩된 overall/closed/open 블록도 펼친다
    for part in ("overall", "closed", "open", "duplicate_free"):
        block = payload.get(part)
        if isinstance(block, dict):
            for key, value in block.items():
                row[f"{part}_{key}"] = value
    return [row] if len(row) > 3 else []


def collect(root: Path, stages: list[str] | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    for stage_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        stage = stage_dir.name
        if stages and stage not in stages:
            continue
        # 스테이지가 만든 요약표를 먼저 (가장 정리된 형태)
        for summary in sorted(stage_dir.rglob("summary*.csv")):
            rows += _rows_from_summary_csv(summary, stage)
        # 요약표가 없는 run은 metrics.json에서
        seen = {r.get("run") for r in rows}
        for metrics in sorted(stage_dir.rglob("metrics*.json")):
            if metrics.parent.name not in seen:
                rows += _rows_from_metrics_json(metrics, stage)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--stage", action="append",
                        help="특정 스테이지만 (여러 번 지정 가능): backbone metric classifier attributes fuse")
    parser.add_argument("--sort", default="macro_f1", help="정렬 기준 컬럼")
    parser.add_argument("--top", type=int, help="상위 N개만 출력")
    parser.add_argument("--out", help="CSV로 저장할 경로 (기본: outputs/summary_all.csv)")
    args = parser.parse_args()

    root = Path(args.outputs)
    if not root.is_dir():
        print(f"outputs 폴더가 없습니다: {root}")
        return 1

    frame = collect(root, args.stage)
    if frame.empty:
        print(f"{root} 아래에서 결과 파일을 찾지 못했습니다 "
              f"(summary*.csv 또는 metrics*.json)")
        return 1

    # 정렬 기준이 없으면 있는 지표 중 하나로
    sort_key = args.sort
    if sort_key not in frame.columns:
        for candidate in ("overall_macro_f1", "macro_f1", "accuracy", "overall_accuracy"):
            if candidate in frame.columns:
                sort_key = candidate
                break
        else:
            sort_key = None
    if sort_key:
        frame = frame.sort_values(sort_key, ascending=False, na_position="last")

    out = Path(args.out) if args.out else root / "summary_all.csv"
    frame.to_csv(out, index=False)

    view = frame.head(args.top) if args.top else frame
    # top_k / finetune_mode 도 넣는다 — 없으면 fuse의 top1/top2 행이 구분되지 않는다
    columns = [c for c in ["stage", "run", "split", "prototype", "distance",
                           "eval", "backbone", "loss", "finetune_mode", "fuse_mode", "top_k",
                           "accuracy", "macro_f1", "overall_accuracy", "overall_macro_f1",
                           "open_macro_f1", "unknown_rate"] if c in view.columns]
    print(f"{len(frame)}개 결과  (정렬: {sort_key or '없음'})\n")
    with pd.option_context("display.width", 220, "display.max_rows", 60):
        print(view[columns].to_string(index=False) if columns else view.to_string(index=False))
    print(f"\n전체 저장 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
