#!/usr/bin/env python3
"""Select one Mahalanobis variant on validation and challenge it on strict 5-shot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from tools.improve_unseen_mahalanobis import candidate_scorers, sha256


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "pooled_ledoit_wolf"
CONDITIONS = {
    name: ROOT / "docs/reports/mahalanobis_improvement_20260813" / name / "embedding_cache.npz"
    for name in ("fresh_backbone", "fresh_projection", "legacy_backbone", "legacy_projection")
}
STRICT_ROOT = ROOT / "docs/reports/vit6supcon_to_5label_strict10_20260814"
STRICT_CACHE = STRICT_ROOT / "strict10/backbone_seed42/embedding_cache.npz"
STRICT_EPISODES = STRICT_ROOT / "episodic_all_target_5shot"
SEEDS = (0, 1, 2, 3, 4, 42)


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
    }


def load_cache(path: Path) -> dict[str, np.ndarray]:
    from pipelines.eval.reproduce_gallery import validate_cache
    with np.load(path, allow_pickle=False) as archive:
        cache = {key: archive[key] for key in archive.files}
    validate_cache(cache)
    return cache


def rank(frame: pd.DataFrame, method_order: list[str]) -> pd.DataFrame:
    result = frame.groupby("method", as_index=False)[["macro_f1", "balanced_accuracy", "accuracy"]].mean()
    result["order"] = result.method.map({name: index for index, name in enumerate(method_order)})
    return result.sort_values(
        ["macro_f1", "balanced_accuracy", "accuracy", "order"],
        ascending=[False, False, False, True],
    ).drop(columns="order").reset_index(drop=True)


def evaluate_selection() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    rows: list[dict] = []
    method_sets: list[set[str]] = []
    method_order: list[str] = []
    fitted = []
    for condition, path in CONDITIONS.items():
        cache = load_cache(path)
        scorers = candidate_scorers(cache["train_x"], cache["train_y"], len(cache["labels"]), 42)
        names = [name for name, _, _ in scorers]
        method_order = method_order or names
        method_sets.append(set(names))
        fitted.append((condition, cache, scorers))
        for name, _, scorer in scorers:
            prediction = scorer(cache["valid_x"]).argmin(axis=1)
            rows.append({"condition": condition, "method": name, "split": "valid", **metrics(cache["valid_y"], prediction)})
    common = set.intersection(*method_sets)
    method_order = [name for name in method_order if name in common]
    results = pd.DataFrame(rows).query("method in @common").reset_index(drop=True)
    validation_rank = rank(results.query("split == 'valid'"), method_order)
    winner = str(validation_rank.iloc[0].method)
    for condition, cache, scorers in fitted:
        for name, _, scorer in scorers:
            if name in common:
                prediction = scorer(cache["test_x"]).argmin(axis=1)
                rows.append({"condition": condition, "method": name, "split": "test", **metrics(cache["test_y"], prediction)})
    return pd.DataFrame(rows).query("method in @common").reset_index(drop=True), validation_rank, winner


def evaluate_strict() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    cache = load_cache(STRICT_CACHE)
    features = np.concatenate([cache[f"{split}_x"] for split in ("train", "valid", "test")])
    targets = np.concatenate([cache[f"{split}_y"] for split in ("train", "valid", "test")])
    paths = np.concatenate([cache[f"{split}_paths"] for split in ("train", "valid", "test")]).astype(str)
    labels = cache["labels"].astype(str).tolist()
    path_to_index = {path: index for index, path in enumerate(paths)}
    if len(path_to_index) != len(paths):
        raise ValueError("strict cache paths must be unique")
    if len({sha256(path) for path in paths}) != len(paths):
        raise ValueError("strict cache contains exact duplicate images")

    rows: list[dict] = []
    method_sets: list[set[str]] = []
    method_order: list[str] = []
    for seed in SEEDS:
        membership = pd.read_csv(STRICT_EPISODES / f"seed{seed}/support_membership.csv")
        support = np.asarray([path_to_index[str(path)] for path in membership.path], dtype=int)
        if len(set(support)) != len(support) or not np.array_equal(np.bincount(targets[support], minlength=len(labels)), np.full(len(labels), 5)):
            raise ValueError(f"Expected five distinct support images per class for seed {seed}")
        if [labels[index] for index in targets[support]] != membership.label.astype(str).tolist():
            raise ValueError(f"strict support membership mismatch for seed {seed}")
        query = np.ones(len(targets), dtype=bool)
        query[support] = False
        support_x, query_x = features[support], features[query]
        support_x = support_x / np.maximum(np.linalg.norm(support_x, axis=1, keepdims=True), 1e-12)
        query_x = query_x / np.maximum(np.linalg.norm(query_x, axis=1, keepdims=True), 1e-12)
        scorers = candidate_scorers(support_x, targets[support], len(labels), seed)
        names = [name for name, _, _ in scorers]
        method_order = method_order or names
        method_sets.append(set(names))
        for name, _, scorer in scorers:
            prediction = scorer(query_x).argmin(axis=1)
            rows.append({"seed": seed, "method": name, "n": int(query.sum()), **metrics(targets[query], prediction)})

    common = set.intersection(*method_sets)
    method_order = [name for name in method_order if name in common]
    results = pd.DataFrame(rows).query("method in @common").reset_index(drop=True)
    summary = rank(results, method_order)

    previous = pd.read_csv(STRICT_EPISODES / "all_seed_results.csv")
    previous = previous[previous.method.isin([BASELINE, "rda_0.75_logdet_1"])]
    reproduced = results[results.method.isin(previous.method.unique())]
    joined = previous.merge(reproduced, on=["seed", "method", "n"], suffixes=("_old", "_new"))
    if len(joined) != 12 or len(previous) != 12 or len(reproduced) != 12:
        raise ValueError("Strict parity requires all six seeds and both reference methods")
    error = max(
        float((joined[f"{column}_old"] - joined[f"{column}_new"]).abs().max())
        for column in ("accuracy", "macro_f1", "balanced_accuracy")
    )
    return results, summary, error


def pct(value: float) -> str:
    return f"{100 * value:.2f}"


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, divider]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def write_report(
    output: Path,
    results: pd.DataFrame,
    validation_rank: pd.DataFrame,
    winner: str,
    strict_results: pd.DataFrame,
    strict_summary: pd.DataFrame,
    reproduction_error: float,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "condition_results.csv", index=False)
    validation_rank.to_csv(output / "validation_ranking.csv", index=False)
    strict_results.to_csv(output / "strict5_results.csv", index=False)
    strict_summary.to_csv(output / "strict5_ranking.csv", index=False)

    selected = results.query("split == 'test' and method in [@BASELINE, @winner]").copy()
    selected.to_csv(output / "selected_test_vs_baseline.csv", index=False)
    pivot = selected.pivot(index="condition", columns="method", values="macro_f1")
    improved_all = bool((pivot[winner] > pivot[BASELINE]).all())
    validation_gain = float(
        validation_rank.set_index("method").loc[winner, "macro_f1"]
        - validation_rank.set_index("method").loc[BASELINE, "macro_f1"]
    )
    test_gain = float(
        selected.groupby("method").macro_f1.mean()[winner]
        - selected.groupby("method").macro_f1.mean()[BASELINE]
    )
    strict_means = strict_results.groupby("method")[["accuracy", "macro_f1", "balanced_accuracy"]].mean()
    strict_gain = float(strict_means.loc[winner, "macro_f1"] - strict_means.loc[BASELINE, "macro_f1"] )

    top = validation_rank.head(10).copy()
    for column in ("macro_f1", "balanced_accuracy", "accuracy"):
        top[column] = top[column].map(pct)
    test_table = selected[["condition", "method", "accuracy", "macro_f1", "balanced_accuracy"]].copy()
    for column in ("accuracy", "macro_f1", "balanced_accuracy"):
        test_table[column] = test_table[column].map(pct)
    strict_table = strict_summary.head(10).copy()
    for column in ("macro_f1", "balanced_accuracy", "accuracy"):
        strict_table[column] = strict_table[column].map(pct)

    strict_difference = (
        f"{100 * abs(strict_gain):.2f}%p 높았다"
        if strict_gain > 0
        else f"{100 * abs(strict_gain):.2f}%p 낮았다"
    )
    test_consistency = (
        "네 조건에서 모두 pooled baseline보다 높았다."
        if improved_all
        else "일부 조건에서는 pooled baseline보다 낮았다."
    )
    report = f"""# 마할라노비스 방법 선택 재실행 결과

## 최종 선택

이번 비교에서는 `{winner}`을 최종 방법으로 골랐다. fresh와 legacy 데이터의 backbone, projection 조합 네 가지에서 공통으로 실행할 수 있는 {len(validation_rank)}개 후보를 비교했다. 순위는 검증 데이터의 Macro-F1 평균으로 정했고, 테스트 결과는 방법을 고른 뒤 확인했다.

`{winner}`의 검증 Macro-F1은 평균 {pct(validation_rank.iloc[0].macro_f1)}%였다. `{BASELINE}`보다 {100 * validation_gain:.2f}%p 높다. 테스트에서도 평균 Macro-F1이 {100 * test_gain:.2f}%p 올랐으며, {test_consistency}

## 검증 성능 순위

{markdown_table(top, ['method', 'macro_f1', 'balanced_accuracy', 'accuracy'])}

## 테스트 결과

{markdown_table(test_table, ['condition', 'method', 'accuracy', 'macro_f1', 'balanced_accuracy'])}

## Strict 5-label 5-shot 확인

클래스마다 5장만 사용해 프로토타입과 공분산을 만들고 여섯 개 seed로 반복했을 때, `{winner}`의 Macro-F1은 pooled 기준보다 {strict_difference}.

{markdown_table(strict_table, ['method', 'macro_f1', 'balanced_accuracy', 'accuracy'])}

저장된 strict 결과와 다시 계산한 값의 최대 절대 오차는 `{reproduction_error:.3g}`이었다.

## 적용 기준

선택 방법은 `{winner}`다. 입력 파일이 원 보고서와 같다는 사실은 별도로 검증해야 한다. 이 실행은 과거 점수와의 일치를 자동으로 주장하지 않는다.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")

    checks = {
        "common_candidate_count_is_24": len(validation_rank) == 24,
        "condition_result_rows_is_192": len(results) == 192,
        "winner_beats_baseline_on_validation": validation_gain > 0,
        "winner_beats_baseline_on_all_four_tests": improved_all,
        "strict_result_rows_is_120": len(strict_results) == 120,
        "strict_previous_results_reproduced": reproduction_error < 1e-12,
        "all_metrics_finite": bool(
            np.isfinite(results[["accuracy", "macro_f1", "balanced_accuracy"]]).all().all()
            and np.isfinite(strict_results[["accuracy", "macro_f1", "balanced_accuracy"]]).all().all()
        ),
    }
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "winner": winner,
        "validation_macro_f1_gain_percentage_points": 100 * validation_gain,
        "test_macro_f1_gain_percentage_points": 100 * test_gain,
        "strict5_macro_f1_gain_percentage_points": 100 * strict_gain,
        "output_artifact_path": str(output / "REPORT.md"),
        "checks": checks,
    }
    (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    global CONDITIONS, STRICT_CACHE, STRICT_EPISODES
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="append", required=True, metavar="CONDITION=PATH")
    parser.add_argument("--strict-cache", type=Path, required=True)
    parser.add_argument("--strict-episodes", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--completion-artifact",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    conditions = dict(value.split("=", 1) for value in args.cache)
    if set(conditions) != set(CONDITIONS) or len(args.cache) != 4:
        parser.error("Exactly fresh_backbone, fresh_projection, legacy_backbone, legacy_projection are required")
    CONDITIONS = {key: Path(value) for key, value in conditions.items()}
    STRICT_CACHE, STRICT_EPISODES = args.strict_cache, args.strict_episodes
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.completion_artifact = args.completion_artifact or args.output_dir / "completion.json"
    results, validation_rank, winner = evaluate_selection()
    strict_results, strict_summary, reproduction_error = evaluate_strict()
    result = write_report(
        args.output_dir.resolve(),
        results,
        validation_rank,
        winner,
        strict_results,
        strict_summary,
        reproduction_error,
    )
    args.completion_artifact.parent.mkdir(parents=True, exist_ok=True)
    args.completion_artifact.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
