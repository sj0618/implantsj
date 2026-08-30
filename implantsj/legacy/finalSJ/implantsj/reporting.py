"""Collect evaluation CSV files and write one reproducible comparison report."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Sequence

import pandas as pd


REQUIRED_AXES = {
    "prototype": ("gt", "mean", "ss"),
    "distance": ("euclidean", "cosine"),
    "group": ("all", "closed", "open"),
    "query": ("valid", "test"),
}
KEY_COLUMNS = ["model", "query", "prototype", "distance", "group"]


def collect_metrics(eval_root: Path, models: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model, query in product(models, REQUIRED_AXES["query"]):
        path = eval_root / model / "eval" / query / "metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if "model" in frame and set(frame["model"].astype(str)) != {model}:
            raise ValueError(f"{path} contains a different model name")
        if "query" in frame and set(frame["query"].astype(str)) != {query}:
            raise ValueError(f"{path} contains a different query name")
        frame["model"] = model
        frame["query"] = query
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def validate_canonical(frame: pd.DataFrame, models: list[str]) -> None:
    required = set(KEY_COLUMNS) | {
        "gallery_classes",
        "accuracy",
        "macro_f1",
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"metrics are missing columns: {missing_columns}")
    duplicated = frame[frame.duplicated(KEY_COLUMNS, keep=False)]
    if not duplicated.empty:
        raise ValueError(
            "duplicate metric combinations: "
            f"{duplicated[KEY_COLUMNS].drop_duplicates().to_records(index=False).tolist()}"
        )
    expected = set(
        product(
            models,
            REQUIRED_AXES["query"],
            REQUIRED_AXES["prototype"],
            REQUIRED_AXES["distance"],
            REQUIRED_AXES["group"],
        )
    )
    actual = set(frame[KEY_COLUMNS].itertuples(index=False, name=None))
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra or len(frame) != len(expected):
        raise ValueError(
            f"canonical metrics mismatch: rows={len(frame)}/{len(expected)}, "
            f"missing={missing}, extra={extra}"
        )
    galleries = sorted(pd.to_numeric(frame["gallery_classes"]).unique().tolist())
    if galleries != [42]:
        raise ValueError(f"gallery_classes must be 42 in every row; found={galleries}")


def select_operating_point(frame: pd.DataFrame) -> dict:
    valid = frame[(frame["query"] == "valid") & (frame["group"] == "all")]
    if valid.empty:
        raise ValueError("validation all-group rows are required")
    per_model = {}
    for model, rows in valid.groupby("model", sort=True):
        best = rows.sort_values(
            ["macro_f1", "accuracy", "prototype", "distance"],
            ascending=[False, False, True, True],
        ).iloc[0]
        per_model[str(model)] = {
            "prototype": str(best.prototype),
            "distance": str(best.distance),
            "valid_macro_f1": float(best.macro_f1),
            "valid_accuracy": float(best.accuracy),
        }
    overall_rows = (
        valid.groupby(["prototype", "distance"], as_index=False)[
            ["macro_f1", "accuracy"]
        ]
        .mean()
        .sort_values(
            ["macro_f1", "accuracy", "prototype", "distance"],
            ascending=[False, False, True, True],
        )
    )
    best = overall_rows.iloc[0]
    return {
        "per_model": per_model,
        "overall": {
            "prototype": str(best.prototype),
            "distance": str(best.distance),
            "valid_macro_f1_mean": float(best.macro_f1),
            "valid_accuracy_mean": float(best.accuracy),
        },
    }


def comparison_table(
    frame: pd.DataFrame, models: list[str], query: str, metric: str
) -> pd.DataFrame:
    if query not in REQUIRED_AXES["query"]:
        raise ValueError(f"query must be one of {REQUIRED_AXES['query']}")
    if metric not in frame.columns:
        raise KeyError(metric)
    selected = frame[(frame["query"] == query) & frame["model"].isin(models)]
    return selected.pivot(
        index=["prototype", "distance", "group"],
        columns="model",
        values=metric,
    ).reindex(columns=models)


def delta_table(frame: pd.DataFrame, base: str, other: str) -> pd.DataFrame:
    keys = ["query", "prototype", "distance", "group"]
    metrics = ["accuracy", "macro_f1"]
    left = frame[frame["model"] == base][keys + metrics]
    right = frame[frame["model"] == other][keys + metrics]
    merged = left.merge(right, on=keys, suffixes=("_base", "_other"), validate="1:1")
    merged.insert(0, "base", base)
    merged.insert(1, "other", other)
    for metric in metrics:
        merged[f"delta_{metric}"] = merged[f"{metric}_other"] - merged[f"{metric}_base"]
    return merged


def _markdown(frame: pd.DataFrame) -> str:
    values = frame.fillna("").astype(str)
    headers = list(values.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(row) + " |"
        for row in values.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _section_table(
    frame: pd.DataFrame, models: list[str], distance: str
) -> pd.DataFrame:
    selected = frame[
        frame["model"].isin(models) & (frame["distance"] == distance)
    ].copy()
    selected["accuracy / macro_f1"] = selected.apply(
        lambda row: f"{row.accuracy:.4f} / {row.macro_f1:.4f}", axis=1
    )
    table = selected.pivot(
        index=["query", "prototype", "group"],
        columns="model",
        values="accuracy / macro_f1",
    ).reset_index()
    table.columns.name = None
    return table.reindex(columns=["query", "prototype", "group", *models])


def build_experiment_sections(
    frame: pd.DataFrame, model_roles: dict[str, str]
) -> dict[str, str]:
    required_roles = {
        "base",
        "closed_ft",
        "closed_ft_supcon",
        "full_ft",
        "full_ft_supcon",
    }
    missing = sorted(required_roles - set(model_roles))
    if missing:
        raise ValueError(f"missing model roles: {missing}")
    comparisons = {
        "1": [
            model_roles["base"],
            model_roles["closed_ft"],
            model_roles["closed_ft_supcon"],
        ],
        "2": [model_roles["closed_ft"], model_roles["full_ft"]],
        "4": [model_roles["closed_ft"], model_roles["closed_ft_supcon"]],
        "5": [model_roles["full_ft"], model_roles["full_ft_supcon"]],
    }
    sections: dict[str, str] = {}
    for number, models in comparisons.items():
        for suffix, distance in (("a", "euclidean"), ("b", "cosine")):
            table = _section_table(frame, models, distance)
            deltas = []
            for other in models[1:]:
                delta = delta_table(frame, models[0], other)
                delta = delta[delta["distance"] == distance].copy()
                delta["comparison"] = f"{other} - {models[0]}"
                deltas.append(
                    delta[
                        [
                            "comparison",
                            "query",
                            "prototype",
                            "group",
                            "delta_accuracy",
                            "delta_macro_f1",
                        ]
                    ]
                )
            delta_text = _markdown(pd.concat(deltas, ignore_index=True).round(4))
            sections[f"{number}{suffix}"] = (
                f"Distance: `{distance}`. Each cell is accuracy / macro-F1.\n\n"
                + _markdown(table)
                + "\n\nDelta (other - base):\n\n"
                + delta_text
            )
    return sections


def write_report(
    output_dir: Path,
    frame: pd.DataFrame,
    sections: dict[str, str],
    geometry_frames: list[pd.DataFrame],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.sort_values(KEY_COLUMNS).to_csv(
        output_dir / "canonical_results.csv", index=False
    )
    tables = []
    models = frame["model"].drop_duplicates().tolist()
    for query, metric in product(REQUIRED_AXES["query"], ("accuracy", "macro_f1")):
        table = comparison_table(frame, models, query, metric).reset_index()
        table.insert(0, "metric", metric)
        table.insert(0, "query", query)
        tables.append(table)
    pd.concat(tables, ignore_index=True).to_csv(
        output_dir / "comparison_tables.csv", index=False
    )
    geometry = (
        pd.concat(geometry_frames, ignore_index=True).drop_duplicates()
        if geometry_frames
        else pd.DataFrame()
    )
    lines = [
        "# ImplantsJ experiment report",
        "",
        f"Canonical metric rows: {len(frame)}",
        "",
        "## Validation-selected operating points",
        "",
        "```json",
        json.dumps(select_operating_point(frame), ensure_ascii=False, indent=2),
        "```",
    ]
    for code, text in sections.items():
        lines.extend(["", f"## Experiment {code}", "", text])
    if not geometry.empty:
        geometry_columns = [
            column
            for column in (
                "model",
                "query",
                "prototype",
                "separation_ratio",
                "embedding_norm_mean",
                "embedding_norm_std",
                "mean_dimension_std",
                "near_zero_dimension_count",
            )
            if column in geometry
        ]
        lines.extend(
            ["", "## Geometry diagnostics", "", _markdown(geometry[geometry_columns])]
        )
    report = output_dir / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _parse_roles(value: str) -> dict[str, str]:
    roles: dict[str, str] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"expected ROLE=MODEL, got {item!r}")
        role, model = item.split("=", 1)
        roles[role.strip()] = model.strip()
    return roles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--models", required=True, help="comma-separated model names")
    parser.add_argument("--roles", required=True, help="comma-separated ROLE=MODEL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-rows", type=int, default=180)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    eval_root = Path(args.eval_root).expanduser().resolve()
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    frame = collect_metrics(eval_root, models)
    validate_canonical(frame, models)
    if len(frame) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} rows, found {len(frame)}")
    geometry_frames = []
    for model, query in product(models, REQUIRED_AXES["query"]):
        path = eval_root / model / "eval" / query / "geometry.csv"
        if path.is_file():
            geometry = pd.read_csv(path)
            geometry["query"] = query
            geometry_frames.append(geometry)
    output_dir = Path(args.output_dir).expanduser().resolve()
    report = write_report(
        output_dir,
        frame,
        build_experiment_sections(frame, _parse_roles(args.roles)),
        geometry_frames,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
