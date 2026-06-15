#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def infer_groups_from_predictions(predictions_csv: Path, min_count: int) -> dict[str, list[str]]:
    df = pd.read_csv(predictions_csv)
    required = {"true_label", "pred_label"}
    if not required.issubset(df.columns):
        raise ValueError(f"{predictions_csv} must contain columns: {sorted(required)}")

    mistakes = df[df["true_label"].astype(str) != df["pred_label"].astype(str)].copy()
    pair_counts: dict[tuple[str, str], int] = {}
    for _, row in mistakes.iterrows():
        pair = tuple(sorted([str(row["true_label"]), str(row["pred_label"])]))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    groups = {}
    for pair, count in sorted(pair_counts.items(), key=lambda item: (-item[1], item[0])):
        if count >= min_count:
            groups["_".join(pair)] = list(pair)
    return groups


def main():
    p = argparse.ArgumentParser(description="Build confusion group JSON from a prediction CSV.")
    p.add_argument("--predictions", required=True, help="CSV with true_label and pred_label columns")
    p.add_argument("--output", default="two_stage/confusion_groups.json")
    p.add_argument("--min-count", type=int, default=1)
    args = p.parse_args()

    groups = infer_groups_from_predictions(Path(args.predictions), args.min_count)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(groups, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(groups)} groups to {out}")


if __name__ == "__main__":
    main()