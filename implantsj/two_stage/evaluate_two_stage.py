#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    p = argparse.ArgumentParser(description="Evaluate two-stage prediction CSV.")
    p.add_argument("--predictions", default="outputs/two_stage_predictions.csv")
    p.add_argument("--output-dir", default="outputs")
    args = p.parse_args()

    df = pd.read_csv(args.predictions)
    y_true = df["true_label"].astype(str).tolist()
    y_pred = df["final_pred_label"].astype(str).tolist()
    labels = sorted(set(y_true) | set(y_pred))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "labels": labels,
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "two_stage_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels).to_csv(out_dir / "two_stage_confusion_matrix.csv")
    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"Wrote metrics to {out_dir}")


if __name__ == "__main__":
    main()
