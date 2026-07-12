#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PANELS = [
    ("Training loss", ["train_loss"], False),
    ("Accuracy", ["valid_accuracy", "test_accuracy"], True),
    ("Macro-F1", ["valid_macro_f1", "test_macro_f1"], True),
]


def axis_limits(values, bounded):
    low, high = float(values.min()), float(values.max())
    span = high - low or max(abs(high) * 0.05, 0.02)
    pad = max(span * 0.12, 0.005 if bounded else span * 0.05)
    return (max(0.0, low - pad), min(1.0, high + pad)) if bounded else (low - pad, high + pad)


def model_name(run):
    name = run.name.casefold()
    if "closedft_supcon" in name:
        return "ViT-ClosedFT-SupCon"
    if "closedft27" in name:
        return "ViT-ClosedFT"
    if "fullft_supcon" in name:
        return "ViT-FullFT-SupCon"
    if "models_vit_42class" in name:
        return "ViT-FullFT"
    return run.name


def curve_status(run, frame):
    status = f"{len(frame)} epoch{'s' if len(frame) != 1 else ''}"
    notes = []
    if "metricdiag" in run.name.casefold() and len(frame) < 10:
        notes.append(f"partial {len(frame)}/10")
    missing = sorted({"test_accuracy", "test_macro_f1"} - set(frame.columns))
    if missing:
        notes.append("curve partial: missing " + ", ".join(missing))
    return status + (" · " + " · ".join(notes) if notes else "")


def plot_run(run, output):
    history = run / "metrics" / "history.csv"
    if not history.is_file():
        print(f"SKIP missing history: {run}")
        return None
    frame = pd.read_csv(history)
    if frame.empty:
        print(f"SKIP empty history: {history}")
        return None

    epochs = frame["epoch"] if "epoch" in frame else range(1, len(frame) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), squeeze=False)
    for axis, (title, columns, bounded) in zip(axes[0], PANELS):
        present = [column for column in columns if column in frame]
        if not present:
            axis.text(0.5, 0.5, "Unavailable", ha="center", va="center", transform=axis.transAxes)
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            continue
        for column in present:
            axis.plot(epochs, frame[column], marker="o", label=column)
        values = pd.concat([frame[column] for column in present]).dropna()
        if not values.empty:
            axis.set_ylim(*axis_limits(values, bounded))
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()

    name = model_name(run)
    status = curve_status(run, frame)
    fig.suptitle(f"{name} · {status}")
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    target = output / (re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-") + ".png")
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"SAVED {target}")
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--output-dir", default="/home/ubuntu/workspace/outputs/metricdiag-viz")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        assert axis_limits(pd.Series([0.8, 0.9]), True)[0] < 0.8
        assert axis_limits(pd.Series([4.0]), False)[0] < 4.0 < axis_limits(pd.Series([4.0]), False)[1]
        assert "missing test_accuracy" in curve_status(Path("metricdiag"), pd.DataFrame({"epoch": [1]}))
        print("VISUALIZATION_SELF_CHECK_OK")
        return
    if not args.runs:
        parser.error("provide at least one run directory")
    for run in map(Path, args.runs):
        plot_run(run, Path(args.output_dir))


if __name__ == "__main__":
    main()
