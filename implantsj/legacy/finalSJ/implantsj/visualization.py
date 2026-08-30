"""Deterministic t-SNE and training-curve visualizations."""

from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from .evaluation import EmbeddingTable, load_labels


def parse_named_paths(values: Sequence[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        result.append((name.strip(), Path(raw_path).expanduser().resolve()))
    return result


def deterministic_sample(
    frame: pd.DataFrame, max_points: int, seed: int
) -> pd.DataFrame:
    if max_points <= 0 or len(frame) <= max_points:
        return frame.copy()
    grouped = list(frame.groupby(["split", "set", "label"], dropna=False, sort=True))
    quota = max(1, max_points // max(len(grouped), 1))
    rng = random.Random(seed)
    selected: list[int] = []
    leftovers: list[int] = []
    for _, group in grouped:
        ranked = sorted(group.index, key=lambda index: str(frame.loc[index, "path"]))
        rng.shuffle(ranked)
        selected.extend(ranked[:quota])
        leftovers.extend(ranked[quota:])
    if len(selected) > max_points:
        rng.shuffle(selected)
        selected = selected[:max_points]
    elif len(selected) < max_points:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max_points - len(selected)])
    return frame.loc[selected].reset_index(drop=True)


def create_tsne(args: argparse.Namespace) -> Path:
    frames: list[pd.DataFrame] = []
    features: list[np.ndarray] = []
    inputs = parse_named_paths(args.embedding)
    all_labels, closed, opened = load_labels(args.labels_json)
    for split_name, path in inputs:
        table = EmbeddingTable.load(path)
        frame = pd.DataFrame(
            {
                "path": table.paths,
                "label": table.labels,
                "split": split_name,
                "set": table.sets,
                "source_index": np.arange(len(table.labels)),
                "source_path": str(path),
            }
        )
        frames.append(frame)
        features.append(table.features)
    combined = pd.concat(frames, ignore_index=True)
    unknown = sorted(set(combined["label"]) - set(all_labels))
    if unknown:
        raise ValueError(f"embeddings contain labels outside labels.json: {unknown}")
    set_by_label = {label: "closed" for label in closed} | {
        label: "open" for label in opened
    }
    combined["set"] = combined["label"].map(set_by_label)
    combined_features = np.concatenate(features, axis=0)
    combined["combined_index"] = np.arange(len(combined))
    sampled = deterministic_sample(combined, args.max_points, args.seed)
    selected_features = combined_features[sampled["combined_index"].to_numpy()]
    if len(sampled) < 3:
        raise ValueError("t-SNE requires at least three samples")
    perplexity = min(args.perplexity, max(2.0, (len(sampled) - 1) / 3))
    tsne_options = {
        "n_components": 2,
        "perplexity": perplexity,
        "init": "pca",
        "learning_rate": "auto",
        "random_state": args.seed,
        (
            "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
        ): args.iterations,
    }
    coordinates = TSNE(**tsne_options).fit_transform(selected_features)
    sampled["x"] = coordinates[:, 0]
    sampled["y"] = coordinates[:, 1]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    coordinates_path = output_dir / "tsne_coordinates.csv"
    sampled.loc[:, ["path", "label", "split", "set", "x", "y"]].to_csv(
        coordinates_path, index=False
    )
    (output_dir / "sampled_paths.txt").write_text(
        "\n".join(sampled["path"].astype(str)) + "\n", encoding="utf-8"
    )

    labels = [label for label in all_labels if label in set(sampled["label"])]
    cmap = plt.get_cmap("gist_ncar", max(len(labels), 1))
    colors = {
        label: "#d62728" if label == "ITI" else cmap(index)
        for index, label in enumerate(labels)
    }
    markers = {"train": "o", "valid": "^", "test": "s", "gt": "P"}
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)
    panels = (
        ("all", sampled),
        ("closed", sampled[sampled["set"] == "closed"]),
        ("open", sampled[sampled["set"] == "open"]),
    )
    for axis, (title, panel) in zip(axes, panels):
        for (label, split), group in panel.groupby(["label", "split"], sort=True):
            axis.scatter(
                group["x"],
                group["y"],
                s=12,
                alpha=0.65,
                c=[colors[label]],
                marker=markers.get(str(split), "o"),
                label=f"{label} [{split}]",
                linewidths=0,
            )
        axis.set_title(f"{args.model_name} — {title} ({len(panel)})")
        axis.set_xticks([])
        axis.set_yticks([])
    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            fontsize=7,
            ncol=1,
        )
    figure_path = output_dir / "tsne.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "model": args.model_name,
        "total_embeddings": len(combined),
        "plotted_embeddings": len(sampled),
        "max_points": args.max_points,
        "seed": args.seed,
        "perplexity": perplexity,
        "iterations": args.iterations,
        "inputs": [{"split": name, "path": str(path)} for name, path in inputs],
        "labels_json": str(Path(args.labels_json).expanduser().resolve()),
        "note": "All embeddings remain in NPZ caches; the plot uses a deterministic stratified subset when max_points is exceeded.",
    }
    (output_dir / "tsne_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return figure_path


def create_curves(args: argparse.Namespace) -> Path:
    histories = parse_named_paths(args.history)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for name, path in histories:
        frame = pd.read_csv(path)
        if "epoch" not in frame:
            raise ValueError(f"history {path} has no epoch column")
        for column, linestyle in (
            ("train_loss", "-"),
            ("valid_loss", "--"),
            ("test_loss", ":"),
        ):
            if column in frame:
                axes[0].plot(
                    frame["epoch"],
                    frame[column],
                    linestyle=linestyle,
                    label=f"{name} {column}",
                )
        for column, linestyle in (
            ("train_accuracy", "-"),
            ("valid_accuracy", "--"),
            ("test_accuracy", ":"),
        ):
            if column in frame and frame[column].notna().any():
                axes[1].plot(
                    frame["epoch"],
                    frame[column],
                    linestyle=linestyle,
                    label=f"{name} {column}",
                )
    axes[0].set(title="Train / validation / test loss", xlabel="epoch", ylabel="loss")
    axes[1].set(
        title="Train / validation / test accuracy",
        xlabel="epoch",
        ylabel="accuracy",
        ylim=(0, 1),
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    tsne = commands.add_parser("tsne")
    tsne.add_argument("--model-name", required=True)
    tsne.add_argument(
        "--embedding", action="append", required=True, help="repeat SPLIT=cache.npz"
    )
    tsne.add_argument("--labels-json", required=True)
    tsne.add_argument("--output-dir", required=True)
    tsne.add_argument("--max-points", type=int, default=5000, help="0 means all points")
    tsne.add_argument("--perplexity", type=float, default=30.0)
    tsne.add_argument("--iterations", type=int, default=1000)
    tsne.add_argument("--seed", type=int, default=2026)
    curves = commands.add_parser("curves")
    curves.add_argument(
        "--history", action="append", required=True, help="repeat MODEL=history.csv"
    )
    curves.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = create_tsne(args) if args.command == "tsne" else create_curves(args)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
