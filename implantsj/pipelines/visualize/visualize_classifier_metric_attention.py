#!/usr/bin/env python3
"""Save matched ViT attention-rollout samples for metric result grids."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipelines.visualize.visualize_attention import (
    attention_rollout,
    collect_attention,
    denormalize,
    disable_fused_attention,
    save_overlay,
    strip_prefixes,
    tokens_to_heatmap,
    unwrap_state_dict,
)
from src.data import build_transforms
from src.models import create_classifier, load_checkpoint


def choose_samples(rows: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    unique = rows[["path", "true_label"]].drop_duplicates().sort_values("path")
    if len(unique) < count:
        raise ValueError(f"requested {count} samples, but only {len(unique)} are available")
    rng = np.random.default_rng(seed)
    queues = []
    for _, group in unique.groupby("true_label", sort=True):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        queues.append(indices.tolist())
    selected = []
    while len(selected) < count:
        for queue in queues:
            if queue and len(selected) < count:
                selected.append(queue.pop())
    return unique.loc[selected].reset_index(drop=True)


def safe(value: object) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("_") or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-setting", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--discard-ratio", type=float, default=0.0)
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions)
    required = {"dataset", "prototype", "distance", "path", "true_label", "predicted_label", "correct"}
    if missing := required - set(predictions.columns):
        raise ValueError(f"predictions are missing columns: {sorted(missing)}")

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    state = strip_prefixes(unwrap_state_dict(checkpoint))
    checkpoint_args = checkpoint.get("args", {})
    class_to_idx = checkpoint.get("class_to_idx", {})
    model = create_classifier(
        checkpoint_args["model_name"],
        len(class_to_idx),
        pretrained=False,
        dropout=float(checkpoint_args.get("dropout", 0.0)),
        drop_path_rate=float(checkpoint_args.get("drop_path_rate", 0.0)),
    )
    model.load_state_dict(state, strict=True)
    disable_fused_attention(model)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model.to(device).eval()
    image_size = int(checkpoint_args.get("image_size", 224))
    transform = build_transforms(image_size=image_size, train=False)

    index_rows = []
    sample_rows = []
    total_unique = 0
    for dataset_index, (dataset, dataset_rows) in enumerate(predictions.groupby("dataset", sort=True)):
        samples = choose_samples(dataset_rows, args.samples_per_setting, args.seed + dataset_index)
        methods = sorted(dataset_rows[["prototype", "distance"]].drop_duplicates().itertuples(index=False, name=None))
        if methods != sorted((prototype, distance) for prototype in ["gt", "mean", "ss"] for distance in ["euclidean", "cosine"]):
            raise ValueError(f"{dataset}: expected GT/Mean/SS x euclidean/cosine")

        shared = output / str(dataset) / "shared"
        shared.mkdir(parents=True)
        for sample_index, sample in samples.iterrows():
            path = Path(sample.path)
            true_label = str(sample.true_label)
            x = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                _, attentions = collect_attention(model, x)
                rollout = attention_rollout(attentions, args.discard_ratio)[0]
            heat = tokens_to_heatmap(rollout, image_size)
            shared_path = shared / f"{sample_index:03d}_true-{safe(true_label)}_{safe(path.stem)}.png"
            save_overlay(denormalize(x[0]), heat, shared_path, f"{dataset} | true={true_label}")
            sample_rows.append({"dataset": dataset, "sample_index": sample_index, "path": str(path), "true_label": true_label, "shared_file": str(shared_path)})

            image_rows = dataset_rows[dataset_rows.path.astype(str) == str(path)]
            for prototype, distance in methods:
                match = image_rows[(image_rows.prototype == prototype) & (image_rows.distance == distance)]
                if len(match) != 1:
                    raise ValueError(f"{dataset}/{prototype}/{distance}/{path}: expected one prediction")
                prediction = match.iloc[0]
                folder = output / str(dataset) / f"{prototype}_{distance}"
                folder.mkdir(parents=True, exist_ok=True)
                target = folder / (
                    f"{sample_index:03d}_true-{safe(true_label)}_pred-{safe(prediction.predicted_label)}_"
                    f"{'correct' if bool(prediction.correct) else 'wrong'}.png"
                )
                os.link(shared_path, target)
                index_rows.append({
                    "dataset": dataset,
                    "prototype": prototype,
                    "distance": distance,
                    "sample_index": sample_index,
                    "path": str(path),
                    "true_label": true_label,
                    "predicted_label": prediction.predicted_label,
                    "correct": bool(prediction.correct),
                    "attention_file": str(target),
                })
            total_unique += 1
            if total_unique % 25 == 0:
                print(f"attention {total_unique}/{args.samples_per_setting * predictions.dataset.nunique()}", flush=True)

    index = pd.DataFrame(index_rows)
    index.to_csv(output / "attention_index.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(output / "sample_paths.csv", index=False)
    for keys, rows in index.groupby(["dataset", "prototype", "distance"], sort=True):
        dataset, prototype, distance = keys
        rows.to_csv(output / str(dataset) / f"{prototype}_{distance}" / "predictions.csv", index=False)
        png_count = len(list((output / str(dataset) / f"{prototype}_{distance}").glob("*.png")))
        assert png_count == args.samples_per_setting, (keys, png_count)

    metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "predictions": str(Path(args.predictions).resolve()),
        "device": str(device),
        "method": "ViT attention rollout over all blocks, head mean, residual identity",
        "discard_ratio": args.discard_ratio,
        "samples_per_setting": args.samples_per_setting,
        "unique_attention_computations": total_unique,
        "linked_overlays": len(index),
        "note": "Attention depends on the frozen backbone and image, not the prototype or distance; identical maps are hard-linked across matched settings.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    assert len(index) == args.samples_per_setting * predictions.dataset.nunique() * 6
    print(f"ATTENTION_MAPS={len(index)} UNIQUE={total_unique} OUTPUT={output}")


if __name__ == "__main__":
    main()
