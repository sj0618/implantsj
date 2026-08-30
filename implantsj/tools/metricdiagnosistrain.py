#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import load_checkpoint

MODEL = "vit_large_patch16_224.augreg_in21k_ft_in1k"
REQUIRED_SPLITS = {"train", "valid", "test"}
OPEN15 = {"simpleline", "exfeelinternal", "ssiii", "biohorizonexternal", "tsv", "scleantapered", "gsiii", "screwvent", "atlasinternal", "stella", "3i", "ankylos", "iu", "replaceselect", "core1"}
MODEL_KEYS = {
    "metricdiag_fresh_closedft27_e10": "ViT-ClosedFT",
    "metricdiag_fresh_closedft_supcon27_e10": "ViT-ClosedFT-SupCon",
    "metricdiag_fresh_fullft_supcon42_e10": "ViT-FullFT-SupCon",
}


def label_name(value):
    value = re.sub(r"^\s*\d+(?:\s*[.\-_:]\s*|\s+)", "", value)
    value = re.sub(r"\s*\([^)]*\)\s*$", "", value)
    return "".join(character for character in unicodedata.normalize("NFKC", value).casefold() if character.isalnum())


def manifest(path, expected):
    path = path.resolve()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        if not rows.fieldnames or not {"label", "split"}.issubset(rows.fieldnames):
            raise RuntimeError(f"manifest needs label and split columns: {path}")
        label_splits, count = {}, 0
        for row in rows:
            if not row["label"] or not row["split"]:
                raise RuntimeError(f"blank label or split in {path}")
            label_splits.setdefault(row["label"], set()).add(row["split"])
            count += 1
    labels = set(label_splits)
    if len(labels) != expected:
        raise RuntimeError(f"{path}: expected {expected} labels, got {len(labels)}")
    incomplete = {label: sorted(REQUIRED_SPLITS - splits) for label, splits in label_splits.items() if not REQUIRED_SPLITS.issubset(splits)}
    if incomplete:
        raise RuntimeError(f"{path}: labels missing splits {incomplete}")
    if not count:
        raise RuntimeError(f"empty manifest: {path}")
    return path, labels


def gpu():
    result = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    line = next((line for line in result.stdout.splitlines() if line.strip()), "")
    try:
        utilization, memory = [int(float(value.strip())) for value in line.split(",")]
    except Exception as error:
        raise RuntimeError(f"unexpected nvidia-smi output: {line!r}") from error
    return utilization, memory


def wait_for_gpu(seconds):
    while True:
        utilization, memory = gpu()
        print(f"gpu0 utilization={utilization}% memory={memory} MiB", flush=True)
        if utilization <= 10 and memory <= 1500:
            return
        print("gpu0 busy; waiting without stopping any process", flush=True)
        time.sleep(seconds)


def validate_fullft(path, labels, full_manifest):
    if not path.is_file():
        raise RuntimeError(f"existing FullFT checkpoint not found: {path}")
    try:
        checkpoint = load_checkpoint(path, map_location="cpu")
        args = checkpoint["args"]
        classes = checkpoint["class_to_idx"]
        state = checkpoint["model_state_dict"]
    except Exception as error:
        raise RuntimeError(f"unreadable FullFT checkpoint: {path}") from error
    if args.get("model_name") != MODEL:
        raise RuntimeError(f"FullFT model mismatch: {args.get('model_name')}")
    if not isinstance(classes, dict) or set(map(str, classes)) != labels or set(classes.values()) != set(range(42)):
        raise RuntimeError("FullFT checkpoint does not have the exact 42 labels")
    if not isinstance(state, dict) or "head.weight" not in state or any(str(key).startswith("projector.") for key in state):
        raise RuntimeError("FullFT checkpoint is not a classifier checkpoint")
    checkpoint_manifest = Path(args.get("manifest", ""))
    if not checkpoint_manifest.is_absolute():
        checkpoint_manifest = ROOT / checkpoint_manifest
    if checkpoint_manifest.resolve() != full_manifest:
        raise RuntimeError(f"FullFT checkpoint manifest is not {full_manifest}")


def completed(output, name, source_manifest, labels, expected, metric):
    final_metrics = "test_prototype_metrics.json" if metric else "test_metrics.json"
    if not output.is_dir():
        return None
    for run in sorted(output.glob(f"*_{name}"), reverse=True):
        try:
            args = json.loads((run / "args.json").read_text(encoding="utf-8"))
            classes = json.loads((run / "class_to_idx.json").read_text(encoding="utf-8"))
            if Path(args["manifest"]).resolve() != source_manifest or set(classes) != labels:
                continue
            if any(args.get(key) != value for key, value in expected.items()):
                continue
            with (run / "metrics/history.csv").open(newline="", encoding="utf-8") as stream:
                rows = csv.DictReader(stream)
                if not {"epoch", "test_accuracy", "test_macro_f1", "test_weighted_f1"}.issubset(rows.fieldnames or []):
                    continue
                epochs = [int(row["epoch"]) for row in rows]
            needed = [run / "checkpoints/best.pt", run / "checkpoints/last.pt", run / "metrics" / final_metrics]
            if epochs == list(range(1, 11)) and all(path.is_file() for path in needed):
                return run
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return None


def commands(closed, full, output):
    common = ["--model-name", MODEL, "--epochs", "10", "--num-workers", "8", "--weight-decay", "0.05", "--dropout", "0.1", "--drop-path-rate", "0.1", "--seed", "42", "--patience", "10"]
    return [
        (
            "metricdiag_fresh_closedft27_e10", closed, False,
            {"experiment_name": "metricdiag_fresh_closedft27_e10", "model_name": MODEL, "pretrained": True, "image_size": 224, "epochs": 10, "batch_size": 32, "num_workers": 8, "lr": 5e-5, "weight_decay": 0.05, "label_smoothing": 0.1, "dropout": 0.1, "drop_path_rate": 0.1, "seed": 42, "amp": True, "patience": 10, "monitor": "macro_f1", "use_class_weights": False, "strong_aug": True, "allow_cpu": False},
            [sys.executable, "-m", "pipelines.train.train_supervised", "--manifest", str(closed), "--output-root", str(output), "--experiment-name", "metricdiag_fresh_closedft27_e10", "--pretrained", "true", "--batch-size", "32", "--lr", "5e-5", "--label-smoothing", "0.1", "--amp", "true", *common],
        ),
        (
            "metricdiag_fresh_closedft_supcon27_e10", closed, True,
            {"experiment_name": "metricdiag_fresh_closedft_supcon27_e10", "model_name": MODEL, "pretrained": True, "image_size": 224, "epochs": 10, "batch_size": 16, "num_workers": 8, "lr": 3e-5, "weight_decay": 0.05, "loss_name": "supcon", "temperature": 0.07, "embedding_dim": 128, "dropout": 0.1, "drop_path_rate": 0.1, "seed": 42, "amp": True, "patience": 10, "allow_cpu": False},
            [sys.executable, "-m", "pipelines.train.train_metric", "--manifest", str(closed), "--output-root", str(output), "--experiment-name", "metricdiag_fresh_closedft_supcon27_e10", "--batch-size", "16", "--lr", "3e-5", "--loss-name", "supcon", "--temperature", "0.07", "--embedding-dim", "128", *common],
        ),
        (
            "metricdiag_fresh_fullft_supcon42_e10", full, True,
            {"experiment_name": "metricdiag_fresh_fullft_supcon42_e10", "model_name": MODEL, "pretrained": True, "image_size": 224, "epochs": 10, "batch_size": 16, "num_workers": 8, "lr": 3e-5, "weight_decay": 0.05, "loss_name": "supcon", "temperature": 0.07, "embedding_dim": 128, "dropout": 0.1, "drop_path_rate": 0.1, "seed": 42, "amp": True, "patience": 10, "allow_cpu": False},
            [sys.executable, "-m", "pipelines.train.train_metric", "--manifest", str(full), "--output-root", str(output), "--experiment-name", "metricdiag_fresh_fullft_supcon42_e10", "--batch-size", "16", "--lr", "3e-5", "--loss-name", "supcon", "--temperature", "0.07", "--embedding-dim", "128", *common],
        ),
    ]


def main():
    data_root = Path(os.environ.get("DATA_ROOT", "/workspace/data"))
    output_root = Path(os.environ.get("OUTPUT_ROOT", "/workspace/outputs"))
    parser = argparse.ArgumentParser(description="Run the three missing fresh metric-diagnosis trainings safely.")
    parser.add_argument("--closed-manifest", default=str(data_root / "manifests/metric-closed27-corrected.csv"))
    parser.add_argument("--open-manifest", default=str(data_root / "manifests/metric-open15-corrected.csv"))
    parser.add_argument("--full-manifest", default=str(data_root / "manifests/models_vit.csv"))
    parser.add_argument("--output-root", default=str(output_root))
    parser.add_argument("--fullft-checkpoint", default=os.environ.get("FULLFT_CHECKPOINT"))
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.fullft_checkpoint:
        parser.error("--fullft-checkpoint is required (or set FULLFT_CHECKPOINT)")
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be at least 1")

    closed, closed_labels = manifest(Path(args.closed_manifest), 27)
    opened, open_labels = manifest(Path(args.open_manifest), 15)
    full, full_labels = manifest(Path(args.full_manifest), 42)
    if closed_labels & open_labels:
        raise RuntimeError("closed and open labels overlap")
    if closed_labels | open_labels != full_labels:
        raise RuntimeError("closed/open union differs from full labels")
    if {label_name(label) for label in open_labels} != OPEN15:
        raise RuntimeError("open manifest does not match the required 15 English label names")
    if "tsiii" not in {label_name(label) for label in closed_labels} or "ssiii" in {label_name(label) for label in closed_labels}:
        raise RuntimeError("closed manifest must contain TS III and exclude SS III")
    for relative in ["src/supervised.py", "src/metric_learning.py"]:
        source = (ROOT / relative).read_text(encoding="utf-8")
        if "'test_accuracy': test_metrics['accuracy']" in source:
            raise RuntimeError(f"per-epoch test leakage remains in {relative}")
    print(f"validated manifests: closed=27 open=15 full=42 ({opened})")

    output = Path(args.output_root).resolve()
    fullft = Path(args.fullft_checkpoint).resolve()
    if not args.dry_run:
        validate_fullft(fullft, full_labels, full)
        output.mkdir(parents=True, exist_ok=True)
    print(f"reuse existing FullFT: {fullft}")
    selection = {"ViT-FullFT": {"checkpoint": str(fullft), "run_dir": str(fullft.parents[1])}}
    for name, source_manifest, is_metric, expected, command in commands(closed, full, output):
        labels = full_labels if source_manifest == full else closed_labels
        existing = completed(output, name, source_manifest, labels, expected, is_metric)
        if existing:
            print(f"reuse completed matching run: {existing}")
            selection[MODEL_KEYS[name]] = {"checkpoint": str(existing / "checkpoints/best.pt"), "run_dir": str(existing)}
            continue
        print("command:", " ".join(command), flush=True)
        if args.dry_run:
            continue
        wait_for_gpu(args.poll_seconds)
        subprocess.run(command, cwd=ROOT, check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
        result = completed(output, name, source_manifest, labels, expected, is_metric)
        if not result:
            raise RuntimeError(f"training did not produce a complete matching run: {name}")
        selection[MODEL_KEYS[name]] = {"checkpoint": str(result / "checkpoints/best.pt"), "run_dir": str(result)}
        print(f"completed: {result}", flush=True)
    if not args.dry_run:
        path = output / "metricdiag_fresh_checkpoints.json"
        path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
        print(f"checkpoint selection: {path}")


if __name__ == "__main__":
    main()
