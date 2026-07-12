#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = Path("/home/ubuntu/workspace/outputs")
DATA = Path("/home/ubuntu/workspace/data")
SELECTION = OUTPUTS / "metricdiag_fresh_checkpoints.json"
RESULT = OUTPUTS / "metricdiag-final"
NAMES = {
    "ViT-ClosedFT": "--closed-checkpoint",
    "ViT-ClosedFT-SupCon": "--closed-supcon-checkpoint",
    "ViT-FullFT": "--full-checkpoint",
    "ViT-FullFT-SupCon": "--full-supcon-checkpoint",
}
FRESH = set(NAMES) - {"ViT-FullFT"}


def load_selection(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    checkpoints = {}
    for name in NAMES:
        checkpoint = Path(data.get(name, {}).get("checkpoint", ""))
        if not checkpoint.is_file():
            raise RuntimeError(f"missing {name} checkpoint: {checkpoint}")
        if name in FRESH:
            history = checkpoint.parents[1] / "metrics/history.csv"
            with history.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if not {"epoch", "test_accuracy", "test_macro_f1"}.issubset(reader.fieldnames or []):
                    raise RuntimeError(f"{name} has incomplete curves: {history}")
                rows = list(reader)
            if [int(row.get("epoch", -1)) for row in rows] != list(range(1, 11)):
                raise RuntimeError(f"{name} is not a complete 10-epoch run: {history}")
        checkpoints[name] = checkpoint.resolve()
    return checkpoints


def gpu_idle():
    result = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    utilization, memory = [int(float(value.strip())) for value in result.stdout.splitlines()[0].split(",")]
    print(f"gpu0 utilization={utilization}% memory={memory} MiB", flush=True)
    return utilization <= 10 and memory <= 1500


def complete(path, checkpoints):
    summary = path / "summary.csv"
    inventory = path / "checkpoint_selection.json"
    if not summary.is_file() or not inventory.is_file():
        return False
    with summary.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    try:
        selected = json.loads(inventory.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        len(rows) == 120
        and all(row.get("status") == "ok" for row in rows)
        and all(Path(selected.get(name, {}).get("checkpoint", "")).resolve() == checkpoint for name, checkpoint in checkpoints.items())
    )


def command(checkpoints):
    args = [
        sys.executable, str(ROOT / "pipelines/eval/evaluate_metric_diagnosis.py"),
        "--closed-manifest", str(DATA / "manifests/metric-closed27-corrected.csv"),
        "--open-manifest", str(DATA / "manifests/metric-open15-corrected.csv"),
        "--full-manifest", str(DATA / "manifests/models_vit.csv"),
        "--checkpoint-root", str(checkpoints["ViT-ClosedFT"].parents[1]),
        "--gt-zip", str(DATA / "AI Hub Data.zip"),
        "--gt-root", str(DATA / "aihub-gt"),
        "--output-dir", str(RESULT),
        "--batch-size", "64", "--num-workers", "8", "--tsne-max-samples", "0", "--seed", "42",
    ]
    for name, flag in NAMES.items():
        args += [flag, str(checkpoints[name])]
    return args


def self_check():
    checkpoints = {name: Path(f"/tmp/{index}/checkpoints/best.pt") for index, name in enumerate(NAMES)}
    args = command(checkpoints)
    assert args[args.index("--tsne-max-samples") + 1] == "0"
    assert args[args.index("--batch-size") + 1] == "64"
    assert all(flag in args for flag in NAMES.values())
    print("METRIC_DIAGNOSIS_EVAL_WATCH_OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be at least 1")
    while True:
        try:
            checkpoints = load_selection(SELECTION)
            break
        except (OSError, ValueError, RuntimeError) as error:
            print(f"waiting for completed checkpoints: {error}", flush=True)
            time.sleep(args.poll_seconds)
    if complete(RESULT, checkpoints):
        print(f"complete: {RESULT}")
        return
    while not gpu_idle():
        print("gpu0 busy; waiting without stopping any process", flush=True)
        time.sleep(args.poll_seconds)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    subprocess.run(command(checkpoints), cwd=ROOT, env=env, check=True)
    if not complete(RESULT, checkpoints):
        raise RuntimeError(f"evaluation did not produce 120 successful rows: {RESULT}")


if __name__ == "__main__":
    main()
