#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipelines.eval.evaluate_metric_diagnosis import digest, run_fingerprint

OUTPUTS = Path("/home/ubuntu/workspace/outputs")
DATA = Path("/home/ubuntu/workspace/data")
SELECTION = OUTPUTS / "metricdiag_fresh_checkpoints.json"
RESULT = OUTPUTS / "metricdiag-final"
BASE = OUTPUTS / "metricdiag-base" / "vit-large-pretrained.pt"
MANIFESTS = {"closed":DATA/"manifests/metric-closed27-corrected.csv","open":DATA/"manifests/metric-open15-corrected.csv","full":DATA/"manifests/models_vit.csv"}
GT_ZIP = DATA / "AI Hub Data.zip"
CONFIG = {"base_model_name":"vit_large_patch16_224.augreg_in21k_ft_in1k","batch_size":64,"num_workers":8,"seed":42,"ss_per_label":5,"tsne_max_samples":0,"representations":"deployed_and_common_backbone_v1"}
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


def ensure_base_checkpoint():
    if BASE.is_file():
        return BASE.resolve()
    import torch
    from src.models import create_classifier
    BASE.parent.mkdir(parents=True,exist_ok=True); temporary=BASE.with_suffix(".tmp")
    model=create_classifier(CONFIG["base_model_name"],0,True); torch.save({"model_state_dict":model.state_dict()},temporary); os.replace(temporary,BASE)
    return BASE.resolve()


def completion_valid(path):
    marker=path/"completion.json"; fingerprint=path/"run_fingerprint.json"
    if not marker.is_file() or not fingerprint.is_file(): return False
    try: completion=json.loads(marker.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError): return False
    if completion.get("schema")!=1 or completion.get("fingerprint_sha256")!=digest(fingerprint): return False
    root=path.resolve()
    for relative,expected in completion.get("artifacts",{}).items():
        target=(root/relative).resolve()
        if (target!=root and root not in target.parents) or not target.is_file() or digest(target)!=expected: return False
    return completion.get("result_rows")==completion.get("backbone_rows")==120 and completion.get("selected_rows")==10


def complete(path, checkpoints, fingerprint, require_tsne=True, require_completion=True):
    summary = path / "summary.csv"
    backbone = path / "backbone_summary.csv"
    inventory = path / "checkpoint_selection.json"
    progress_path = path / "progress.json"
    provenance_path = path / "provenance_audit.json"
    fingerprint_path = path / "run_fingerprint.json"
    if not all(item.is_file() for item in [summary, backbone, inventory, progress_path, provenance_path, fingerprint_path]):
        return False
    with summary.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with backbone.open(newline="", encoding="utf-8") as stream:
        backbone_rows = list(csv.DictReader(stream))
    try:
        selected = json.loads(inventory.read_text(encoding="utf-8"))
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        recorded_fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    tsne_ok=all(progress.get("models", {}).get(name, {}).get("tsne", {}).get("status") == "ok" and progress["models"][name]["tsne"].get("sampled")==progress["models"][name]["tsne"].get("total_features") and progress["models"][name]["tsne"].get("max_samples")==0 for name in set(NAMES) | {"ViT Base"})
    return (
        len(rows) == 120
        and all(row.get("status") == "ok" for row in rows)
        and len(backbone_rows) == 120
        and all(row.get("status") == "ok" for row in backbone_rows)
        and all(Path(selected.get(name, {}).get("checkpoint", "")).resolve() == checkpoint for name, checkpoint in checkpoints.items())
        and (tsne_ok or not require_tsne)
        and provenance.get("manifests", {}).get("row_union_exact") is True
        and provenance.get("content", {}).get("conflicts") == 0
        and recorded_fingerprint == fingerprint
        and (completion_valid(path) or not require_completion)
    )


def command(checkpoints, base_checkpoint):
    args = [
        sys.executable, str(ROOT / "pipelines/eval/evaluate_metric_diagnosis.py"),
        "--closed-manifest", str(MANIFESTS["closed"]),
        "--open-manifest", str(MANIFESTS["open"]),
        "--full-manifest", str(MANIFESTS["full"]),
        "--checkpoint-root", str(checkpoints["ViT-ClosedFT"].parents[1]),
        "--base-checkpoint", str(base_checkpoint),
        "--gt-zip", str(GT_ZIP),
        "--gt-root", str(DATA / "aihub-gt"),
        "--output-dir", str(RESULT),
        "--batch-size", "64", "--num-workers", "8", "--tsne-max-samples", "0", "--seed", "42",
    ]
    for name, flag in NAMES.items():
        args += [flag, str(checkpoints[name])]
    return args


def self_check():
    checkpoints = {name: Path(f"/tmp/{index}/checkpoints/best.pt") for index, name in enumerate(NAMES)}
    args = command(checkpoints,Path("/tmp/base.pt"))
    assert args[args.index("--tsne-max-samples") + 1] == "0"
    assert args[args.index("--batch-size") + 1] == "64"
    assert args[args.index("--base-checkpoint") + 1] == "/tmp/base.pt"
    assert all(flag in args for flag in NAMES.values())
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory); artifact=root/"REPORT.md"; fingerprint=root/"run_fingerprint.json"; artifact.write_text("ok"); fingerprint.write_text("{}")
        (root/"completion.json").write_text(json.dumps({"schema":1,"result_rows":120,"backbone_rows":120,"selected_rows":10,"fingerprint_sha256":digest(fingerprint),"artifacts":{"REPORT.md":digest(artifact)}}))
        assert completion_valid(root); artifact.write_text("changed"); assert not completion_valid(root)
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
    while not gpu_idle():
        print("gpu0 busy; waiting without stopping any process", flush=True)
        time.sleep(args.poll_seconds)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    os.environ.update({"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"}); base_checkpoint=ensure_base_checkpoint(); fingerprint=run_fingerprint(checkpoints,MANIFESTS,GT_ZIP,CONFIG,base_checkpoint)
    if complete(RESULT,checkpoints,fingerprint): print(f"complete: {RESULT}"); return
    completed=subprocess.run(command(checkpoints,base_checkpoint),cwd=ROOT,env=env)
    if completed.returncode:
        if not complete(RESULT,checkpoints,fingerprint,False,False): completed.check_returncode()
        subprocess.run(command(checkpoints,base_checkpoint)+["--resume-tsne"],cwd=ROOT,env=env,check=True)
    if not complete(RESULT, checkpoints, fingerprint):
        raise RuntimeError(f"evaluation did not produce 120 successful rows: {RESULT}")


if __name__ == "__main__":
    main()
