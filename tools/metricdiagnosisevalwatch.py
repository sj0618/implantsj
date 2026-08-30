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
DISCOVERY = OUTPUTS / "metricdiag-selection-discovery"
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
PASS = "PASS \u2014 checkpoint \uc5c6\uc74c"


def load_selection(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    checkpoints = {}
    for name in NAMES:
        raw = data.get(name, {}).get("checkpoint")
        if raw:
            checkpoint=Path(raw)
            if not checkpoint.is_file(): raise RuntimeError(f"declared {name} checkpoint is invalid: {checkpoint}")
            checkpoints[name] = checkpoint.resolve()
    return checkpoints


def discover_selection():
    subprocess.run([
        sys.executable,str(ROOT/"pipelines/eval/evaluate_metric_diagnosis.py"),
        "--closed-manifest",str(MANIFESTS["closed"]),"--open-manifest",str(MANIFESTS["open"]),"--full-manifest",str(MANIFESTS["full"]),
        "--checkpoint-root",str(OUTPUTS),"--output-dir",str(DISCOVERY),"--inventory-only",
    ],cwd=ROOT,env={**os.environ,"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"},check=True)
    selected=json.loads((DISCOVERY/"checkpoint_selection.json").read_text(encoding="utf-8")); errors=json.loads((DISCOVERY/"checkpoint_selection_errors.json").read_text(encoding="utf-8"))
    invalid={name:errors.get(name) for name in NAMES if name not in selected and errors.get(name)!=PASS}
    if invalid: raise RuntimeError(f"checkpoint discovery is ambiguous or invalid: {invalid}")
    return {name:Path(selected[name]["checkpoint"]).resolve() for name in NAMES if name in selected}


def merge_selection(declared, discovered):
    return {**discovered,**declared}


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
    marker=path/"completion.json"; fingerprint=path/"run_fingerprint.json"; inventory=path/"checkpoint_selection.json"
    if not marker.is_file() or not fingerprint.is_file() or not inventory.is_file(): return False
    try: completion=json.loads(marker.read_text(encoding="utf-8")); selected=json.loads(inventory.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError): return False
    if completion.get("schema")!=1 or completion.get("fingerprint_sha256")!=digest(fingerprint): return False
    root=path.resolve()
    for relative,expected in completion.get("artifacts",{}).items():
        target=(root/relative).resolve()
        if (target!=root and root not in target.parents) or not target.is_file() or digest(target)!=expected: return False
    return completion.get("result_rows")==completion.get("backbone_rows")==120 and completion.get("selected_rows")==2*len(selected)


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
    available=set(checkpoints)|{"ViT Base"}; missing=set(NAMES)-set(checkpoints)
    def statuses(records):
        return all(len(group:=[row for row in records if row.get("model")==name])==24 and all(row.get("status")==expected for row in group) for name,expected in [(name,"ok") for name in available]+[(name,PASS) for name in missing])
    tsne_ok=all(progress.get("models", {}).get(name, {}).get("tsne", {}).get("status") == "ok" and progress["models"][name]["tsne"].get("sampled")==progress["models"][name]["tsne"].get("total_features") and progress["models"][name]["tsne"].get("max_samples")==0 for name in available)
    return (
        len(rows) == 120
        and statuses(rows)
        and len(backbone_rows) == 120
        and statuses(backbone_rows)
        and set(selected)==available
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
        "--checkpoint-root", str(RESULT),
        "--base-checkpoint", str(base_checkpoint),
        "--gt-zip", str(GT_ZIP),
        "--gt-root", str(DATA / "aihub-gt"),
        "--output-dir", str(RESULT),
        "--batch-size", "64", "--num-workers", "8", "--tsne-max-samples", "0", "--seed", "42",
    ]
    for name, flag in NAMES.items():
        if name in checkpoints: args += [flag, str(checkpoints[name])]
    return args


def self_check():
    checkpoints = {name: Path(f"/tmp/{index}/checkpoints/best.pt") for index, name in enumerate(list(NAMES)[:3])}
    args = command(checkpoints,Path("/tmp/base.pt"))
    assert args[args.index("--tsne-max-samples") + 1] == "0"
    assert args[args.index("--batch-size") + 1] == "64"
    assert args[args.index("--base-checkpoint") + 1] == "/tmp/base.pt"
    assert all(NAMES[name] in args for name in checkpoints) and NAMES["ViT-FullFT-SupCon"] not in args
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory); artifact=root/"REPORT.md"; fingerprint=root/"run_fingerprint.json"; artifact.write_text("ok"); fingerprint.write_text("{}")
        existing=root/"best.pt"; existing.write_bytes(b"x"); selection=root/"selection.json"; selection.write_text(json.dumps({"ViT-ClosedFT":{"checkpoint":str(existing)},"ViT-FullFT-SupCon":{}})); assert set(load_selection(selection))=={"ViT-ClosedFT"}
        assert merge_selection({"ViT-ClosedFT":Path("declared")},{"ViT-ClosedFT":Path("old"),"ViT-FullFT-SupCon":Path("discovered")})=={"ViT-ClosedFT":Path("declared"),"ViT-FullFT-SupCon":Path("discovered")}
        selection.write_text(json.dumps({"ViT-ClosedFT":{"checkpoint":str(root/"missing.pt")}}))
        try: load_selection(selection)
        except RuntimeError: pass
        else: raise AssertionError("invalid declared checkpoint became PASS")
        (root/"checkpoint_selection.json").write_text(json.dumps({"ViT Base":{}}))
        (root/"completion.json").write_text(json.dumps({"schema":1,"result_rows":120,"backbone_rows":120,"selected_rows":2,"fingerprint_sha256":digest(fingerprint),"artifacts":{"REPORT.md":digest(artifact)}}))
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
    while not gpu_idle():
        print("gpu0 busy; waiting without stopping any process", flush=True)
        time.sleep(args.poll_seconds)
    declared=load_selection(SELECTION) if SELECTION.is_file() else {}
    if not declared: print(f"discovering exact-manifest checkpoints under {OUTPUTS}",flush=True)
    checkpoints=merge_selection(declared,discover_selection())
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    os.environ.update({"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"}); base_checkpoint=ensure_base_checkpoint(); fingerprint=run_fingerprint(checkpoints,MANIFESTS,GT_ZIP,CONFIG,base_checkpoint)
    if complete(RESULT,checkpoints,fingerprint): print(f"complete: {RESULT}"); return
    completed=subprocess.run(command(checkpoints,base_checkpoint),cwd=ROOT,env=env)
    if completed.returncode:
        if not complete(RESULT,checkpoints,fingerprint,False,False): completed.check_returncode()
        subprocess.run(command(checkpoints,base_checkpoint)+["--resume-tsne"],cwd=ROOT,env=env,check=True)
    if not complete(RESULT, checkpoints, fingerprint):
        raise RuntimeError(f"evaluation did not cover all 120 rows: {RESULT}")


if __name__ == "__main__":
    main()
