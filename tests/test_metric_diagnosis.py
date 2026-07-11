#!/usr/bin/env python3
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = next(path for path in [Path.cwd(), Path(__file__).resolve().parents[1]] if (path / "src").is_dir())
EVALUATOR = Path(__file__).with_name("evaluator_v5.py")
if not EVALUATOR.is_file(): EVALUATOR = REPO / "pipelines" / "eval" / "evaluate_metric_diagnosis.py"
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("evaluator_v5", EVALUATOR)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def write_manifest(path, labels):
    pd.DataFrame([
        {"path": f"/unused/{label}/{split}.png", "label": label, "split": split}
        for label in labels for split in ("train", "valid", "test")
    ]).to_csv(path, index=False)


def write_checkpoint(path, labels, manifest, supcon=False, score=0.5, epoch=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"projector.3.weight": torch.zeros((2, 2))} if supcon else {"head.weight": torch.zeros((2, 2))}
    torch.save({
        "model_state_dict": state,
        "args": {"manifest": str(manifest), "model_name": "vit_large_patch16_224.augreg_in21k_ft_in1k", "loss_name": "supcon" if supcon else "ce"},
        "class_to_idx": {label: i for i, label in enumerate(labels)},
        "metrics": {"macro_f1": score},
        "epoch": epoch,
    }, path)


def check_normalization_and_gt(root):
    assert ev.normalize_label_name("13.3i(i3)") == "3i"
    assert ev.normalize_label_name("3i") == "3i"
    assert ev.normalize_label_name("23.55 III(55l)") == "55iii"
    assert ev.normalize_label_name("55 III") == "55iii"
    assert ev.normalize_label_name("1 Simpleline (sl)") == "simpleline"
    folders = [root / "1.Simpleline(alias)", root / "duplicate" / "2-Simpleline(copy)", root / "13.3i(i3)", root / "23.55 III(55l)"]
    for i, folder in enumerate(folders):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"image{i}.jpg").write_bytes(b"x")
    mapped, rows = ev.map_gt_label_folders(root, ["1.Simpleline(sl)", "13.3i(i3)", "23.55 III(55l)"])
    assert len(mapped["1.Simpleline(sl)"]) == 2
    assert {row["status"] for row in rows} == {"matched"}


def check_distances_and_sampling():
    query = np.asarray([[1.0, 0.0]])
    prototypes = np.asarray([[10.0, 0.0], [1.0, 1.0]])
    assert ev.prototype_predict(query, prototypes, ["aligned", "near"], "cosine")[0].tolist() == ["aligned"]
    assert ev.prototype_predict(query, prototypes, ["aligned", "near"], "euclidean")[0].tolist() == ["near"]
    rows = [(part, label, split, f"{part}-{label}-{split}-{i}", np.asarray([i])) for part in ("closed", "open") for label in ("a", "b") for split in ("train", "valid", "test") for i in range(3)]
    first = ev.stratified_rows(rows, 12, 7); second = ev.stratified_rows(rows, 12, 7)
    assert [row[3] for row in first] == [row[3] for row in second]
    assert len({row[:3] for row in first}) == 12
    assert len(ev.MODELS) * len(ev.PROTOS) * len(ev.DISTS) * 2 * len(ev.SPLITS) == 120


def check_visual_artifacts(root):
    tsne = root / "tsne"; curves = root / "curves"; tsne.mkdir(parents=True); curves.mkdir()
    rng = np.random.default_rng(3); caches = {}
    for part in ("closed", "open"):
        labels=[]; splits=[]; paths=[]
        for label in (f"{part}0", f"{part}1"):
            for split in ("train", "valid", "test"):
                for i in range(3): labels.append(label); splits.append(split); paths.append(f"/{part}/{label}/{split}/{i}.jpg")
        caches[part] = {"features": rng.normal(size=(len(labels), 8)).astype(np.float32), "labels": np.asarray(labels, object), "splits": np.asarray(splits, object), "paths": np.asarray(paths, object)}
    meta = ev.save_tsne("Synthetic", caches, tsne, 42, 24)
    assert meta["total_features"] == 36 and meta["sampled"] == 24
    assert (tsne / "Synthetic.png").is_file() and (tsne / "Synthetic_valid.png").is_file() and (tsne / "Synthetic.sampling.json").is_file()
    checkpoint = root / "run" / "checkpoints" / "best.pt"; checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b"x")
    history = checkpoint.parent.parent / "metrics" / "history.csv"; history.parent.mkdir()
    pd.DataFrame({"epoch":[1,2],"train_loss":[1.0,.5],"valid_accuracy":[.4,.6],"test_accuracy":[.3,.5],"valid_macro_f1":[.35,.55],"test_macro_f1":[.25,.45]}).to_csv(history,index=False)
    curve = ev.save_curve("Synthetic", {"checkpoint":str(checkpoint)}, curves)
    assert curve["status"] == "ok" and Path(curve["png"]).is_file() and Path(curve["csv"]).is_file()
    pd.DataFrame({"epoch":[1,2],"train_loss":[1.0,.5],"valid_accuracy":[.4,.6],"valid_macro_f1":[.35,.55]}).to_csv(history,index=False)
    partial = ev.save_curve("Synthetic partial", {"checkpoint":str(checkpoint)}, curves)
    assert partial["status"] == "partial" and "test_accuracy" in partial["reason"] and "test_macro_f1" in partial["reason"]


def check_inventory(root):
    manifests = root / "manifests"; checkpoints = root / "checkpoints"; manifests.mkdir(parents=True); checkpoints.mkdir()
    closed = ["20.TS III(ts3)"] + [f"closed{i:02d}" for i in range(26)]
    opened = [f"open{i:02d}" for i in range(15)]; full = closed + opened
    closed_manifest = manifests / "closed27.csv"; open_manifest = manifests / "open15.csv"; full_manifest = manifests / "full42.csv"; old26_manifest = manifests / "old26.csv"
    write_manifest(closed_manifest, closed); write_manifest(open_manifest, opened); write_manifest(full_manifest, full); write_manifest(old26_manifest, closed[1:])
    old_high = checkpoints / "old_high" / "best.pt"; old26 = checkpoints / "old26" / "best.pt"
    fresh_closed = checkpoints / "fresh_closed" / "best.pt"; fresh_closed_supcon = checkpoints / "fresh_closed_supcon" / "best.pt"
    fresh_full = checkpoints / "fresh_full" / "best.pt"; fresh_full_supcon = checkpoints / "fresh_full_supcon" / "best.pt"
    write_checkpoint(old_high, closed, closed_manifest, score=0.99, epoch=5); write_checkpoint(old26, closed[1:], old26_manifest, score=1.0)
    write_checkpoint(fresh_closed, closed, closed_manifest, score=0.1); write_checkpoint(fresh_closed_supcon, closed, closed_manifest, True, score=0.1)
    write_checkpoint(fresh_full, full, full_manifest, score=0.1); write_checkpoint(fresh_full_supcon, full, full_manifest, True, score=0.1)
    output = root / "inventory"
    cmd = [
        sys.executable, str(EVALUATOR), "--closed-manifest", str(closed_manifest), "--open-manifest", str(open_manifest), "--full-manifest", str(full_manifest),
        "--checkpoint-root", str(checkpoints), "--closed-checkpoint", str(fresh_closed), "--closed-supcon-checkpoint", str(fresh_closed_supcon),
        "--full-checkpoint", str(fresh_full), "--full-supcon-checkpoint", str(fresh_full_supcon), "--output-dir", str(output), "--inventory-only",
    ]
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    completed = subprocess.run(cmd, cwd=REPO, env=env, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    selected = json.loads((output / "checkpoint_selection.json").read_text())
    assert set(selected) == set(ev.MODELS)
    assert Path(selected["ViT-ClosedFT"]["checkpoint"]) == fresh_closed.resolve()
    assert all(selected[name]["selection_basis"].startswith("explicit ") for name in ev.MODELS[1:])
    assert selected["ViT Base"]["kind"] == "pretrained_base"
    bad_output = root / "bad_inventory"
    bad = subprocess.run([sys.executable, str(EVALUATOR), "--closed-manifest", str(old26_manifest), "--open-manifest", str(open_manifest), "--full-manifest", str(full_manifest), "--checkpoint-root", str(checkpoints), "--output-dir", str(bad_output), "--inventory-only"], cwd=REPO, env=env, text=True, capture_output=True)
    assert bad.returncode != 0
    audit = json.loads((bad_output / "manifest_audit.json").read_text())
    assert "closed label count is not 27" in audit["errors"]

    originals = ev.make_model, ev.extract, ev.map_gt_label_folders, ev.save_tsne, ev.save_curve
    gt_root = root / "gt"; gt_root.mkdir(); (gt_root / "ready").write_text("x")
    def fake_map(_root, labels, report_csv=None):
        mapped = {label: [Path(f"/unused/gt/{label}.jpg")] for label in labels}
        return mapped, [{"label": label, "status": "matched"} for label in labels]
    def fake_extract(_model, _forward, manifest, _size, _batch, _workers, _device):
        frame = pd.read_csv(manifest); values = frame.label.astype(str).tolist()
        features = np.asarray([[sum(map(ord, label)) % 997, (sum(map(ord, label)) ** 2) % 991] for label in values], dtype=np.float32)
        return {"features": features, "labels": np.asarray(values, dtype=object), "splits": frame.split.astype(str).to_numpy(object), "paths": frame.path.astype(str).to_numpy(object)}
    ev.map_gt_label_folders = fake_map
    ev.extract = fake_extract
    ev.save_tsne = lambda name, caches, out, seed, limit: {"total_features": sum(len(c["labels"]) for c in caches.values()), "sampled": 10, "max_samples": limit}
    ev.save_curve = lambda name, item, out: {"model": name, "history": "", "status": "not_applicable", "png": ""}
    base_args = [
        "evaluator_v5.py", "--closed-manifest", str(closed_manifest), "--open-manifest", str(open_manifest), "--full-manifest", str(full_manifest),
        "--checkpoint-root", str(checkpoints), "--closed-checkpoint", str(fresh_closed), "--closed-supcon-checkpoint", str(fresh_closed_supcon),
        "--full-checkpoint", str(fresh_full), "--full-supcon-checkpoint", str(fresh_full_supcon), "--gt-zip", str(root / "unused.zip"), "--gt-root", str(gt_root), "--allow-cpu",
    ]
    old_argv = sys.argv
    try:
        ev.make_model = lambda item, device: (_ for _ in ()).throw(RuntimeError("base unavailable")) if item["kind"] == "pretrained_base" else (object(), lambda x: x, 224)
        failed_output = root / "failed_full"; sys.argv = base_args + ["--output-dir", str(failed_output)]
        try: ev.main()
        except RuntimeError as error: assert "96/120" in str(error)
        failed_summary = pd.read_csv(failed_output / "summary.csv")
        assert len(failed_summary) == 120 and (failed_summary.status == "ok").sum() == 96
        failed_progress = json.loads((failed_output / "progress.json").read_text())
        assert failed_progress["tsne_max_samples"] == 0
        assert failed_progress["models"]["ViT Base"]["status"] == "unavailable"
        assert all(failed_progress["models"][name]["status"] == "ok" for name in ev.MODELS[1:])
        ev.make_model = lambda item, device: (object(), lambda x: x, 224)
        complete_output = root / "complete_full"; sys.argv = base_args + ["--output-dir", str(complete_output)]
        ev.main()
        complete_summary = pd.read_csv(complete_output / "summary.csv")
        assert len(complete_summary) == 120 and (complete_summary.status == "ok").sum() == 120
        report = (complete_output / "REPORT.md").read_text()
        assert "t-SNE: all features" in report and "## Curve coverage" in report
        assert "## Mean performance by partition" in report and "### Closed" in report and "### Open" in report
        closed_section = report.split("### Closed", 1)[1].split("### Open", 1)[0]; open_section = report.split("### Open", 1)[1].split("## Matched model comparisons", 1)[0]
        assert all(name in closed_section and name in open_section for name in ev.MODELS)
        assert report.count("  - closed:") == 4 and report.count("  - open:") == 4
        assert all(f"- {scope} best {setting}:" in report for scope in ("Overall", "Closed", "Open") for setting in ("distance", "prototype"))
    finally:
        sys.argv = old_argv
        ev.make_model, ev.extract, ev.map_gt_label_folders, ev.save_tsne, ev.save_curve = originals


def test_metric_diagnosis():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        check_normalization_and_gt(root / "gt")
        check_distances_and_sampling()
        check_visual_artifacts(root / "visuals")
        check_inventory(root / "inventory_case")


def main():
    test_metric_diagnosis()
    print("EVALUATOR_V5_TESTS_OK")


if __name__ == "__main__":
    main()
