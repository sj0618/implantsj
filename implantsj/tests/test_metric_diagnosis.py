#!/usr/bin/env python3
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = next(path for path in [Path.cwd(), Path(__file__).resolve().parents[1]] if (path / "src").is_dir())
EVALUATOR = REPO / "pipelines" / "eval" / "evaluate_metric_diagnosis.py"
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("evaluator_v5", EVALUATOR)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def write_manifest(path, labels):
    images = path.parent / "images"; images.mkdir(exist_ok=True)
    rows = []
    for label in labels:
        for split in ("train", "valid", "test"):
            image = images / f"{ev.normalize_label_name(label)}-{split}.png"; image.write_bytes(f"{label}:{split}".encode())
            rows.append({"path": str(image.resolve()), "label": label, "split": split})
    pd.DataFrame(rows).to_csv(path, index=False)


def write_checkpoint(path, labels, manifest, supcon=False, score=0.5, epoch=10, history_epochs=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"projector.3.weight": torch.zeros((2, 2))} if supcon else {"head.weight": torch.zeros((2, 2))}
    torch.save({
        "model_state_dict": state,
        "args": {"manifest": str(manifest), "model_name": "vit_large_patch16_224.augreg_in21k_ft_in1k", "loss_name": "supcon" if supcon else "ce"},
        "class_to_idx": {label: i for i, label in enumerate(labels)},
        "metrics": {"macro_f1": score},
        "epoch": epoch,
    }, path)
    history = path.parent.parent / "metrics" / "history.csv"; history.parent.mkdir(exist_ok=True)
    pd.DataFrame({"epoch":range(1,history_epochs+1),"train_loss":np.linspace(1,.1,history_epochs),"valid_accuracy":np.linspace(.1,.9,history_epochs),"test_accuracy":np.linspace(.1,.8,history_epochs),"valid_macro_f1":np.linspace(.1,.9,history_epochs),"test_macro_f1":np.linspace(.1,.8,history_epochs)}).to_csv(history,index=False)


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
    leak=root/"leak"; leak.mkdir(); first=leak/"train.jpg"; second=leak/"test.jpg"; gt_image=leak/"gt.jpg"; first.write_bytes(b"same"); second.write_bytes(b"same"); gt_image.write_bytes(b"gt")
    frames={"closed":pd.DataFrame([{"path":first,"label":"a","split":"train"},{"path":second,"label":"a","split":"test"}]),"open":pd.DataFrame(columns=["path","label","split"])}
    try: ev.content_audit(frames,{"a":[gt_image]},leak)
    except RuntimeError as error: assert "leakage" in str(error)
    else: raise AssertionError("content leakage was not rejected")


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


def check_checkpoint_provenance(root):
    from src.models import save_checkpoint
    manifest=root/"manifest.csv"; manifest.parent.mkdir(parents=True); manifest.write_text("path,label,split\n/a,a,train\n")
    checkpoint=root/"best.pt"; save_checkpoint(checkpoint,torch.nn.Linear(2,2),None,1,{}, {"manifest":str(manifest)}, {"a":0})
    assert torch.load(checkpoint,map_location="cpu")["args"]["manifest_sha256"]==ev.digest(manifest)


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
    assert meta["partitions_overlaid"] is True
    assert set(meta["axis_limits"]) == {"x", "y"} and json.loads((tsne / "Synthetic_common_backbone.sampling.json").read_text())["axis_limits"] == {key:list(value) for key,value in meta["axis_limits"].items()}
    assert (tsne / "Synthetic_common_backbone.png").is_file() and (tsne / "Synthetic_common_backbone_valid.png").is_file() and (tsne / "Synthetic_common_backbone.sampling.json").is_file()
    gt = {key:np.concatenate([c[key] for c in caches.values()]) for key in ("features", "labels", "paths")}
    prototype = ev.save_prototype_tsne("Synthetic", caches, gt, tsne / "prototype", 42, 2)
    assert prototype["partitions_overlaid"] is True and Path(prototype["png"]).is_file()
    checkpoint = root / "run" / "checkpoints" / "best.pt"; checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b"x")
    history = checkpoint.parent.parent / "metrics" / "history.csv"; history.parent.mkdir()
    pd.DataFrame({"epoch":[1,2],"train_loss":[1.0,.5],"valid_accuracy":[.4,.6],"test_accuracy":[.3,.5],"valid_macro_f1":[.35,.55],"test_macro_f1":[.25,.45]}).to_csv(history,index=False)
    curve = ev.save_curve("Synthetic", {"checkpoint":str(checkpoint)}, curves)
    assert curve["status"] == "partial" and "2/10 epochs" in curve["reason"] and Path(curve["png"]).is_file() and Path(curve["csv"]).is_file()
    assert ev.save_curve("ViT Base", {"kind":"base","checkpoint":str(checkpoint)}, curves)["status"] == "N/A"
    pd.DataFrame({"epoch":[1,2],"train_loss":[1.0,.5],"valid_accuracy":[.4,.6],"valid_macro_f1":[.35,.55]}).to_csv(history,index=False)
    partial = ev.save_curve("Synthetic partial", {"checkpoint":str(checkpoint)}, curves)
    assert partial["status"] == "partial" and "2/10 epochs" in partial["reason"]
    assert "test_accuracy" not in partial["reason"] and "test_macro_f1" not in partial["reason"]


def check_inventory(root):
    manifests = root / "manifests"; checkpoints = root / "checkpoints"; manifests.mkdir(parents=True); checkpoints.mkdir()
    closed = ["20.TS III(ts3)"] + [f"closed{i:02d}" for i in range(26)]
    opened = ["Simpleline","exfeel internal","SS III","Biohorizon external","TSV","S clean tapered","GS III","screw vent","atlas internal","stella","3i","ankylos","iu","replace select","core1"]; full = closed + opened
    closed_manifest = manifests / "closed27.csv"; open_manifest = manifests / "open15.csv"; full_manifest = manifests / "full42.csv"; old26_manifest = manifests / "old26.csv"
    write_manifest(closed_manifest, closed); write_manifest(open_manifest, opened); write_manifest(full_manifest, full); write_manifest(old26_manifest, closed[1:])
    old_high = checkpoints / "old_high" / "best.pt"; old26 = checkpoints / "old26" / "best.pt"
    fresh_closed = checkpoints / "fresh_closed" / "best.pt"; fresh_closed_supcon = checkpoints / "fresh_closed_supcon" / "best.pt"
    fresh_full = checkpoints / "fresh_full" / "best.pt"; fresh_full_supcon = checkpoints / "fresh_full_supcon" / "best.pt"
    write_checkpoint(old_high, closed, closed_manifest, score=0.99, epoch=5, history_epochs=5); write_checkpoint(old26, closed[1:], old26_manifest, score=1.0)
    write_checkpoint(fresh_closed, closed, closed_manifest, score=0.1); write_checkpoint(fresh_full, full, full_manifest, score=0.1); write_checkpoint(fresh_full_supcon, full, full_manifest, True, score=0.1)
    write_checkpoint(fresh_closed_supcon, closed, closed_manifest, True, score=0.1, epoch=3, history_epochs=3)
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
    assert selected["ViT-ClosedFT-SupCon"]["history_epochs"] == [1,2,3]
    assert all(selected[name]["selection_basis"].startswith("explicit ") for name in ev.MODELS[1:])
    assert selected["ViT Base"]["kind"] == "pretrained_base"
    fallback_output=root/"missing_declared_but_discovered"; fallback=subprocess.run([sys.executable,str(EVALUATOR),"--closed-manifest",str(closed_manifest),"--open-manifest",str(open_manifest),"--full-manifest",str(full_manifest),"--checkpoint-root",str(checkpoints),"--full-supcon-checkpoint",str(root/"absent.pt"),"--output-dir",str(fallback_output),"--inventory-only"],cwd=REPO,env=env,text=True,capture_output=True)
    assert fallback.returncode==0 and Path(json.loads((fallback_output/"checkpoint_selection.json").read_text())["ViT-FullFT-SupCon"]["checkpoint"])==fresh_full_supcon.resolve()
    bad_output = root / "bad_inventory"
    bad = subprocess.run([sys.executable, str(EVALUATOR), "--closed-manifest", str(old26_manifest), "--open-manifest", str(open_manifest), "--full-manifest", str(full_manifest), "--checkpoint-root", str(checkpoints), "--output-dir", str(bad_output), "--inventory-only"], cwd=REPO, env=env, text=True, capture_output=True)
    assert bad.returncode != 0
    audit = json.loads((bad_output / "manifest_audit.json").read_text())
    assert "closed label count is not 27" in audit["errors"]
    wrong_open=opened[:-1]+["wrong label"]; wrong_open_manifest=manifests/"wrong_open.csv"; wrong_full_manifest=manifests/"wrong_full.csv"
    write_manifest(wrong_open_manifest,wrong_open); write_manifest(wrong_full_manifest,closed+wrong_open)
    wrong_output=root/"wrong_open_inventory"; wrong=subprocess.run([sys.executable,str(EVALUATOR),"--closed-manifest",str(closed_manifest),"--open-manifest",str(wrong_open_manifest),"--full-manifest",str(wrong_full_manifest),"--checkpoint-root",str(checkpoints),"--output-dir",str(wrong_output),"--inventory-only"],cwd=REPO,env=env,text=True,capture_output=True)
    assert wrong.returncode!=0
    assert "open labels do not match the required Open15 English names" in json.loads((wrong_output/"manifest_audit.json").read_text())["errors"]
    invalid_auto=root/"invalid_auto"; invalid_auto.mkdir(); write_checkpoint(invalid_auto/"bad"/"best.pt",closed,old26_manifest)
    invalid_output=root/"invalid_auto_inventory"; invalid=subprocess.run([sys.executable,str(EVALUATOR),"--closed-manifest",str(closed_manifest),"--open-manifest",str(open_manifest),"--full-manifest",str(full_manifest),"--checkpoint-root",str(invalid_auto),"--output-dir",str(invalid_output),"--inventory-only"],cwd=REPO,env=env,text=True,capture_output=True)
    assert invalid.returncode==0; invalid_errors=json.loads((invalid_output/"checkpoint_selection_errors.json").read_text()); assert invalid_errors["ViT-ClosedFT"]!=ev.PASS and set(json.loads((invalid_output/"checkpoint_selection.json").read_text()))=={"ViT Base"}

    originals = ev.make_model, ev.extract, ev.map_gt_label_folders, ev.save_tsne, ev.save_prototype_tsne, ev.save_curve
    gt_root = root / "gt"; gt_zip = root / "unused.zip"
    with zipfile.ZipFile(gt_zip,"w") as archive: archive.writestr("seed/readme.txt","synthetic")
    def fake_map(_root, labels, report_csv=None):
        folder=Path(_root)/"fake_gt"; folder.mkdir(exist_ok=True); mapped={}
        for label in labels:
            image=folder/f"{ev.normalize_label_name(label)}.jpg"; image.write_bytes(f"gt:{label}".encode()); mapped[label]=[image.resolve()]
        rows=[{"label":label,"status":"matched"} for label in labels]
        if report_csv: pd.DataFrame(rows).to_csv(report_csv,index=False)
        return mapped,rows
    def fake_extract(_model, _forward, manifest, _transform, _batch, _workers, _device):
        frame = pd.read_csv(manifest); values = frame.label.astype(str).tolist()
        features = np.asarray([[sum(map(ord, label)) % 997, (sum(map(ord, label)) ** 2) % 991] for label in values], dtype=np.float32)
        if _transform=="native": features=np.zeros_like(features)
        cache={"features":features,"labels":np.asarray(values,dtype=object),"splits":frame.split.astype(str).to_numpy(object),"paths":frame.path.astype(str).to_numpy(object)}
        return {"backbone":cache,"deployed":cache}
    ev.map_gt_label_folders = fake_map
    ev.extract = fake_extract
    tsne_calls=[]
    def fake_tsne(name,caches,out,seed,limit,representation="common_backbone"):
        tsne_calls.append((name,representation))
        out.mkdir(exist_ok=True); stem=name.replace(" ","_")+"_"+representation; paths={}
        for key,suffix in [("csv",".csv"),("npz",".npz"),("sampling",".sampling.json"),("train_test_png",".png"),("valid_png","_valid.png")]:
            path=out/f"{stem}{suffix}"; path.write_bytes(b"artifact"); paths[key]=str(path)
        total=sum(len(c["labels"]) for c in caches.values()); return {"representation":representation,"total_features":total,"sampled":total if limit<=0 else min(total,limit),"max_samples":limit,**paths}
    ev.save_tsne = fake_tsne
    def fake_prototype_tsne(name,caches,gtc,out,seed,count,representation="common_backbone"):
        out.mkdir(exist_ok=True); stem=name.replace(" ","_")+"_"+representation
        png=out/f"{stem}.png"; csv=out/f"{stem}.csv"; metadata=out/f"{stem}.json"
        png.write_bytes(b"png"); csv.write_text("label\nsynthetic\n"); metadata.write_text("{}")
        return {"status":"ok","representation":representation,"png":str(png),"csv":str(csv),"metadata":str(metadata)}
    ev.save_prototype_tsne = fake_prototype_tsne
    def fake_curve(name,item,out):
        if not item.get("checkpoint"): return {"model":name,"history":"","status":"not_applicable","png":"","csv":"","epochs":0}
        out.mkdir(exist_ok=True); stem=name.replace(" ","_"); csv_path=out/f"{stem}.csv"; png_path=out/f"{stem}.png"; csv_path.write_text("epoch\n1\n"); png_path.write_bytes(b"png")
        return {"model":name,"history":"","status":"ok","png":str(png_path),"csv":str(csv_path),"epochs":10,"best_epoch":10}
    ev.save_curve = fake_curve
    base_args = [
        "evaluator_v5.py", "--closed-manifest", str(closed_manifest), "--open-manifest", str(open_manifest), "--full-manifest", str(full_manifest),
        "--checkpoint-root", str(checkpoints), "--closed-checkpoint", str(fresh_closed), "--closed-supcon-checkpoint", str(fresh_closed_supcon),
        "--full-checkpoint", str(fresh_full), "--full-supcon-checkpoint", str(fresh_full_supcon), "--gt-zip", str(gt_zip), "--gt-root", str(gt_root), "--allow-cpu",
    ]
    old_argv = sys.argv
    try:
        ev.make_model = lambda item, device: (_ for _ in ()).throw(RuntimeError("existing checkpoint invalid at runtime")) if item.get("checkpoint")==str(fresh_full_supcon.resolve()) else ((object(),lambda x:x,"native","common",{"source":"fake"}) if item["kind"]=="pretrained_base" else (object(),lambda x:x,"common","common",{"source":"fake"}))
        failed_output = root / "failed_full"; sys.argv = base_args + ["--output-dir", str(failed_output)]
        try: ev.main()
        except RuntimeError as error: assert "96/120" in str(error)
        failed_summary = pd.read_csv(failed_output / "summary.csv")
        assert len(failed_summary) == 120 and (failed_summary.status == "ok").sum() == 96
        assert (failed_summary.status == ev.PASS).sum() == 0 and (failed_summary.status == "unavailable").sum() == 24
        failed_progress = json.loads((failed_output / "progress.json").read_text())
        assert failed_progress["tsne_max_samples"] == 0
        assert failed_progress["models"]["ViT-FullFT-SupCon"]["status"] == "unavailable"
        assert all(failed_progress["models"][name]["status"] == "ok" for name in ev.MODELS if name!="ViT-FullFT-SupCon")
        ev.make_model = lambda item, device: (object(),lambda x:x,"native","common",{"source":"fake"}) if item["kind"]=="pretrained_base" else (object(),lambda x:x,"common","common",{"source":"fake"})
        complete_output = root / "complete_full"; sys.argv = base_args + ["--output-dir", str(complete_output)]
        ev.main()
        complete_summary = pd.read_csv(complete_output / "summary.csv")
        assert len(complete_summary) == 120 and (complete_summary.status == "ok").sum() == 120
        backbone_summary=pd.read_csv(complete_output/"backbone_summary.csv"); assert len(backbone_summary)==120
        assert complete_summary[complete_summary.model=="ViT Base"].accuracy.mean()<backbone_summary[backbone_summary.model=="ViT Base"].accuracy.mean()
        complete_progress=json.loads((complete_output/"progress.json").read_text()); assert all(item["tsne"]["sampled"]==item["tsne"]["total_features"] for item in complete_progress["models"].values())
        assert all((name,"common_backbone") in tsne_calls for name in ev.MODELS)
        assert ("ViT Base","deployed") in tsne_calls and ("ViT-ClosedFT-SupCon","deployed") in tsne_calls and ("ViT-FullFT-SupCon","deployed") in tsne_calls
        fingerprint=json.loads((complete_output/"run_fingerprint.json").read_text())
        assert fingerprint["config"]["tsne_max_samples"]==0
        assert fingerprint["schema"]==2 and fingerprint["sources"]["src/models.py"]["sha256"]==ev.digest(REPO/"src/models.py")
        assert {"python","torch","torchvision","timm","sklearn","numpy","pandas","matplotlib","pillow"}<=set(fingerprint["runtime"])
        selected_results=pd.read_csv(complete_output/"selected_results.csv"); assert len(selected_results)==10
        completion=json.loads((complete_output/"completion.json").read_text()); assert completion["selected_rows"]==10 and completion["artifacts"]["REPORT.md"]==ev.digest(complete_output/"REPORT.md")
        report = (complete_output / "REPORT.md").read_text(encoding="utf-8")
        assert "t-SNE: all features" in report and "## Training curves" in report and "## Artifact index" in report and "## SHA-256 provenance" in report
        experiment_codes=("1a","1b","2a","2b","4a","4b","5a","5b"); assert all(f"## Experiment {code}" in report for code in experiment_codes) and "Experiment 3" not in report
        for code in experiment_codes:
            section=report.split(f"## Experiment {code}",1)[1].split("\n## ",1)[0]; assert "accuracy / macro-F1 / weighted-F1" in section and all(f"| {partition} | {prototype} |" in section for partition in ("Closed","Open") for prototype in ("GT","Mean","SS"))
        assert "## Five research-question conclusions" in report
        assert report.split("## Five research-question conclusions",1)[1].count("\u0394accuracy")>=3
        assert "## Held-out test performance by partition" in report and "### Closed" in report and "### Open" in report
        closed_section = report.split("### Closed", 1)[1].split("### Open", 1)[0]; open_section = report.split("### Open", 1)[1].split("## Matched held-out test comparisons", 1)[0]
        assert all(name in closed_section and name in open_section for name in ev.MODELS)
        assert all(f"- {scope} best {setting}:" in report for scope in ("Overall", "Closed", "Open") for setting in ("distance", "prototype"))
        assert "Covered common-backbone rows: 120/120 (OK 120, PASS 0)" in report
        selected_only=root/"selected_only"; selected_only.mkdir(); missing_args=list(base_args); missing_args[missing_args.index("--checkpoint-root")+1]=str(selected_only); flag=missing_args.index("--full-supcon-checkpoint"); del missing_args[flag:flag+2]
        missing_output=root/"missing_full_supcon"; sys.argv=missing_args+["--output-dir",str(missing_output)]; ev.main()
        missing_summary=pd.read_csv(missing_output/"summary.csv"); assert len(missing_summary)==120 and (missing_summary.status=="ok").sum()==96 and (missing_summary.status==ev.PASS).sum()==24
        assert set(missing_summary[missing_summary.status==ev.PASS].model)=={"ViT-FullFT-SupCon"}
        missing_progress=json.loads((missing_output/"progress.json").read_text()); assert missing_progress["models"]["ViT-FullFT-SupCon"]["status"]==ev.PASS and missing_progress["models"]["ViT-FullFT-SupCon"]["tsne"]["status"]==ev.PASS
        assert len(pd.read_csv(missing_output/"selected_results.csv"))==8 and json.loads((missing_output/"completion.json").read_text())["selected_rows"]==8
        missing_report=(missing_output/"REPORT.md").read_text(encoding="utf-8"); assert ev.PASS in missing_report.split("## Experiment 5a",1)[1].split("## Experiment 5b",1)[0] and ev.PASS in missing_report.split("## Experiment 5b",1)[1].split("## Five research-question conclusions",1)[0]
        ev.save_tsne=lambda *args,**kwargs: (_ for _ in ()).throw(RuntimeError("visual failure"))
        visual_output=root/"visual_failure"; sys.argv=base_args+["--output-dir",str(visual_output)]
        try: ev.main()
        except RuntimeError as error: assert "t-SNE complete=False" in str(error)
        else: raise AssertionError("visual failure was not reported")
        assert (pd.read_csv(visual_output/"summary.csv").status=="ok").sum()==120
        assert all(item["tsne"]["status"]=="unavailable" for item in json.loads((visual_output/"progress.json").read_text())["models"].values())
        assert not (visual_output/"completion.json").exists()
        assert "## t-SNE coverage\n- ViT Base: unavailable" in (visual_output/"REPORT.md").read_text(encoding="utf-8")
        ev.save_tsne=fake_tsne; sys.argv=base_args+["--output-dir",str(visual_output),"--resume-tsne"]; ev.main()
        assert all(item["tsne"]["status"]=="ok" for item in json.loads((visual_output/"progress.json").read_text())["models"].values())
        assert (visual_output/"completion.json").is_file()
        assert "## t-SNE coverage\n- ViT Base: ok" in (visual_output/"REPORT.md").read_text(encoding="utf-8")
    finally:
        sys.argv = old_argv
        ev.make_model, ev.extract, ev.map_gt_label_folders, ev.save_tsne, ev.save_prototype_tsne, ev.save_curve = originals


def test_metric_diagnosis():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        check_normalization_and_gt(root / "gt")
        check_distances_and_sampling()
        check_checkpoint_provenance(root / "provenance")
        check_visual_artifacts(root / "visuals")
        check_inventory(root / "inventory_case")


def main():
    test_metric_diagnosis()
    print("EVALUATOR_V5_TESTS_OK")


if __name__ == "__main__":
    main()
