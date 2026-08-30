import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
EVALUATOR = REPO / "pipelines" / "eval" / "evaluate_metric_diagnosis.py"
spec = importlib.util.spec_from_file_location("metric_diagnosis_full_only", EVALUATOR)
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


def test_full_only_needs_no_checkpoint_or_partition_manifests(tmp_path, monkeypatch):
    labels = [f"label-{index:02d}" for index in range(42)]
    rows = [{"path": str(tmp_path / f"{label}-{split}.png"), "label": label, "split": split} for label in labels for split in ["train", "test"]]
    manifest = tmp_path / "full42.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    captured = {}
    monkeypatch.setattr(evaluator, "make_model", lambda item, device: (object(), object(), object(), object(), {"deployed": {"source": "test"}}))
    def extract(*args):
        selected = pd.read_csv(args[2]); captured["frame"] = selected
        features = np.stack([np.eye(42, dtype=np.float32)[labels.index(label)] for label in selected.label])
        return {"backbone": {"features": features, "labels": selected.label.to_numpy(object), "splits": selected.split.to_numpy(object), "paths": selected.path.to_numpy(object)}}
    monkeypatch.setattr(evaluator, "extract", extract)
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["evaluate_metric_diagnosis.py", "--full-only", "--full-manifest", str(manifest), "--output-dir", str(output), "--allow-cpu", "--num-workers", "0", "--ss-per-label", "10"])

    evaluator.main()

    summary = pd.read_csv(output / "summary.csv")
    predictions = pd.read_csv(output / "predictions.csv")
    assert set(summary.distance) == {"euclidean", "cosine"}
    assert len(summary) == 2 and (summary.accuracy == 1.0).all()
    assert len(predictions) == 84 and predictions.correct.all()
