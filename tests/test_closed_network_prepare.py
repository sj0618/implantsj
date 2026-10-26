#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def write_sample_pair(image_root: Path, json_root: Path, hospital: str, case_id: str, stem: str, label: str) -> None:
    image_dir = image_root / hospital / case_id
    json_dir = json_root / hospital / case_id
    image_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    img_path = image_dir / f"{stem}.jpg"
    json_path = json_dir / f"{stem}.json"
    Image.new("RGB", (12, 12), (100, 120, 140)).save(img_path)
    json_path.write_text(json.dumps({"모델명": label}, ensure_ascii=False), encoding="utf-8")


def test_closed_network_prepare() -> None:
    with tempfile.TemporaryDirectory(prefix="closed_network_prepare_") as tmp:
        tmp_dir = Path(tmp)
        tl1 = tmp_dir / "TL1"
        ts1 = tmp_dir / "TS1"
        vl1 = tmp_dir / "VL1"
        vs1 = tmp_dir / "VS1"
        out_root = tmp_dir / "prepared"

        # Train-side labels: SeenA(3), SeenB(2), NovelC(1)
        write_sample_pair(ts1, tl1, "h01", "h01_0001", "img_001", "SeenA")
        write_sample_pair(ts1, tl1, "h01", "h01_0002", "img_002", "SeenA")
        write_sample_pair(ts1, tl1, "h02", "h02_0001", "img_003", "SeenA")
        write_sample_pair(ts1, tl1, "h02", "h02_0002", "img_004", "SeenB")
        write_sample_pair(ts1, tl1, "h03", "h03_0001", "img_005", "SeenB")
        write_sample_pair(ts1, tl1, "h03", "h03_0002", "img_006", "NovelC")

        # Test-side
        write_sample_pair(vs1, vl1, "h04", "h04_0001", "img_101", "SeenA")
        write_sample_pair(vs1, vl1, "h04", "h04_0002", "img_102", "SeenB")
        write_sample_pair(vs1, vl1, "h05", "h05_0001", "img_103", "NovelC")

        cmd = [
            sys.executable,
            str(ROOT / "pipelines" / "data" / "prepare_closed_network_dataset.py"),
            "--train-json-root",
            str(tl1),
            "--train-image-root",
            str(ts1),
            "--test-json-root",
            str(vl1),
            "--test-image-root",
            str(vs1),
            "--output-root",
            str(out_root),
            "--valid-ratio",
            "0.34",
            "--seen-label-count",
            "2",
            "--unseen-label-count",
            "1",
            "--materialize-imagefolders",
        ]
        subprocess.run(cmd, check=True)

        manifest_all = pd.read_csv(out_root / "manifests" / "closed_network_3label.csv")
        manifest_seen = pd.read_csv(out_root / "manifests" / "closed_network_seen2.csv")
        manifest_unseen = pd.read_csv(out_root / "manifests" / "closed_network_unseen1.csv")
        summary = json.loads((out_root / "summary.json").read_text(encoding="utf-8"))

        assert sorted(manifest_all["label"].unique().tolist()) == ["NovelC", "SeenA", "SeenB"]
        assert set(manifest_all["split"].unique().tolist()) == {"train", "valid", "test"}
        assert sorted(manifest_seen["label"].unique().tolist()) == ["SeenA", "SeenB"]
        assert sorted(manifest_unseen["label"].unique().tolist()) == ["NovelC"]
        assert set(manifest_unseen["split"].unique().tolist()) == {"test"}

        # Materialized ImageFolder structures exist.
        assert (out_root / "imagefolders" / "data_3label" / "train").is_dir()
        assert (out_root / "imagefolders" / "data_metric_seen2" / "val").is_dir()
        assert (out_root / "imagefolders" / "data_metric_unseen1" / "test").is_dir()

        # The smallest train-count label should have been selected as unseen.
        unseen_labels = (out_root / "label_splits" / "unseen1_labels.txt").read_text(encoding="utf-8").strip().splitlines()
        assert unseen_labels == ["NovelC"]
        assert summary["seen_label_count"] == 2
        assert summary["unseen_label_count"] == 1

    print("OK: closed-network dataset preparation test passed")


if __name__ == "__main__":
    test_closed_network_prepare()
