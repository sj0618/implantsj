"""임베딩 저장 hook + attributes separate/shared 스모크 테스트 (합성 데이터, CPU, 수십 초).

    python tests/test_smoke.py        # 또는 pytest tests/test_smoke.py
Docker 빌드의 마지막 자가검증 레이어에서도 돈다. 실제 가중치를 받지 않는다(pretrained=False).
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CLASSES = ["classA", "classB"]
SPLITS = {"train": 3, "valid": 1, "test": 1}


def make_dataset(root: Path):
    rng = np.random.default_rng(0)
    rows = []
    for label in CLASSES:
        for split, n in SPLITS.items():
            for i in range(n):
                path = root / "images" / label / f"{split}_{i}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rng.integers(0, 255, (20, 28, 3), dtype=np.uint8)).save(path)
                rows.append({"path": str(path), "label": label, "split": split})
    full = pd.DataFrame(rows)
    full.to_csv(root / "manifest_full.csv", index=False)
    full[full["label"] == "classA"].to_csv(root / "manifest_closed.csv", index=False)
    return root / "manifest_full.csv", full


def make_spec(root: Path) -> Path:
    columns = ["dataset_label", "Company", "Name", "System", "Connection(1)", "Conection(2)",
               "Flange", "Collar", "Microthread", "Body Shape", "Thread Shape", "Apex Shape", "Apex Hole"]
    rows = [
        ["classA", "Osstem", "A", "Bone Level", "Internal", "HEX", "Parallel", "Yes", "Yes", "Tapered", "V", "Flat", "No"],
        ["classB", "Osstem", "B", "Tissue Level", "Internal", "HEX", "Tapered", "Yes", "Yes", "Tapered", "V", "Flat", "No"],
    ]
    path = root / "spec.csv"
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def test_embeddings_dump(tmp_path):
    import timm

    from embeddings import EmbedPlan, dump_embeddings, square_pad_transform, timm_classifier_forward

    full, frame = make_dataset(tmp_path / "emb")
    model = timm.create_model("resnet18", pretrained=False, num_classes=2)
    plan = EmbedPlan(enabled=True, manifests=[str(full)], closed_labels={"classA"},
                     batch_size=4, num_workers=0)
    out = tmp_path / "emb" / "embeddings"
    files = dump_embeddings(
        output_dir=out, plan=plan, forward=timm_classifier_forward(model),
        transform=square_pad_transform(32), stage="test", model_name="resnet18",
        checkpoint=None, device="cpu", head_kind="logits", extra={"k": 1},
    )
    assert len(files) == 6, files                       # 3 split × (backbone, head)
    backbone = np.load(out / "manifest_full_train_backbone.npz")
    head = np.load(out / "manifest_full_train_head.npz")
    assert backbone["features"].shape == (6, 512) and backbone["features"].dtype == np.float32
    assert head["features"].shape == (6, 2)
    train_paths = frame[frame["split"] == "train"]["path"].tolist()
    assert list(backbone["paths"]) == train_paths        # manifest 순서 그대로
    assert list(backbone["labels"]) == ["classA"] * 3 + ["classB"] * 3
    assert list(backbone["split"]) == ["train"] * 6                # dtype 잘림 회귀 방지
    assert list(backbone["partition"]) == ["closed"] * 3 + ["open"] * 3
    assert str(backbone["source"]) == "backbone" and str(head["head_kind"]) == "logits"
    assert json.loads(str(head["extra"])) == {"k": 1}
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert [row["split"] for row in index["files"]] == ["test", "train", "valid"]

    # 백본 스테이지 전처리와 같은 정사각 패딩인지 (두 구현이 따로 있으므로 어긋나면 여기서 잡는다)
    from backbone.core import pad_to_square as backbone_pad
    from embeddings import pad_to_square as embed_pad
    sample = Image.fromarray(np.arange(20 * 28 * 3, dtype=np.uint8).reshape(20, 28, 3))
    assert np.array_equal(np.asarray(backbone_pad(sample)), np.asarray(embed_pad(sample)))

    # 안전 래퍼: forward 가 죽어도 예외를 올리지 않고 빈 목록을 돌려준다
    from embeddings import try_dump_embeddings

    def broken(images):
        raise RuntimeError("boom")

    assert try_dump_embeddings(output_dir=tmp_path / "broken", plan=plan, forward=broken,
                               transform=square_pad_transform(32), stage="test", model_name="x") == []

    # 꺼져 있으면 아무것도 쓰지 않는다
    plan_off = EmbedPlan(enabled=False, manifests=[str(full)])
    assert dump_embeddings(output_dir=tmp_path / "off", plan=plan_off, forward=timm_classifier_forward(model),
                           transform=square_pad_transform(32), stage="test", model_name="x") == []


def test_plan_from_config(tmp_path):
    from embeddings import plan_from_config

    full, _ = make_dataset(tmp_path / "plan")
    closed = tmp_path / "plan" / "manifest_closed.csv"
    cfg = {"embeddings": {"enabled": True, "batch_size": 8},
           "train": {"num_workers": 2},
           "data": {"closed_manifest": str(closed), "full_manifest": str(full)}}
    plan = plan_from_config(cfg)
    assert plan.manifests == [str(full)] and plan.closed_labels == {"classA"}
    # 꺼져 있으면 manifest 를 열지 않는다 (없는 파일을 줘도 조용히 비활성 계획)
    off = plan_from_config({"embeddings": {"enabled": False, "input_mode": "bogus"},
                            "data": {"closed_manifest": "/nonexistent.csv"}})
    assert off.enabled is False and off.manifests == []
    assert plan.batch_size == 8 and plan.num_workers == 2 and plan.input_mode == "original"
    # full 이 없으면 closed + open 을 각각
    cfg["data"] = {"closed_manifest": str(closed), "open_manifest": str(full)}
    assert plan_from_config(cfg).manifests == [str(closed), str(full)]


def test_attributes_modes(tmp_path):
    from attributes.core import read_manifest
    from attributes.spec_table import build_spec_table
    from attributes.train import train_attributes
    from embeddings import EmbedPlan

    root = tmp_path / "attr"
    full, _ = make_dataset(root)
    spec = make_spec(root)
    frames = {split: read_manifest(full, split=split) for split in SPLITS}
    table = build_spec_table(CLASSES, spec, attributes=["System", "Flange"])
    assert table.attributes == ["System", "Flange"], table.attributes
    plan = EmbedPlan(enabled=True, manifests=[str(full)], closed_labels={"classA"},
                     batch_size=4, num_workers=0)
    entry = {"timm": "resnet18", "checkpoint": None, "image_size": 32}
    common = dict(backbone_entry=entry, table=table, run_features=table.attributes, frames=frames,
                  image_size=32, skip_existing=True, epochs=1, batch_size=2, lr=1e-3,
                  num_workers=0, patience=1, device="cpu", amp=False, pretrained=False,
                  embed_plan=plan)

    for mode in ("separate", "shared"):
        run_dir = root / f"run_{mode}"
        result = train_attributes(mode=mode, run_dir=run_dir, **common)
        assert result["mode"] == mode and result["attributes"] == ["System", "Flange"]
        for split, n in SPLITS.items():
            assert result["probabilities"][split]["System"].shape == (2 * n, 2)
            assert result["probabilities"][split]["Flange"].shape == (2 * n, 2)
        assert (run_dir / "attribute_metrics.json").is_file()
        checkpoints = json.loads((run_dir / "checkpoints.json").read_text(encoding="utf-8"))
        assert checkpoints["mode"] == mode

        if mode == "shared":
            assert (run_dir / "checkpoints" / "shared.pt").is_file()
            assert (run_dir / "history_shared.csv").is_file()
            head = np.load(run_dir / "embeddings" / "manifest_full_test_head.npz")
            assert set(head["heads"]) == {"System", "Flange"}
            assert head["features__System"].shape == (2, 2) and head["features__Flange"].shape == (2, 2)
            assert np.load(run_dir / "embeddings" / "manifest_full_test_backbone.npz")["features"].shape == (2, 512)
            # skip_existing: head 목록이 같으면 재사용 (임베딩도 같은 checkpoint 면 다시 뽑지 않는다)
            again = train_attributes(mode=mode, run_dir=run_dir, **common)
            assert again["checkpoints"]["System"].endswith("shared.pt")
            # head 목록이 달라지면 재사용하지 않고 다시 학습한다
            narrower = train_attributes(mode=mode, run_dir=run_dir, **{**common, "run_features": ["System"]})
            assert narrower["attributes"] == ["System"]
            saved = json.loads((run_dir / "checkpoints.json").read_text(encoding="utf-8"))
            assert list(saved["valid_macro_f1"]) == ["System"]
        else:
            for attribute in ("System", "Flange"):
                assert (run_dir / "checkpoints" / f"{attribute}.pt").is_file()
                assert (run_dir / f"history_{attribute}.csv").is_file()
                head = np.load(run_dir / "embeddings" / attribute / "manifest_full_test_head.npz")
                assert list(head["heads"]) == [attribute]          # 이름 있는 head 1개
                assert head[f"features__{attribute}"].shape == (2, 2)


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        for test in (test_embeddings_dump, test_plan_from_config, test_attributes_modes):
            print(f"  {test.__name__} ...", flush=True)
            test(tmp / test.__name__)
    print("smoke OK: embeddings hook + attributes separate/shared", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
