from __future__ import annotations

import ast
import inspect
import itertools
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from implantsj.contracts import CLOSED_LABELS, validate_partition  # noqa: E402
from implantsj.evaluation import (  # noqa: E402
    EmbeddingTable,
    build_prototypes,
    classification_metrics,
    predict_42way,
    prototype_geometry,
    require_coverage,
)
from implantsj.losses import (  # noqa: E402
    CrossEntropyStrategy,
    SupConLoss,
    SupConStrategy,
    TripletStrategy,
)
from implantsj.manifests import build_manifests, build_parser  # noqa: E402
from implantsj.models import (  # noqa: E402
    ModelAdapter,
    ModelSpec,
    create_adapter,
    register_adapter,
)
from implantsj.training import validate_supcon_pk  # noqa: E402
from implantsj.models import validate_canonical_checkpoint  # noqa: E402
from implantsj.reporting import (  # noqa: E402
    REQUIRED_AXES,
    build_experiment_sections,
    collect_metrics,
    main as reporting_main,
    select_operating_point,
    validate_canonical,
    write_report,
)
from implantsj.training import Experiment  # noqa: E402


OPEN_LABELS = [f"Open {index:02d}" for index in range(1, 16)]
ALL_LABELS = list(CLOSED_LABELS) + OPEN_LABELS


def table(features: np.ndarray, labels: list[str], prefix: str) -> EmbeddingTable:
    count = len(labels)
    return EmbeddingTable(
        features=np.asarray(features, dtype=np.float32),
        labels=np.asarray(labels, dtype=str),
        paths=np.asarray([f"/{prefix}/{index}.png" for index in range(count)]),
        splits=np.full(count, prefix),
        sets=np.asarray(
            ["closed" if label in CLOSED_LABELS else "open" for label in labels]
        ),
        metadata={},
    )


def canonical_source(root: Path) -> Path:
    rows = []
    for label in ALL_LABELS:
        for index in range(3):
            rows.append(
                {
                    "path": f"images/{label}/train_{index}.png",
                    "label": label,
                    "split": "valid" if index == 0 else "train",
                }
            )
        rows.append(
            {
                "path": f"images/{label}/test.png",
                "label": label,
                "split": "test",
            }
        )
    source = root / "source.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    return source


def test_closed_labels_exact_27():
    assert len(CLOSED_LABELS) == 27
    assert (PROJECT / "configs" / "closed_labels.txt").read_text().splitlines() == list(
        CLOSED_LABELS
    )


def test_open_is_exact_complement_15_and_original_test_preserved(tmp_path):
    source = canonical_source(tmp_path)
    output = tmp_path / "out"
    args = build_parser().parse_args(
        [
            "--source-manifest",
            str(source),
            "--output-dir",
            str(output),
            "--valid-fraction",
            "0.34",
        ]
    )
    build_manifests(args)
    labels = json.loads((output / "labels.json").read_text())
    source_frame = pd.read_csv(source)
    test = pd.read_csv(output / "test.csv")
    valid = pd.read_csv(output / "valid.csv")
    assert labels["closed"] == list(CLOSED_LABELS)
    assert set(labels["open"]) == set(OPEN_LABELS)
    assert len(labels["open"]) == 15
    assert set(test.path) == {
        str((tmp_path / value).resolve())
        for value in source_frame.loc[source_frame.split == "test", "path"]
    }
    assert set(valid.origin_split) == {"train"}


def test_gt_requires_multiple_independent_images(tmp_path):
    source = canonical_source(tmp_path)
    gt = tmp_path / "gt.csv"
    rows = [{"path": f"gt/{label}.png", "label": label} for label in ALL_LABELS]
    pd.DataFrame(rows).to_csv(gt, index=False)
    args = build_parser().parse_args(
        [
            "--source-manifest",
            str(source),
            "--gt-manifest",
            str(gt),
            "--output-dir",
            str(tmp_path / "one-image"),
        ]
    )
    with pytest.raises(ValueError, match="at least two independent images"):
        build_manifests(args)

    rows += [{"path": f"gt/{label}_2.png", "label": label} for label in ALL_LABELS]
    rows[0]["path"] = f"images/{ALL_LABELS[0]}/train_1.png"
    pd.DataFrame(rows).to_csv(gt, index=False)
    args.output_dir = str(tmp_path / "overlap")
    with pytest.raises(ValueError, match="independent from train/valid/test"):
        build_manifests(args)


def test_duplicate_paths_rejected(tmp_path):
    source = tmp_path / "source.csv"
    pd.DataFrame(
        [
            {"path": "same.png", "label": "A", "split": "train"},
            {"path": "same.png", "label": "A", "split": "test"},
        ]
    ).to_csv(source, index=False)
    args = build_parser().parse_args(
        [
            "--source-manifest",
            str(source),
            "--output-dir",
            str(tmp_path / "out"),
            "--allow-incomplete",
        ]
    )
    with pytest.raises(ValueError, match="duplicate image paths"):
        build_manifests(args)


def test_group_disjoint_split_and_audit(tmp_path):
    labels = ["Closed A", "Open B"]
    rows = [
        {
            "path": f"{group}_{label}.png",
            "label": label,
            "split": "train",
            "case_id": group,
        }
        for group in ("g1", "g2", "g3", "g4")
        for label in labels
    ] + [
        {
            "path": f"test_{label}.png",
            "label": label,
            "split": "test",
            "case_id": f"test-{label}",
        }
        for label in labels
    ]
    source = tmp_path / "source.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    closed = tmp_path / "closed.txt"
    closed.write_text("Closed A\n")
    output = tmp_path / "out"
    args = build_parser().parse_args(
        [
            "--source-manifest",
            str(source),
            "--closed-labels",
            str(closed),
            "--output-dir",
            str(output),
            "--valid-fraction",
            "0.5",
        ]
    )
    build_manifests(args)
    train = pd.read_csv(output / "train.csv")
    valid = pd.read_csv(output / "valid.csv")
    audit = json.loads((output / "audit.json").read_text())
    assert set(train.case_id).isdisjoint(valid.case_id)
    assert audit["group_column_used"] == "case_id"


def test_partition_validation():
    validate_partition(ALL_LABELS, CLOSED_LABELS, OPEN_LABELS, strict=True)
    with pytest.raises(ValueError, match="invalid label partition"):
        validate_partition(ALL_LABELS, CLOSED_LABELS, OPEN_LABELS[:-1], strict=True)


def test_prototypes_cover_42_missing_fails_and_ss_is_deterministic():
    labels = [label for label in ALL_LABELS for _ in range(3)]
    features = np.repeat(np.eye(42, dtype=np.float32), 3, axis=0)
    source = table(features, labels, "support")
    for method in ("gt", "mean", "ss"):
        prototypes, members = build_prototypes(
            source, ALL_LABELS, method, seed=7, ss_count=2
        )
        assert prototypes.shape == (42, 42)
        assert set(members.label) == set(ALL_LABELS)
    first = build_prototypes(source, ALL_LABELS, "ss", seed=7, ss_count=2)[1]
    second = build_prototypes(source, ALL_LABELS, "ss", seed=7, ss_count=2)[1]
    assert first.path.tolist() == second.path.tolist()
    incomplete = table(features[:-3], labels[:-3], "support")
    with pytest.raises(ValueError, match="coverage"):
        require_coverage(incomplete, ALL_LABELS, "support")


def test_euclidean_cosine_disagree_and_groups_share_predictions():
    prototypes = np.asarray([[100.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    euclidean = predict_42way(query, prototypes, ["direction", "near"], "euclidean")[0]
    cosine = predict_42way(query, prototypes, ["direction", "near"], "cosine")[0]
    assert euclidean.tolist() == ["near"]
    assert cosine.tolist() == ["direction"]
    truth = np.asarray(ALL_LABELS)
    predictions = truth.copy()
    predictions[0] = OPEN_LABELS[0]
    assert classification_metrics(truth, predictions, CLOSED_LABELS)["n_samples"] == 27
    assert classification_metrics(truth, predictions, OPEN_LABELS)["accuracy"] == 1.0


def test_geometry_has_norm_and_collapse_fields():
    source = table(np.eye(3, dtype=np.float32), ["a", "b", "c"], "support")
    geometry = prototype_geometry(source, source.features, ["a", "b", "c"])
    assert {
        "embedding_norm_mean",
        "embedding_norm_std",
        "mean_dimension_std",
        "near_zero_dimension_count",
    } <= set(geometry)


def test_pk_supcon_measures_cross_image_positive_fraction():
    torch = pytest.importorskip("torch")

    class Identity(torch.nn.Module):
        def forward(self, values, return_projection=True):
            return torch.nn.functional.normalize(values, dim=-1)

    views = (torch.eye(4), torch.eye(4))
    targets = torch.tensor([0, 0, 1, 1])
    loader = [(views, targets, ["a", "a", "b", "b"], ["p1", "p2", "p3", "p4"])]
    metrics = SupConStrategy().run_epoch(Identity(), loader, "cpu")
    assert metrics.cross_image_positive_fraction == pytest.approx(2 / 3)
    assert metrics.cross_image_positive_count == 16
    assert metrics.positive_count == 24


def test_supcon_loss_is_projection_only():
    torch = pytest.importorskip("torch")
    features = torch.nn.functional.normalize(
        torch.eye(2).repeat(1, 2).view(2, 2, 2), dim=-1
    )
    criterion = SupConLoss()
    loss = criterion(features, torch.tensor([0, 1]))
    assert loss.ndim == 0
    assert not hasattr(criterion, "classifier")


@pytest.mark.parametrize(
    ("labels", "batch_size", "samples_per_class", "message"),
    [
        (["a", "b"], 3, 2, "divisible"),
        (["a", "a", "b"], 4, 2, "at least K"),
        (["a", "a"], 6, 2, "requires P"),
    ],
)
def test_supcon_pk_preflight_rejects_impossible_batches(
    labels, batch_size, samples_per_class, message
):
    with pytest.raises(ValueError, match=message):
        validate_supcon_pk(labels, batch_size, samples_per_class)


@pytest.mark.parametrize(
    ("strategy_type", "loader", "message"),
    [
        (CrossEntropyStrategy, "all_unknown", "no known-label examples"),
        (SupConStrategy, "empty", "no examples"),
        (TripletStrategy, "empty", "no examples"),
    ],
)
def test_loss_strategies_reject_epochs_without_processed_examples(
    strategy_type, loader, message
):
    torch = pytest.importorskip("torch")
    strategy = strategy_type()

    class Model:
        def train(self, _training):
            return self

    batches = []
    if loader == "all_unknown":
        batches = [
            (
                torch.zeros((2, 3), dtype=torch.float32),
                torch.full((2,), -1, dtype=torch.long),
                ["open", "open"],
                ["a.png", "b.png"],
            )
        ]
    with pytest.raises(ValueError, match=message):
        strategy.run_epoch(Model(), batches, "cpu")


def test_canonical_checkpoint_provenance_requires_last_and_matches_contract():
    provenance = {
        "model_role": "ClosedFT-SupCon",
        "architecture": "vit_base_patch16_224",
        "base_checkpoint_sha256": "a" * 64,
        "scope": "closed",
        "objective": "pure_supcon",
        "ordered_labels_sha256": "b" * 64,
        "projection": {"output_dim": 128, "hidden_dim": 512},
        "seed": 42,
        "epochs": 15,
        "epoch": 15,
        "final_epoch": 15,
        "checkpoint_selector": "last.pt",
        "protocol_fingerprint": "c" * 64,
        "assignment_sha256": "d" * 64,
        "lineage": {},
    }
    validate_canonical_checkpoint(
        provenance,
        {
            "model_role": "ClosedFT-SupCon",
            "objective": "pure_supcon",
            "epoch": 15,
            "checkpoint_selector": "last.pt",
        },
    )
    provenance["checkpoint_selector"] = "best.pt"
    with pytest.raises(ValueError, match="last.pt"):
        validate_canonical_checkpoint(provenance, {"epoch": 15})


def test_fake_adapter_registry():
    class FakeAdapter(ModelAdapter):
        def load(self):
            self.model = object()
            return self

        def encode_batch(self, images):
            return images

    register_adapter("test_fake", FakeAdapter, replace=True)
    adapter = create_adapter(ModelSpec(name="fake", adapter="test_fake"))
    assert isinstance(adapter, FakeAdapter)


def synthetic_metrics(models: list[str]) -> pd.DataFrame:
    rows = []
    for index, values in enumerate(
        itertools.product(
            models,
            REQUIRED_AXES["query"],
            REQUIRED_AXES["prototype"],
            REQUIRED_AXES["distance"],
            REQUIRED_AXES["group"],
        )
    ):
        model, query, prototype, distance, group = values
        rows.append(
            {
                "model": model,
                "query": query,
                "prototype": prototype,
                "distance": distance,
                "group": group,
                "gallery_classes": 42,
                "accuracy": 0.5 + index / 10000,
                "macro_f1": 0.4 + index / 10000,
            }
        )
    return pd.DataFrame(rows)


def test_reporting_180_rows_and_validation_only_selection(tmp_path):
    models = [f"model-{index}" for index in range(5)]
    frame = synthetic_metrics(models)
    validate_canonical(frame, models)
    selected = select_operating_point(frame)
    changed = frame.copy()
    changed.loc[changed["query"] == "test", ["accuracy", "macro_f1"]] = -999
    assert select_operating_point(changed) == selected
    roles = {
        "base": models[0],
        "closed_ft": models[1],
        "closed_ft_supcon": models[2],
        "full_ft": models[3],
        "full_ft_supcon": models[4],
    }
    sections = build_experiment_sections(frame, roles)
    assert set(sections) == {"1a", "1b", "2a", "2b", "4a", "4b", "5a", "5b"}
    report = write_report(tmp_path, frame, sections, [])
    assert report.is_file()
    assert (tmp_path / "canonical_results.csv").is_file()
    assert (tmp_path / "comparison_tables.csv").is_file()
    eval_root = tmp_path / "eval-root"
    for (model, query), rows in frame.groupby(["model", "query"]):
        output = eval_root / model / "eval" / query
        output.mkdir(parents=True)
        rows.to_csv(output / "metrics.csv", index=False)
    validate_canonical(collect_metrics(eval_root, models), models)
    cli_output = tmp_path / "cli-report"
    assert (
        reporting_main(
            [
                "--eval-root",
                str(eval_root),
                "--models",
                ",".join(models),
                "--roles",
                ",".join(f"{role}={model}" for role, model in roles.items()),
                "--output-dir",
                str(cli_output),
                "--expected-rows",
                "180",
            ]
        )
        == 0
    )
    with pytest.raises(ValueError, match="missing"):
        validate_canonical(frame.iloc[:-1], models)


def test_training_selection_assignment_does_not_reference_test():
    tree = ast.parse(textwrap.dedent(inspect.getsource(Experiment.run)))
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "selection_score"
            for target in node.targets
        )
    ]
    assert assignments
    for assignment in assignments:
        names = {
            node.id for node in ast.walk(assignment.value) if isinstance(node, ast.Name)
        }
        assert not any(name.startswith("test") for name in names)


def test_bash_syntax_dry_run_and_name_contract():
    subprocess.run(
        ["bash", "-n", str(PROJECT / "run.sh"), str(PROJECT / "config.sh")],
        check=True,
    )
    completed = subprocess.run(
        ["bash", str(PROJECT / "run.sh"), "all"],
        cwd=PROJECT,
        env={**os.environ, "DRY_RUN": "1"},
        text=True,
        capture_output=True,
        check=True,
    )
    for stage in ("manifests", "training", "embed", "evaluate", "tsne", "reporting"):
        assert stage in completed.stdout
    forbidden_name = "metric" + "diag"
    assert (
        forbidden_name
        not in "\n".join(
            path.read_text(errors="ignore")
            for path in PROJECT.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        ).lower()
    )
