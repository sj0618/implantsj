from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from src.closed_network_data import split_train_valid
from src.data import make_stratified_splits
from src.experiment_data import PKBatchSampler
from src.feature_attributes import exact_row_match_metrics
from src.metric_learning import resolve_pk_batch_size


ROOT = Path(__file__).resolve().parents[1]


def _epoch_loop_source(path: Path, function_name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    loop = next(node for node in function.body if isinstance(node, ast.For))
    return ast.unparse(loop)


def test_training_epoch_loops_do_not_evaluate_test_split() -> None:
    supervised = _epoch_loop_source(ROOT / "src" / "supervised.py", "train_supervised")
    metric = _epoch_loop_source(ROOT / "src" / "metric_learning.py", "train_metric")
    assert "'valid'" in supervised and "'test'" not in supervised
    assert "split='valid'" in metric and "split='test'" not in metric
    assert "test_accuracy" not in supervised
    assert "test_accuracy" not in metric


def test_active_metric_sampler_has_cross_image_positives() -> None:
    labels = ["A"] * 4 + ["B"] * 4 + ["C"] * 4 + ["D"] * 4
    sampler = PKBatchSampler(labels, batch_size=8, samples_per_class=2, seed=7)
    covered = set()
    for batch in sampler:
        selected = [labels[index] for index in batch]
        covered.update(selected)
        assert len(batch) == len(set(batch))
        assert all(selected.count(label) == 2 for label in set(selected))
    assert covered == set(labels)


def test_metric_batch_size_adapts_to_small_class_sets() -> None:
    assert resolve_pk_batch_size(["A", "A", "B", "B", "C", "C"], 32, 2) == 6
    with pytest.raises(ValueError, match="at least 2"):
        resolve_pk_batch_size(["A", "A"], 2, 1)
    with pytest.raises(ValueError, match="positive multiple"):
        resolve_pk_batch_size(["A", "A"], 7, 2)


def test_split_ratios_reject_invalid_values(tmp_path: Path) -> None:
    source = tmp_path / "manifest.csv"
    pd.DataFrame([{"path": "a.jpg", "label": "A"}]).to_csv(source, index=False)
    with pytest.raises(ValueError, match="finite, non-negative"):
        make_stratified_splits(source, tmp_path / "out.csv", train=0, valid=-1, test=1)


def test_unknown_only_rows_do_not_count_as_exact_matches() -> None:
    rows = [
        {"a_true": "unknown", "a_pred": "x", "b_true": "unknown", "b_pred": "y"},
        {"a_true": "x", "a_pred": "x", "b_true": "unknown", "b_pred": "z"},
    ]
    metrics = exact_row_match_metrics(rows, ["a", "b"])
    assert metrics["exact_row_match"] == 1.0
    assert metrics["exact_row_match_n"] == 1
    assert metrics["exact_row_match_coverage"] == 0.5


def test_group_split_rejects_conflicting_labels_within_a_case() -> None:
    frame = pd.DataFrame(
        [
            {"path": "a.jpg", "label": "A", "origin_split": "train", "hospital": "h1", "case_id": "c1"},
            {"path": "b.jpg", "label": "B", "origin_split": "train", "hospital": "h1", "case_id": "c1"},
            {"path": "c.jpg", "label": "A", "origin_split": "train", "hospital": "h1", "case_id": "c2"},
            {"path": "d.jpg", "label": "B", "origin_split": "train", "hospital": "h1", "case_id": "c3"},
            {"path": "e.jpg", "label": "A", "origin_split": "test", "hospital": "h2", "case_id": "c4"},
        ]
    )
    with pytest.raises(ValueError, match="multiple labels"):
        split_train_valid(frame, valid_ratio=0.2, required_valid_labels=["A", "B"])
