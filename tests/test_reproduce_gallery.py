import json

import numpy as np
import pandas as pd
import pytest

from pipelines.eval.reproduce_gallery import evaluate, select_support, validate_cache
from tools.improve_unseen_mahalanobis import candidate_scorers, self_check


@pytest.fixture
def cache(tmp_path):
    rng = np.random.default_rng(123)
    result = {"labels": np.asarray(["a", "b", "c"])}
    for split, count in (("train", 12), ("valid", 3), ("test", 4)):
        x = np.vstack([rng.normal(label * 4, 0.3, size=(count, 6)) for label in range(3)])
        paths = []
        for index in range(len(x)):
            path = tmp_path / f"{split}_{index}.bin"
            path.write_bytes(f"{split}-{index}".encode())
            paths.append(str(path))
        result.update({f"{split}_x": x, f"{split}_y": np.repeat(np.arange(3), count),
                       f"{split}_paths": np.asarray(paths)})
    return result


def test_historical_distance_self_check():
    self_check()


def test_strict_support_and_query_are_disjoint(cache, tmp_path):
    out = tmp_path / "strict"
    result = evaluate(cache, out, protocol="strict", policies=("ss",), shots=10,
                      seeds=(0, 42), covariance="support", methods=("pooled_ledoit_wolf",))
    assert len(result) == 2 and set(result.n) == {12}
    membership = pd.read_csv(out / "membership.csv")
    for _, rows in membership.groupby("seed"):
        support = rows[rows.role == "support"]
        assert support.groupby("label").size().tolist() == [10, 10, 10]
        assert set(support.sha256).isdisjoint(rows[rows.role == "test"].sha256)
    assert (out / "per_class.csv").is_file()


def test_episodic_query_is_pool_minus_support(cache, tmp_path):
    result = evaluate(cache, tmp_path / "episodes", protocol="episodic", policies=("ss",), shots=5,
                      seeds=(42,), covariance="support", methods=("pooled_ledoit_wolf",))
    assert result.iloc[0].n == 3 * (12 + 3 + 4) - 15
    membership = pd.read_csv(tmp_path / "episodes/membership.csv")
    assert not membership.path.duplicated().any()


def test_strict_rejects_relaxed_covariance_and_validation_selection(cache, tmp_path):
    for kwargs in ({"covariance": "train"}, {"covariance": "support", "select_on_valid": True}):
        with pytest.raises(ValueError, match="support-only"):
            evaluate(cache, tmp_path / "invalid", protocol="strict", policies=("ss",), **kwargs)
    assert not (tmp_path / "invalid").exists()


def test_exact_duplicates_are_not_silently_called_strict(cache, tmp_path):
    from pathlib import Path
    Path(cache["test_paths"][0]).write_bytes(Path(cache["train_paths"][0]).read_bytes())
    with pytest.raises(ValueError, match="exact duplicate"):
        evaluate(cache, tmp_path / "duplicate", protocol="episodic", policies=("ss",), covariance="support")


def test_query_never_changes_fitted_distance(cache):
    x, y = cache["train_x"], cache["train_y"]
    selected = select_support(y, cache["train_paths"], 10, 42, True)
    scorers = dict((name, fn) for name, _, fn in candidate_scorers(x[selected], y[selected], 3, 42))
    query = cache["test_x"][:2]
    before = scorers["pooled_ledoit_wolf"](query)
    cache["valid_x"][:] = 1e9
    np.testing.assert_array_equal(before, scorers["pooled_ledoit_wolf"](query))


def test_cache_rejects_nonfinite_or_misaligned_inputs(cache):
    cache["test_x"][0, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        validate_cache(cache)
    with pytest.raises(ValueError, match="fewer"):
        select_support(cache["train_y"], cache["train_paths"], 13, 42, True)


def test_cli_writes_provenance_and_refuses_overwrite(cache, tmp_path, monkeypatch):
    from pipelines.eval import reproduce_gallery
    source = tmp_path / "cache.npz"
    np.savez_compressed(source, **cache)
    out = tmp_path / "cli"
    monkeypatch.setattr("sys.argv", ["reproduce_gallery", "--cache", str(source), "--normalization", "l2",
                                   "--protocol", "strict", "--policies", "ss", "--covariance", "support",
                                   "--methods", "pooled_ledoit_wolf", "--output-dir", str(out)])
    assert reproduce_gallery.main() == 0
    assert json.loads((out / "protocol.json").read_text())["historical_numeric_parity"] == "not verified"
    with pytest.raises(FileExistsError):
        reproduce_gallery.main()
