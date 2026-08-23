import csv
import json

import pytest
import yaml

from implantsj.protocol import (
    PRETEST_FIELDS,
    ProtocolError,
    artifact_inventory_hash,
    expected_comparison_keys,
    expected_evaluation_keys,
    freeze_validation,
    load_protocol,
    normalized_assignment_hash,
    paired_uncertainty,
    validate_pretest_ledger,
    validate_upstream_run,
)


def _sha(character):
    return character * 64


def _pairs():
    return [
        {
            "experiment_id": experiment,
            "left_model": left,
            "right_model": right,
            "distance": distance,
        }
        for experiment, left, right, distance in [
            ("1a", "Base", "ClosedFT", "euclidean"),
            ("1a", "Base", "ClosedFT-SupCon", "euclidean"),
            ("1a", "ClosedFT", "ClosedFT-SupCon", "euclidean"),
            ("1b", "Base", "ClosedFT", "cosine"),
            ("1b", "Base", "ClosedFT-SupCon", "cosine"),
            ("1b", "ClosedFT", "ClosedFT-SupCon", "cosine"),
            ("2a", "ClosedFT", "FullFT", "euclidean"),
            ("2b", "ClosedFT", "FullFT", "cosine"),
            ("4a", "ClosedFT", "ClosedFT-SupCon", "euclidean"),
            ("4b", "ClosedFT", "ClosedFT-SupCon", "cosine"),
            ("5a", "FullFT", "FullFT-SupCon", "euclidean"),
            ("5b", "FullFT", "FullFT-SupCon", "cosine"),
        ]
    ]


def _raw():
    return {
        "protocol_version": "metric-learning-full42-v1",
        "supcon_intent": "pure_supcon",
        "architecture": "vit",
        "preprocessing": {"name": "fixed"},
        "base_checkpoint": {"path": "base.pt", "sha256": _sha("a")},
        "labels": {
            "ordered": [f"label-{i}" for i in range(42)],
            "closed_count": 27,
            "open_count": 15,
            "total_count": 42,
        },
        "seed_bindings": {
            str(seed): {
                "dataset_manifest": f"data-{seed}.yaml",
                "assignment_csv_sha256": _sha(str(index + 1)),
                "normalized_assignment_sha256": _sha(str(index + 4)),
                "split_manifest": f"split-{seed}.yaml",
                "dataset_sha256": _sha(f"{index + 7:x}"),
                "split_sha256": _sha(f"{index + 10:x}"),
            }
            for index, seed in enumerate((42, 43, 44))
        },
        "budget": {"epochs": 1},
        "encoder_output": "encoder",
        "prototypes": {"methods": ["gt", "mean", "ss"]},
        "evaluation": {"candidate_space": "full42"},
        "model_roles": {
            "Base": {
                "training_applied": False,
                "scope": "all",
                "objective": "none",
            },
            "ClosedFT": {
                "training_applied": True,
                "scope": "closed",
                "objective": "ce",
            },
            "ClosedFT-SupCon": {
                "training_applied": True,
                "scope": "closed",
                "objective": "supcon",
            },
            "FullFT": {
                "training_applied": True,
                "scope": "all",
                "objective": "ce",
            },
            "FullFT-SupCon": {
                "training_applied": True,
                "scope": "all",
                "objective": "supcon",
            },
        },
        "treatment_pairs": _pairs(),
    }


def _protocol(tmp_path):
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(_raw()), encoding="utf-8")
    return load_protocol(path)


def _ledger_rows(protocol, tmp_path):
    rows = []
    for key in expected_evaluation_keys(42):
        complete = key.query == "validation"
        rows.append(
            {
                "record_type": "completed_validation" if complete else "planned_test",
                "seed": "42",
                "model": key.model,
                "query": key.query,
                "prototype": key.prototype,
                "distance": key.distance,
                "candidate_space": "full42",
                "output_path": str(
                    tmp_path / f"{key.model}-{key.query}-{key.prototype}-{key.distance}"
                ),
                "artifact_sha256": _sha("a") if complete else "",
                "status": "completed" if complete else "planned",
                "protocol_fingerprint": protocol.fingerprint,
            }
        )
    return rows


def _write_ledger(path, rows, fields=PRETEST_FIELDS):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def test_protocol_is_strict_and_requires_pure_supcon(tmp_path):
    raw = _raw()
    raw["extra"] = True
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ProtocolError, match="unknown"):
        load_protocol(path)
    raw = _raw()
    raw["supcon_intent"] = "ce_plus_supcon"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ProtocolError, match="pure_supcon"):
        load_protocol(path)
    raw = _raw()
    raw["base_checkpoint"]["sha256"] = "z" * 64
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ProtocolError, match="SHA-256"):
        load_protocol(path)
    raw = _raw()
    del raw["seed_bindings"]["42"]["dataset_sha256"]
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ProtocolError, match="missing"):
        load_protocol(path)


def test_assignment_hash_is_order_invariant_and_detects_change():
    first = [{"sample_id": "b", "split": "test"}, {"sample_id": "a", "split": "train"}]
    assert normalized_assignment_hash(first) == normalized_assignment_hash(
        reversed(first)
    )
    assert normalized_assignment_hash(first) != normalized_assignment_hash(
        [
            {"sample_id": "b", "split": "validation"},
            {"sample_id": "a", "split": "train"},
        ]
    )


def test_static_key_totals_and_mapping(tmp_path):
    protocol = _protocol(tmp_path)
    assert len(expected_evaluation_keys(42)) == 60
    assert len(expected_comparison_keys(protocol, 42)) == 72
    assert {
        (key.experiment_id, key.distance)
        for key in expected_comparison_keys(protocol, 42)
    } == {(pair["experiment_id"], pair["distance"]) for pair in _pairs()}


def test_pretest_ledger_rejects_missing_duplicate_and_malformed(tmp_path):
    protocol = _protocol(tmp_path)
    rows = _ledger_rows(protocol, tmp_path)
    ledger = tmp_path / "ledger.csv"
    _write_ledger(ledger, rows[:-1])
    with pytest.raises(ProtocolError, match="60"):
        validate_pretest_ledger(ledger, protocol, 42, verify_artifacts=False)
    rows = _ledger_rows(protocol, tmp_path)
    rows[-1] = dict(rows[-2])
    _write_ledger(ledger, rows)
    with pytest.raises(ProtocolError, match="duplicate"):
        validate_pretest_ledger(ledger, protocol, 42, verify_artifacts=False)
    _write_ledger(ledger, _ledger_rows(protocol, tmp_path), PRETEST_FIELDS[:-1])
    with pytest.raises(ProtocolError, match="columns"):
        validate_pretest_ledger(ledger, protocol, 42, verify_artifacts=False)


def test_freeze_commitment_is_deterministic(tmp_path):
    protocol = _protocol(tmp_path)
    ledger = tmp_path / "ledger.csv"
    _write_ledger(ledger, _ledger_rows(protocol, tmp_path))
    assert freeze_validation(
        ledger, protocol, 42, verify_artifacts=False
    ) == freeze_validation(ledger, protocol, 42, verify_artifacts=False)


def test_upstream_validation_rejects_altered_artifacts(tmp_path):
    protocol = _protocol(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    artifact = output / "payload.txt"
    artifact.write_text("original")
    digest = artifact_inventory_hash(output)
    (output / "run.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "command": "evaluate-prototype",
                "role": "evaluation",
                "protocol_fingerprint": protocol.fingerprint,
                "seed": 42,
                "artifact_sha256": digest,
            }
        )
    )
    validate_upstream_run(
        output,
        command="evaluate-prototype",
        role="evaluation",
        protocol=protocol,
        seed=42,
    )
    artifact.write_text("altered")
    with pytest.raises(ProtocolError, match="hash mismatch"):
        validate_upstream_run(
            output,
            command="evaluate-prototype",
            role="evaluation",
            protocol=protocol,
            seed=42,
        )


def test_paired_uncertainty_uses_student_t_n_three():
    summary = paired_uncertainty([1.0, 2.0, 3.0], [0.5, 1.0, 1.5])
    assert summary["n"] == 3 and summary["df"] == 2
    assert summary["ci95_low"] < summary["mean"] < summary["ci95_high"]
    assert summary["paired_mean"] == 1.0
    with pytest.raises(ProtocolError, match="three"):
        paired_uncertainty([1.0, 2.0])
    with pytest.raises(ProtocolError, match="finite"):
        paired_uncertainty([1.0, float("nan"), 3.0])
