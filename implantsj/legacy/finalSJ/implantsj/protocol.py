"""Strict, non-executing contracts for the canonical metric-learning protocol."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence

import yaml


class ProtocolError(ValueError):
    """Raised when a canonical protocol contract is violated."""


MODELS = ("Base", "ClosedFT", "ClosedFT-SupCon", "FullFT", "FullFT-SupCon")
PROTOTYPES = ("gt", "mean", "ss")
DISTANCES = ("euclidean", "cosine")
QUERIES = ("validation", "test")
SLICES = ("all", "closed", "open")
PRETEST_FIELDS = (
    "record_type",
    "seed",
    "model",
    "query",
    "prototype",
    "distance",
    "candidate_space",
    "output_path",
    "artifact_sha256",
    "status",
    "protocol_fingerprint",
)
EVALUATION_FIELDS = (
    "seed",
    "model",
    "query",
    "prototype",
    "distance",
    "candidate_space",
    "evaluation_path",
    "evaluation_sha256",
    "status",
    "protocol_fingerprint",
    "freeze_commitment_sha256",
)
COMPARISON_FIELDS = (
    "seed",
    "experiment_id",
    "left_model",
    "right_model",
    "query",
    "prototype",
    "distance",
    "candidate_space",
    "left_evaluation_sha256",
    "right_evaluation_sha256",
    "comparison_path",
    "comparison_sha256",
    "status",
    "protocol_fingerprint",
    "freeze_commitment_sha256",
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def normalized_assignment_hash(rows: Iterable[Mapping[str, Any]]) -> str:
    pairs: list[tuple[str, str]] = []
    ids: set[str] = set()
    for row in rows:
        if set(row) < {"sample_id", "split"}:
            raise ProtocolError("assignment row requires sample_id and split")
        sample_id, split = str(row["sample_id"]), str(row["split"])
        if not sample_id or split not in {"train", "validation", "test"}:
            raise ProtocolError("invalid assignment row")
        if sample_id in ids:
            raise ProtocolError("duplicate assignment sample_id")
        ids.add(sample_id)
        pairs.append((sample_id, split))
    if not pairs:
        raise ProtocolError("assignment must not be empty")
    return fingerprint([[sample_id, split] for sample_id, split in sorted(pairs)])


def _mapping(
    value: Any, name: str, allowed: set[str], required: set[str]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{name} must be a mapping")
    unknown, missing = set(value) - allowed, required - set(value)
    if unknown:
        raise ProtocolError(f"{name} has unknown fields: {sorted(unknown)}")
    if missing:
        raise ProtocolError(f"{name} is missing fields: {sorted(missing)}")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


@dataclass(frozen=True)
class SeedBinding:
    seed: int
    dataset_manifest: str
    assignment_csv_sha256: str
    normalized_assignment_sha256: str
    split_manifest: str
    dataset_sha256: str = ""
    split_sha256: str = ""


@dataclass(frozen=True)
class Protocol:
    raw: Mapping[str, Any]
    fingerprint: str
    seed_bindings: Mapping[int, SeedBinding]

    def seed_binding(self, seed: int) -> SeedBinding:
        try:
            return self.seed_bindings[int(seed)]
        except KeyError as exc:
            raise ProtocolError(f"seed {seed} is not a protocol binding") from exc


_PROTOCOL_FIELDS = {
    "protocol_version",
    "supcon_intent",
    "architecture",
    "base_checkpoint",
    "preprocessing",
    "labels",
    "seed_bindings",
    "budget",
    "encoder_output",
    "prototypes",
    "evaluation",
    "model_roles",
    "treatment_pairs",
}


def _substitute_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _substitute_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute_environment(item) for item in value]
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "${" in expanded:
            raise ProtocolError(f"unresolved environment variable in {value!r}")
        return expanded
    return value


def load_protocol(path: str | Path) -> Protocol:
    with Path(path).open(encoding="utf-8") as handle:
        raw = _substitute_environment(yaml.safe_load(handle))
    data = _mapping(raw, "protocol", _PROTOCOL_FIELDS, _PROTOCOL_FIELDS)
    if data["protocol_version"] != "metric-learning-full42-v1":
        raise ProtocolError("unsupported protocol_version")
    if data["supcon_intent"] != "pure_supcon":
        raise ProtocolError("canonical protocol requires supcon_intent: pure_supcon")
    _mapping(
        data["base_checkpoint"],
        "base_checkpoint",
        {"path", "sha256"},
        {"path", "sha256"},
    )
    if not _is_sha256(data["base_checkpoint"]["sha256"]):
        raise ProtocolError("base_checkpoint.sha256 must be a SHA-256")
    if (
        not isinstance(data["base_checkpoint"]["path"], str)
        or not data["base_checkpoint"]["path"].strip()
    ):
        raise ProtocolError("base_checkpoint.path must be non-empty")
    if not isinstance(data["architecture"], str) or not data["architecture"].strip():
        raise ProtocolError("architecture must be non-empty")
    if not isinstance(data["preprocessing"], dict) or not data["preprocessing"]:
        raise ProtocolError("preprocessing must be a non-empty mapping")
    budget = _mapping(
        data["budget"],
        "budget",
        {
            "epochs",
            "optimizer",
            "scheduler",
            "batch_size",
            "checkpoint_selector",
        },
        {"epochs"},
    )
    if not isinstance(budget["epochs"], int) or budget["epochs"] <= 0:
        raise ProtocolError("budget.epochs must be a positive integer")
    if budget.get("checkpoint_selector", "last") not in {"last", "last.pt"}:
        raise ProtocolError("canonical checkpoint selector must be last.pt")
    if data["encoder_output"] != "encoder":
        raise ProtocolError("canonical encoder_output must be encoder")
    prototypes = _mapping(
        data["prototypes"],
        "prototypes",
        {
            "methods",
            "distances",
            "ss_samples_per_label",
            "ss_seed_policy",
        },
        {"methods"},
    )
    if prototypes["methods"] != list(PROTOTYPES):
        raise ProtocolError("prototype methods must be ordered gt/mean/ss")
    if prototypes.get("distances", list(DISTANCES)) != list(DISTANCES):
        raise ProtocolError("prototype distances must be euclidean/cosine")
    if prototypes.get("ss_samples_per_label", 1) < 1:
        raise ProtocolError("SS prototype sample count must be positive")
    if prototypes.get("ss_seed_policy", "paired_seed") != "paired_seed":
        raise ProtocolError("SS prototype must use paired seed")
    evaluation = _mapping(
        data["evaluation"],
        "evaluation",
        {
            "candidate_space",
            "queries",
            "slices",
            "expected_evaluations_per_seed",
            "expected_metric_rows_per_seed",
            "expected_comparisons_per_seed",
            "expected_delta_rows_per_seed",
        },
        {"candidate_space"},
    )
    if evaluation["candidate_space"] != "full42":
        raise ProtocolError("canonical candidate space must be full42")
    if evaluation.get("queries", list(QUERIES)) != list(QUERIES):
        raise ProtocolError("canonical queries must be validation/test")
    if evaluation.get("slices", list(SLICES)) != list(SLICES):
        raise ProtocolError("canonical slices must be all/closed/open")
    expected_counts = {
        "expected_evaluations_per_seed": 60,
        "expected_metric_rows_per_seed": 180,
        "expected_comparisons_per_seed": 72,
        "expected_delta_rows_per_seed": 216,
    }
    for field, expected in expected_counts.items():
        if field in evaluation and evaluation[field] != expected:
            raise ProtocolError(f"evaluation.{field} must equal {expected}")
    labels = _mapping(
        data["labels"],
        "labels",
        {"ordered", "closed_count", "open_count", "total_count"},
        {"ordered", "closed_count", "open_count", "total_count"},
    )
    if (
        not isinstance(labels["ordered"], list)
        or len(labels["ordered"]) != 42
        or len(set(labels["ordered"])) != 42
        or (labels["closed_count"], labels["open_count"], labels["total_count"])
        != (27, 15, 42)
    ):
        raise ProtocolError("labels must define ordered 42 labels and 27/15/42 counts")
    roles = data["model_roles"]
    if not isinstance(roles, dict) or set(roles) != set(MODELS):
        raise ProtocolError("model_roles must define exactly the five canonical models")
    expected_roles = {
        "Base": (False, "all", "none"),
        "ClosedFT": (True, "closed", "ce"),
        "ClosedFT-SupCon": (True, "closed", "supcon"),
        "FullFT": (True, "all", "ce"),
        "FullFT-SupCon": (True, "all", "supcon"),
    }
    for name, expected in expected_roles.items():
        role = _mapping(
            roles[name],
            f"model_roles.{name}",
            {"training_applied", "scope", "objective"},
            {"training_applied", "scope", "objective"},
        )
        if (
            role["training_applied"],
            role["scope"],
            role["objective"],
        ) != expected:
            raise ProtocolError(f"model role treatment mismatch: {name}")
    bindings_raw = data["seed_bindings"]
    if not isinstance(bindings_raw, dict) or {int(seed) for seed in bindings_raw} != {
        42,
        43,
        44,
    }:
        raise ProtocolError("seed_bindings must define exactly seeds 42, 43, 44")
    bindings: dict[int, SeedBinding] = {}
    allowed = {
        "dataset_manifest",
        "assignment_csv_sha256",
        "normalized_assignment_sha256",
        "split_manifest",
        "dataset_sha256",
        "split_sha256",
    }
    for seed_text, binding in bindings_raw.items():
        seed = int(seed_text)
        b = _mapping(
            binding,
            f"seed_bindings.{seed}",
            allowed,
            {
                "dataset_manifest",
                "assignment_csv_sha256",
                "normalized_assignment_sha256",
                "split_manifest",
                "dataset_sha256",
                "split_sha256",
            },
        )
        for key in (
            "assignment_csv_sha256",
            "normalized_assignment_sha256",
            "dataset_sha256",
            "split_sha256",
        ):
            if key in b and b[key] and not _is_sha256(b[key]):
                raise ProtocolError(f"seed {seed} {key} must be a SHA-256")
        bindings[seed] = SeedBinding(
            seed,
            b["dataset_manifest"],
            b["assignment_csv_sha256"],
            b["normalized_assignment_sha256"],
            b["split_manifest"],
            b.get("dataset_sha256", ""),
            b.get("split_sha256", ""),
        )
    if len({b.normalized_assignment_sha256 for b in bindings.values()}) != 3:
        raise ProtocolError(
            "seed assignments must have three distinct normalized hashes"
        )
    expected_comparison_keys(data)  # validates treatment pairs
    return Protocol(data, fingerprint(data), bindings)


@dataclass(frozen=True, order=True)
class EvaluationKey:
    seed: int
    model: str
    query: str
    prototype: str
    distance: str
    candidate_space: str = "full42"


@dataclass(frozen=True, order=True)
class ComparisonKey:
    seed: int
    experiment_id: str
    left_model: str
    right_model: str
    query: str
    prototype: str
    distance: str
    candidate_space: str = "full42"


def expected_evaluation_keys(seed: int) -> tuple[EvaluationKey, ...]:
    return tuple(
        EvaluationKey(seed, model, query, prototype, distance)
        for query in QUERIES
        for model in MODELS
        for prototype in PROTOTYPES
        for distance in DISTANCES
    )


def expected_pretest_keys(seed: int) -> tuple[EvaluationKey, ...]:
    return expected_evaluation_keys(seed)


def expected_comparison_keys(
    protocol: Protocol | Mapping[str, Any], seed: int | None = None
) -> tuple[ComparisonKey, ...]:
    raw = protocol.raw if isinstance(protocol, Protocol) else protocol
    pairs = raw["treatment_pairs"]
    if not isinstance(pairs, list) or len(pairs) != 12:
        raise ProtocolError(
            "treatment_pairs must contain the twelve declared experiment/distance transitions"
        )
    normalized: list[tuple[str, str, str, str]] = []
    for pair in pairs:
        p = _mapping(
            pair,
            "treatment_pair",
            {"experiment_id", "left_model", "right_model", "distance"},
            {"experiment_id", "left_model", "right_model", "distance"},
        )
        item = (p["experiment_id"], p["left_model"], p["right_model"], p["distance"])
        if (
            item[1] not in MODELS
            or item[2] not in MODELS
            or item[3] not in DISTANCES
            or item[1] == item[2]
        ):
            raise ProtocolError("invalid treatment pair")
        normalized.append(item)
    if len(set(normalized)) != len(normalized):
        raise ProtocolError("duplicate treatment pair")
    # 1a/1b contain three transitions each; 2, 4 and 5 contain one each.
    if {p[0] for p in normalized} != {"1a", "1b", "2a", "2b", "4a", "4b", "5a", "5b"}:
        raise ProtocolError("unexpected treatment-pair experiment mapping")
    seeds = (seed,) if seed is not None else (42,)
    return tuple(
        ComparisonKey(s, exp, left, right, query, prototype, distance)
        for s in seeds
        for query in QUERIES
        for prototype in PROTOTYPES
        for exp, left, right, distance in normalized
    )


def _read_csv(path: str | Path, fields: Sequence[str]) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise ProtocolError(f"ledger columns must be exactly {tuple(fields)}")
        return list(reader)


def _key_from_row(row: Mapping[str, str]) -> EvaluationKey:
    return EvaluationKey(
        int(row["seed"]),
        row["model"],
        row["query"],
        row["prototype"],
        row["distance"],
        row["candidate_space"],
    )


def _verify_artifact(path: str, expected_sha: str) -> None:
    target = Path(path)
    if not target.exists():
        raise ProtocolError(f"missing artifact: {path}")
    actual = (
        sha256_file(target) if target.is_file() else artifact_inventory_hash(target)
    )
    if actual != expected_sha:
        raise ProtocolError(f"artifact hash mismatch: {path}")


def artifact_inventory_hash(directory: str | Path) -> str:
    root = Path(directory)
    if not root.is_dir():
        raise ProtocolError("artifact inventory requires a directory")
    entries = []
    for child in sorted(root.rglob("*")):
        if child.is_file() and child.name != "run.json":
            entries.append([child.relative_to(root).as_posix(), sha256_file(child)])
    if not entries:
        raise ProtocolError("artifact inventory is empty")
    return fingerprint(entries)


def validate_upstream_run(
    directory: str | Path,
    *,
    command: str,
    role: str,
    protocol: Protocol,
    seed: int,
    artifact_sha256: str | None = None,
) -> Mapping[str, Any]:
    root = Path(directory)
    try:
        run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("missing or malformed upstream run.json") from exc
    if (
        run.get("status") != "completed"
        or run.get("command") != command
        or run.get("role") != role
    ):
        raise ProtocolError("upstream run metadata is not canonical completed output")
    if (
        run.get("protocol_fingerprint") != protocol.fingerprint
        or int(run.get("seed", -1)) != seed
    ):
        raise ProtocolError("upstream protocol lineage mismatch")
    expected = artifact_sha256 or run.get("artifact_sha256")
    if not isinstance(expected, str):
        raise ProtocolError("upstream artifact hash missing")
    _verify_artifact(root, expected)
    return run


def validate_pretest_ledger(
    path: str | Path, protocol: Protocol, seed: int, *, verify_artifacts: bool = True
) -> list[dict[str, str]]:
    rows = _read_csv(path, PRETEST_FIELDS)
    if len(rows) != 60:
        raise ProtocolError("pretest ledger must have exactly 60 rows")
    expected = set(expected_pretest_keys(seed))
    seen: set[tuple[str, EvaluationKey]] = set()
    paths: set[str] = set()
    for row in rows:
        if not row["output_path"].strip():
            raise ProtocolError("pretest output path is empty")
        row["output_path"] = str(Path(row["output_path"]).expanduser().resolve())
        key = _key_from_row(row)
        kind = row["record_type"]
        if (
            kind not in {"completed_validation", "planned_test"}
            or key.seed != seed
            or key.candidate_space != "full42"
            or row["protocol_fingerprint"] != protocol.fingerprint
        ):
            raise ProtocolError("invalid pretest ledger row")
        if (kind == "completed_validation") != (key.query == "validation") or (
            kind == "planned_test"
        ) != (key.query == "test"):
            raise ProtocolError("pretest record type/query mismatch")
        if (
            row["status"]
            != ("completed" if kind == "completed_validation" else "planned")
            or (kind == "completed_validation" and not row["artifact_sha256"])
            or (kind == "planned_test" and row["artifact_sha256"])
        ):
            raise ProtocolError("invalid pretest status or artifact hash")
        marker = (kind, key)
        if marker in seen or row["output_path"] in paths:
            raise ProtocolError("duplicate pretest key or output path")
        seen.add(marker)
        paths.add(row["output_path"])
        if kind == "completed_validation" and verify_artifacts:
            validate_upstream_run(
                row["output_path"],
                command="evaluate-prototype",
                role="evaluation",
                protocol=protocol,
                seed=seed,
                artifact_sha256=row["artifact_sha256"],
            )
        if kind == "planned_test" and Path(row["output_path"]).exists():
            raise ProtocolError("planned test destination already exists")
    if {key for _, key in seen} != expected:
        raise ProtocolError("pretest ledger keys do not equal the expected set")
    return rows


def freeze_validation(
    path: str | Path, protocol: Protocol, seed: int, *, verify_artifacts: bool = True
) -> dict[str, Any]:
    rows = validate_pretest_ledger(
        path, protocol, seed, verify_artifacts=verify_artifacts
    )
    validations = sorted(
        (r for r in rows if r["record_type"] == "completed_validation"),
        key=lambda r: canonical_json(r),
    )
    plans = sorted(
        (r for r in rows if r["record_type"] == "planned_test"),
        key=lambda r: canonical_json(r),
    )
    payload = {
        "protocol_fingerprint": protocol.fingerprint,
        "seed": seed,
        "assignment_sha256": protocol.seed_binding(seed).normalized_assignment_sha256,
        "ledger_sha256": sha256_file(path),
        "validation": validations,
        "validation_set_sha256": fingerprint(validations),
        "test_plan": plans,
        "test_plan_sha256": fingerprint(plans),
    }
    payload["freeze_commitment_sha256"] = fingerprint(payload)
    return payload


def validate_evaluation_ledger(
    path: str | Path, protocol: Protocol, seed: int, freeze: Mapping[str, Any]
) -> list[dict[str, str]]:
    rows = _read_csv(path, EVALUATION_FIELDS)
    if len(rows) != 60:
        raise ProtocolError("evaluation ledger must have exactly 60 rows")
    expected = set(expected_evaluation_keys(seed))
    seen: set[EvaluationKey] = set()
    paths: set[str] = set()
    validation = {_key_from_row(row): row for row in freeze["validation"]}
    plans = {_key_from_row(row): row for row in freeze["test_plan"]}
    for row in rows:
        if not row["evaluation_path"].strip():
            raise ProtocolError("evaluation path is empty")
        row["evaluation_path"] = str(
            Path(row["evaluation_path"]).expanduser().resolve()
        )
        key = _key_from_row(row)
        if (
            key not in expected
            or key in seen
            or row["evaluation_path"] in paths
            or row["status"] != "completed"
            or row["protocol_fingerprint"] != protocol.fingerprint
        ):
            raise ProtocolError("invalid evaluation ledger row")
        seen.add(key)
        paths.add(row["evaluation_path"])
        if key.query == "validation":
            frozen = validation.get(key)
            if (
                not frozen
                or row["evaluation_path"] != frozen["output_path"]
                or row["evaluation_sha256"] != frozen["artifact_sha256"]
                or row["freeze_commitment_sha256"]
            ):
                raise ProtocolError("validation ledger does not match freeze")
        else:
            planned = plans.get(key)
            if (
                not planned
                or row["evaluation_path"] != planned["output_path"]
                or row["freeze_commitment_sha256"] != freeze["freeze_commitment_sha256"]
            ):
                raise ProtocolError("test ledger does not match freeze plan")
        validate_upstream_run(
            row["evaluation_path"],
            command="evaluate-prototype",
            role="evaluation",
            protocol=protocol,
            seed=seed,
            artifact_sha256=row["evaluation_sha256"],
        )
    if seen != expected:
        raise ProtocolError("evaluation ledger keys do not equal expected set")
    return rows


def validate_comparison_ledger(
    path: str | Path,
    protocol: Protocol,
    seed: int,
    freeze: Mapping[str, Any],
    evaluations: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    rows = _read_csv(path, COMPARISON_FIELDS)
    expected = set(expected_comparison_keys(protocol, seed))
    if len(rows) != len(expected):
        raise ProtocolError("comparison ledger must have exactly 72 rows")
    evaluation_hashes = {
        _key_from_row(row): row["evaluation_sha256"] for row in evaluations
    }
    seen: set[ComparisonKey] = set()
    paths: set[str] = set()
    for row in rows:
        if not row["comparison_path"].strip():
            raise ProtocolError("comparison path is empty")
        row["comparison_path"] = str(
            Path(row["comparison_path"]).expanduser().resolve()
        )
        key = ComparisonKey(
            int(row["seed"]),
            row["experiment_id"],
            row["left_model"],
            row["right_model"],
            row["query"],
            row["prototype"],
            row["distance"],
            row["candidate_space"],
        )
        if (
            key not in expected
            or key in seen
            or row["comparison_path"] in paths
            or row["status"] != "completed"
            or row["protocol_fingerprint"] != protocol.fingerprint
        ):
            raise ProtocolError("invalid comparison ledger row")
        if row["left_evaluation_sha256"] != evaluation_hashes.get(
            EvaluationKey(seed, key.left_model, key.query, key.prototype, key.distance)
        ) or row["right_evaluation_sha256"] != evaluation_hashes.get(
            EvaluationKey(seed, key.right_model, key.query, key.prototype, key.distance)
        ):
            raise ProtocolError("comparison references incorrect evaluation hash")
        if (key.query == "test") != (
            row["freeze_commitment_sha256"] == freeze["freeze_commitment_sha256"]
        ):
            raise ProtocolError("comparison freeze commitment mismatch")
        validate_upstream_run(
            row["comparison_path"],
            command="compare-results",
            role="comparison",
            protocol=protocol,
            seed=seed,
            artifact_sha256=row["comparison_sha256"],
        )
        seen.add(key)
        paths.add(row["comparison_path"])
    return rows


def validate_freeze_commitment(
    freeze: Mapping[str, Any], protocol: Protocol, seed: int
) -> None:
    required = {
        "protocol_fingerprint",
        "seed",
        "assignment_sha256",
        "ledger_sha256",
        "validation",
        "validation_set_sha256",
        "test_plan",
        "test_plan_sha256",
        "freeze_commitment_sha256",
    }
    receipt_fields = {"frozen_results_sha256", "created_at"}
    if not required <= set(freeze) or set(freeze) - required - receipt_fields:
        raise ProtocolError("freeze has unknown or missing fields")
    if freeze["protocol_fingerprint"] != protocol.fingerprint or freeze["seed"] != seed:
        raise ProtocolError("freeze protocol lineage mismatch")
    if (
        freeze["assignment_sha256"]
        != protocol.seed_binding(seed).normalized_assignment_sha256
    ):
        raise ProtocolError("freeze assignment lineage mismatch")
    if freeze["validation_set_sha256"] != fingerprint(freeze["validation"]):
        raise ProtocolError("freeze validation set hash mismatch")
    if freeze["test_plan_sha256"] != fingerprint(freeze["test_plan"]):
        raise ProtocolError("freeze test plan hash mismatch")
    claimed = freeze["freeze_commitment_sha256"]
    payload = {
        key: value
        for key, value in freeze.items()
        if key != "freeze_commitment_sha256" and key not in receipt_fields
    }
    if not isinstance(claimed, str) or claimed != fingerprint(payload):
        raise ProtocolError("freeze commitment hash mismatch")


def protocol_lineage(
    protocol: Protocol,
    *,
    model: str,
    checkpoint_sha256: str,
    treatment: Mapping[str, Any],
) -> dict[str, Any]:
    if model not in MODELS or not _is_sha256(checkpoint_sha256):
        raise ProtocolError("invalid lineage model or checkpoint SHA")
    descriptor = {
        "model": model,
        "treatment": dict(treatment),
        "checkpoint_sha256": checkpoint_sha256,
    }
    invariant = {
        "protocol_version": protocol.raw["protocol_version"],
        "architecture": protocol.raw["architecture"],
        "base_checkpoint_sha256": protocol.raw["base_checkpoint"]["sha256"],
        "preprocessing": protocol.raw["preprocessing"],
        "labels": protocol.raw["labels"],
        "budget": protocol.raw["budget"],
        "encoder_output": protocol.raw["encoder_output"],
        "prototypes": protocol.raw["prototypes"],
        "evaluation": protocol.raw["evaluation"],
    }
    return {"invariant_fingerprint": fingerprint(invariant), "treatment": descriptor}


def validate_lineage_compatibility(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    protocol: Protocol,
    *,
    experiment_id: str,
    distance: str,
) -> None:
    expected_invariant = protocol_lineage(
        protocol,
        model="Base",
        checkpoint_sha256=protocol.raw["base_checkpoint"]["sha256"],
        treatment=protocol.raw["model_roles"]["Base"],
    )["invariant_fingerprint"]
    if (
        left.get("invariant_fingerprint") != expected_invariant
        or right.get("invariant_fingerprint") != expected_invariant
    ):
        raise ProtocolError("lineage does not match canonical protocol")
    if left.get("invariant_fingerprint") != right.get("invariant_fingerprint"):
        raise ProtocolError("lineage invariant drift")
    left_treatment, right_treatment = left.get("treatment"), right.get("treatment")
    if not isinstance(left_treatment, dict) or not isinstance(right_treatment, dict):
        raise ProtocolError("malformed treatment lineage")
    for descriptor in (left_treatment, right_treatment):
        model = descriptor.get("model")
        if (
            model not in MODELS
            or descriptor.get("treatment") != protocol.raw["model_roles"][model]
            or not _is_sha256(descriptor.get("checkpoint_sha256"))
        ):
            raise ProtocolError("treatment lineage does not match declared model role")
    declared = {
        (p["experiment_id"], p["left_model"], p["right_model"], p["distance"])
        for p in protocol.raw["treatment_pairs"]
    }
    actual = (
        experiment_id,
        left_treatment.get("model"),
        right_treatment.get("model"),
        distance,
    )
    if actual not in declared:
        raise ProtocolError("undeclared ordered treatment transition")


def audit_study(
    evaluation_ledger: str | Path,
    comparison_ledger: str | Path,
    protocol: Protocol,
    seed: int,
    freeze: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the complete explicit post-test ledgers without discovering or executing work."""
    validate_freeze_commitment(freeze, protocol, seed)
    evaluations = validate_evaluation_ledger(evaluation_ledger, protocol, seed, freeze)
    comparisons = validate_comparison_ledger(
        comparison_ledger, protocol, seed, freeze, evaluations
    )
    return {
        "protocol_fingerprint": protocol.fingerprint,
        "seed": seed,
        "freeze_commitment_sha256": freeze["freeze_commitment_sha256"],
        "evaluation_ledger_sha256": sha256_file(evaluation_ledger),
        "comparison_ledger_sha256": sha256_file(comparison_ledger),
        "evaluation_count": len(evaluations),
        "comparison_count": len(comparisons),
        "result_slice_count": len(evaluations) * len(SLICES),
        "comparison_slice_count": len(comparisons) * len(SLICES),
    }


def paired_uncertainty(
    values: Sequence[float], paired_deltas: Sequence[float] | None = None
) -> dict[str, float]:
    if len(values) != 3:
        raise ProtocolError("paired uncertainty requires exactly three values")
    if paired_deltas is not None and len(paired_deltas) != 3:
        raise ProtocolError("paired uncertainty requires exactly three paired deltas")
    if not all(math.isfinite(float(value)) for value in values):
        raise ProtocolError("paired uncertainty values must be finite")
    if paired_deltas is not None and not all(
        math.isfinite(float(value)) for value in paired_deltas
    ):
        raise ProtocolError("paired uncertainty deltas must be finite")

    def summary(xs: Sequence[float], prefix: str = "") -> dict[str, float]:
        avg = mean(xs)
        sd = stdev(xs)
        se = sd / math.sqrt(3)
        margin = 4.302652729911275 * se
        return {
            f"{prefix}mean": avg,
            f"{prefix}sd": sd,
            f"{prefix}se": se,
            f"{prefix}ci95_low": avg - margin,
            f"{prefix}ci95_high": avg + margin,
            f"{prefix}n": 3.0,
            f"{prefix}df": 2.0,
        }

    result = summary(values)
    if paired_deltas is not None:
        result.update(summary(paired_deltas, "paired_"))
    return result


__all__ = [
    "ProtocolError",
    "Protocol",
    "SeedBinding",
    "EvaluationKey",
    "ComparisonKey",
    "canonical_json",
    "sha256_file",
    "fingerprint",
    "normalized_assignment_hash",
    "load_protocol",
    "expected_evaluation_keys",
    "expected_pretest_keys",
    "expected_comparison_keys",
    "artifact_inventory_hash",
    "validate_upstream_run",
    "validate_pretest_ledger",
    "freeze_validation",
    "validate_freeze_commitment",
    "protocol_lineage",
    "validate_lineage_compatibility",
    "validate_evaluation_ledger",
    "validate_comparison_ledger",
    "audit_study",
    "paired_uncertainty",
]
