"""Convert dataset manifests into deterministic canonical records and views."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
from PIL import Image, UnidentifiedImageError

from .schema import DataError, DependencyError, sha256_json


CANONICAL_COLUMNS = (
    "sample_id",
    "image_uri",
    "image_key",
    "label",
    "source_id",
    "source_role",
    "split",
    "fold",
    "group_id",
    "content_hash",
    "metadata",
)
FILTER_FIELDS = set(CANONICAL_COLUMNS) - {"metadata"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class MaterializedDataset:
    dataset_id: str
    records: pd.DataFrame
    views: dict[str, pd.DataFrame]
    view_usages: dict[str, str]
    view_parents: dict[str, str]
    label_map: dict[str, int]
    hashes: dict[str, str]
    summary: dict[str, Any]
    duplicates: list[dict[str, Any]]


def _normalize_label(value: Any, context: str) -> str:
    if value is None or pd.isna(value):
        raise DataError(f"{context}: label is null")
    normalized = unicodedata.normalize("NFC", str(value)).strip()
    if not normalized:
        raise DataError(f"{context}: label is empty")
    return normalized


_CANONICAL_SPLITS = frozenset({"train", "validation", "test"})


def _canonical_text(value: Any, context: str) -> str:
    if value is None or pd.isna(value):
        raise DataError(f"{context}: value is null")
    text = unicodedata.normalize("NFC", str(value)).strip()
    if not text:
        raise DataError(f"{context}: value is empty")
    return text


def _canonical_frame(
    frame: pd.DataFrame,
    *,
    context: str,
    fields: Iterable[str],
) -> pd.DataFrame:
    required = tuple(fields)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise DataError(f"{context}: missing columns: {missing}")
    if frame.empty:
        raise DataError(f"{context}: table is empty")
    result = frame.loc[:, required].copy()
    for field in required:
        result[field] = [
            _canonical_text(value, f"{context} {field}")
            for value in result[field].tolist()
        ]
    return result


def canonical_table_identity(
    frame: pd.DataFrame,
    *,
    context: str = "canonical table",
    fields: Iterable[str] = ("sample_id", "image_uri", "label", "group_id"),
) -> str:
    """Return a row-order-independent identity hash for a canonical table."""
    normalized = _canonical_frame(frame, context=context, fields=fields)
    if "sample_id" in normalized:
        duplicate_ids = sorted(
            normalized.loc[normalized["sample_id"].duplicated(), "sample_id"].unique()
        )
        if duplicate_ids:
            raise DataError(f"{context}: duplicate sample_id values: {duplicate_ids}")
    rows = normalized.sort_values(list(normalized.columns)).to_dict("records")
    return sha256_json(rows)


def canonical_assignment_sha256(
    frame: pd.DataFrame, *, context: str = "canonical assignment"
) -> str:
    """Return the canonical hash of one supplied predefined split assignment."""
    assignments = _canonical_frame(
        frame, context=context, fields=("sample_id", "split")
    )
    invalid = sorted(set(assignments["split"]) - _CANONICAL_SPLITS)
    if invalid:
        raise DataError(f"{context}: invalid predefined splits: {invalid}")
    duplicate_ids = sorted(
        assignments.loc[assignments["sample_id"].duplicated(), "sample_id"].unique()
    )
    if duplicate_ids:
        raise DataError(f"{context}: duplicate sample_id assignments: {duplicate_ids}")
    return sha256_json(
        [
            {"sample_id": row.sample_id, "split": row.split}
            for row in assignments.sort_values("sample_id").itertuples(index=False)
        ]
    )


def validate_distinct_assignments(
    assignments: Mapping[str, pd.DataFrame],
) -> dict[str, str]:
    """Hash named assignments and reject reused assignment content."""
    hashes = {
        name: canonical_assignment_sha256(frame, context=f"assignment {name}")
        for name, frame in assignments.items()
    }
    by_hash: dict[str, list[str]] = {}
    for name, digest in hashes.items():
        by_hash.setdefault(digest, []).append(name)
    duplicates = [names for names in by_hash.values() if len(names) > 1]
    if duplicates:
        raise DataError(f"canonical assignments: duplicate content: {duplicates}")
    return hashes


def validate_canonical_source(source: dict[str, Any]) -> pd.DataFrame:
    """Load and validate one canonical CSV source without materializing views."""
    role = source.get("role")
    if source.get("type") not in {"csv", "parquet"}:
        raise DataError(
            f"source {source.get('id', '<unknown>')}: canonical sources must be tables"
        )
    if role not in {"sample", "prototype"}:
        raise DataError(
            f"source {source.get('id', '<unknown>')}: invalid canonical role {role!r}"
        )
    columns = source.get("columns", {})
    required = {"image_uri", "label", "sample_id", "group_id"}
    if role == "sample":
        required.add("split")
    missing = sorted(required - set(columns))
    if missing:
        raise DataError(
            f"source {source.get('id', '<unknown>')}: missing canonical columns: {missing}"
        )
    if role == "prototype" and "split" in columns:
        raise DataError(
            f"source {source.get('id', '<unknown>')}: canonical prototype has split"
        )
    frame = _table(source).rename(
        columns={
            source_column: canonical for canonical, source_column in columns.items()
        }
    )
    frame["source_role"] = role
    validate_canonical_table(
        frame,
        role=role,
        require_split=role == "sample",
        context=f"source {source.get('id', '<unknown>')}",
    )
    return frame


def validate_canonical_table(
    frame: pd.DataFrame,
    *,
    role: str,
    require_split: bool,
    context: str = "canonical table",
) -> None:
    """Require explicit canonical identity fields and the declared source role."""
    if role not in {"sample", "prototype"}:
        raise DataError(f"{context}: unsupported canonical role {role!r}")
    identity = _canonical_frame(
        frame,
        context=context,
        fields=("sample_id", "image_uri", "label", "group_id"),
    )
    duplicate_ids = sorted(
        identity.loc[identity["sample_id"].duplicated(), "sample_id"].unique()
    )
    if duplicate_ids:
        raise DataError(f"{context}: duplicate sample_id values: {duplicate_ids}")
    if "source_role" in frame.columns:
        roles = {
            _canonical_text(value, f"{context} source_role")
            for value in frame["source_role"].tolist()
        }
        if roles != {role}:
            raise DataError(f"{context}: source_role must be exactly {role!r}")
    if require_split:
        canonical_assignment_sha256(frame, context=context)
    elif "split" in frame.columns:
        values = frame["split"].dropna()
        if not values.empty:
            raise DataError(f"{context}: canonical {role} table must not contain split")


def validate_same_universe(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    reference_name: str = "reference",
    candidate_name: str = "candidate",
) -> str:
    """Require equal non-assignment canonical sample universes."""
    reference_hash = canonical_table_identity(reference, context=reference_name)
    candidate_hash = canonical_table_identity(candidate, context=candidate_name)
    if reference_hash != candidate_hash:
        raise DataError(
            f"{candidate_name}: canonical universe differs from {reference_name}"
        )
    return reference_hash


def validate_common_gt_identity(
    gt_tables: Mapping[str, pd.DataFrame],
) -> str:
    """Require every named GT table to have identical identities and content."""
    if not gt_tables:
        raise DataError("canonical GT: no tables supplied")
    iterator = iter(gt_tables.items())
    reference_name, reference = next(iterator)
    validate_canonical_table(
        reference, role="prototype", require_split=False, context=f"GT {reference_name}"
    )
    reference_hash = canonical_table_identity(
        reference,
        context=f"GT {reference_name}",
        fields=("sample_id", "image_uri", "label", "group_id", "content_hash"),
    )
    for name, table in iterator:
        validate_canonical_table(
            table, role="prototype", require_split=False, context=f"GT {name}"
        )
        candidate_hash = canonical_table_identity(
            table,
            context=f"GT {name}",
            fields=("sample_id", "image_uri", "label", "group_id", "content_hash"),
        )
        if candidate_hash != reference_hash:
            raise DataError(f"GT {name}: content differs from GT {reference_name}")
    return reference_hash


def validate_exact_label_coverage(
    views: Mapping[str, pd.DataFrame],
    *,
    closed_labels: Iterable[str],
    open_labels: Iterable[str],
    minimums: Mapping[str, int | Mapping[str, int]],
) -> None:
    """Require caller-declared exact labels and per-label minimum coverage."""
    closed = {_normalize_label(label, "closed label") for label in closed_labels}
    opened = {_normalize_label(label, "open label") for label in open_labels}
    overlap = sorted(closed & opened)
    if overlap:
        raise DataError(f"canonical labels: closed/open overlap: {overlap}")
    expected = closed | opened
    if not expected:
        raise DataError("canonical labels: no labels supplied")
    for view_id, minimum in minimums.items():
        if view_id not in views:
            raise DataError(f"canonical labels: missing view {view_id!r}")
        labels = [
            _normalize_label(value, f"view {view_id} label")
            for value in views[view_id]["label"].tolist()
        ]
        actual = set(labels)
        if actual != expected:
            raise DataError(
                f"view {view_id}: labels must exactly match canonical label space; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        if isinstance(minimum, Mapping):
            required = {
                _normalize_label(label, f"view {view_id} minimum label"): count
                for label, count in minimum.items()
            }
        else:
            required = {label: minimum for label in expected}
        for label, count in required.items():
            if label not in expected or not isinstance(count, int) or count < 1:
                raise DataError(f"view {view_id}: invalid minimum for {label!r}")
            if labels.count(label) < count:
                raise DataError(
                    f"view {view_id}: label {label!r} has fewer than {count} samples"
                )


def validate_canonical_isolation(
    views: Mapping[str, pd.DataFrame], *, gt_view: str = "gt"
) -> None:
    """Require pairwise sample, file-content, and group isolation across views."""
    required = ("train", "validation", "test", gt_view)
    missing = sorted(set(required) - set(views))
    if missing:
        raise DataError(f"canonical isolation: missing views: {missing}")
    identities: dict[str, dict[str, set[str]]] = {}
    for view_id in required:
        frame = _canonical_frame(
            views[view_id],
            context=f"view {view_id}",
            fields=("sample_id", "content_hash", "group_id"),
        )
        identities[view_id] = {
            field: set(frame[field])
            for field in ("sample_id", "content_hash", "group_id")
        }
    for offset, left in enumerate(required):
        for right in required[offset + 1 :]:
            for field in ("sample_id", "content_hash", "group_id"):
                overlap = sorted(identities[left][field] & identities[right][field])
                if overlap:
                    raise DataError(
                        f"canonical isolation: {field} overlap between {left} and "
                        f"{right}: {overlap}"
                    )


def canonical_isolation_audit(
    views: Mapping[str, pd.DataFrame], *, gt_view: str = "gt"
) -> dict[str, Any]:
    """Return non-sensitive proof for a validated four-way isolation boundary."""
    validate_canonical_isolation(views, gt_view=gt_view)
    required = ("train", "validation", "test", gt_view)
    audit_views: dict[str, Any] = {}
    for view_id in required:
        frame = _canonical_frame(
            views[view_id],
            context=f"view {view_id}",
            fields=("sample_id", "content_hash", "group_id"),
        )
        audit_views[view_id] = {
            "rows": len(frame),
            "sample_id_sha256": sha256_json(sorted(frame["sample_id"].tolist())),
            "content_hash_sha256": sha256_json(sorted(frame["content_hash"].tolist())),
            "group_id_sha256": sha256_json(sorted(frame["group_id"].tolist())),
        }
    return {
        "schema_version": 1,
        "views": audit_views,
        "pairwise_disjoint": {
            "sample_id": True,
            "content_hash": True,
            "group_id": True,
        },
    }


def canonical_materialized_metadata(dataset: MaterializedDataset) -> dict[str, Any]:
    """Return canonical provenance metadata for a validated materialization."""
    source_hashes = {
        source_id: canonical_table_identity(
            source_frame,
            context=f"source {source_id}",
            fields=(
                "sample_id",
                "image_uri",
                "label",
                "group_id",
                "content_hash",
            ),
        )
        for source_id, source_frame in dataset.records.groupby("source_id", sort=True)
    }
    assignment_rows = dataset.records.loc[
        dataset.records["source_role"] == "sample", ["sample_id", "split"]
    ]
    return {
        "roles": dict(dataset.summary["roles"]),
        "views": {
            view_id: {
                "role": sorted(set(dataset.views[view_id]["source_role"])),
                "usage": dataset.view_usages[view_id],
                "parent": dataset.view_parents[view_id],
                "hash": dataset.hashes[f"view:{view_id}"],
            }
            for view_id in sorted(dataset.views)
        },
        "source_hashes": source_hashes,
        "assignment_sha256": canonical_assignment_sha256(assignment_rows),
    }


def _resolved_aliases(config: dict[str, Any], dataset_id: str) -> dict[str, str]:
    raw = {
        _normalize_label(
            source, f"dataset {dataset_id} alias source"
        ): _normalize_label(target, f"dataset {dataset_id} alias target")
        for source, target in config.items()
    }
    resolved: dict[str, str] = {}
    for source in raw:
        seen: list[str] = []
        value = source
        while value in raw:
            if value in seen:
                chain = " -> ".join((*seen, value))
                raise DataError(f"dataset {dataset_id}: alias cycle: {chain}")
            seen.append(value)
            value = raw[value]
        resolved[source] = value
    return resolved


def _file_sha256(path: Path, cache: dict[Path, str]) -> str:
    if path in cache:
        return cache[path]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    cache[path] = digest.hexdigest()
    return cache[path]


def _validate_image(path: Path, context: str) -> None:
    if not path.is_file():
        raise DataError(f"{context}: image does not exist: {path}")
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise DataError(f"{context}: image decode failed: {path}: {exc}") from exc


def _image_identity(path: Path, root: Path, context: str) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise DataError(
            f"{context}: image is outside source root {root}: {path}"
        ) from exc


def _table(source: dict[str, Any]) -> pd.DataFrame:
    path = Path(source["path"])
    try:
        if source["type"] == "csv":
            return pd.read_csv(path)
        return pd.read_parquet(path)
    except ImportError as exc:
        raise DependencyError(
            "Parquet support requires pyarrow; install requirements.txt"
        ) from exc
    except Exception as exc:
        raise DataError(f"source {source['id']}: cannot read {path}: {exc}") from exc


def _table_records(
    source: dict[str, Any],
    aliases: dict[str, str],
    hash_cache: dict[Path, str],
) -> list[dict[str, Any]]:
    frame = _table(source)
    columns = source["columns"]
    missing = sorted(set(columns.values()) - set(frame.columns))
    if missing:
        raise DataError(
            f"source {source['id']}: mapped columns are missing from {source['path']}: {missing}"
        )
    source_path = Path(source["path"])
    image_root = Path(source.get("image_root", source_path.parent)).resolve()
    records: list[dict[str, Any]] = []
    for offset, (_, row) in enumerate(frame.iterrows(), start=2):
        context = f"source {source['id']} row {offset}"
        raw_uri = row[columns["image_uri"]]
        if raw_uri is None or pd.isna(raw_uri) or not str(raw_uri).strip():
            raise DataError(f"{context}: image_uri is empty")
        image_path = Path(str(raw_uri))
        if not image_path.is_absolute():
            image_path = image_root / image_path
        image_path = image_path.resolve()
        _validate_image(image_path, context)
        image_key = _image_identity(image_path, image_root, context)
        label = _normalize_label(row[columns["label"]], context)
        label = aliases.get(label, label)
        supplied_id = row[columns["sample_id"]] if "sample_id" in columns else None
        if supplied_id is None or pd.isna(supplied_id):
            sample_id = hashlib.sha256(
                f"{source['id']}\0{image_key}".encode("utf-8")
            ).hexdigest()
        else:
            sample_id = str(supplied_id).strip()
            if not sample_id:
                raise DataError(f"{context}: sample_id is empty")
        metadata: dict[str, Any] = {}
        if "source_metadata" in columns:
            raw_metadata = row[columns["source_metadata"]]
            if isinstance(raw_metadata, dict):
                metadata = raw_metadata
            elif raw_metadata is not None and not pd.isna(raw_metadata):
                try:
                    parsed = json.loads(str(raw_metadata))
                except json.JSONDecodeError as exc:
                    raise DataError(
                        f"{context}: source_metadata must be a JSON object"
                    ) from exc
                if not isinstance(parsed, dict):
                    raise DataError(f"{context}: source_metadata must be a JSON object")
                metadata = parsed

        def optional(name: str) -> Any:
            if name not in columns:
                return None
            value = row[columns[name]]
            return None if value is None or pd.isna(value) else value

        records.append(
            {
                "sample_id": sample_id,
                "image_uri": str(image_path),
                "image_key": image_key,
                "label": label,
                "source_id": source["id"],
                "source_role": source["role"],
                "split": optional("split"),
                "fold": optional("fold"),
                "group_id": optional("group_id"),
                "content_hash": _file_sha256(image_path, hash_cache),
                "metadata": metadata,
            }
        )
    if not records:
        raise DataError(f"source {source['id']}: source is empty")
    return records


def _folder_records(
    source: dict[str, Any],
    aliases: dict[str, str],
    hash_cache: dict[Path, str],
) -> list[dict[str, Any]]:
    root = Path(source["path"]).resolve()
    if not root.is_dir():
        raise DataError(f"source {source['id']}: folder does not exist: {root}")
    extensions = {
        item.lower() if str(item).startswith(".") else f".{str(item).lower()}"
        for item in source.get("extensions", IMAGE_SUFFIXES)
    }
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.suffix.lower() not in extensions:
            continue
        relative = path.relative_to(root)
        if len(relative.parts) < 2:
            raise DataError(
                f"source {source['id']}: expected root/<label>/<image>, got {path}"
            )
        context = f"source {source['id']} image {relative.as_posix()}"
        _validate_image(path, context)
        normalized_label = _normalize_label(relative.parts[0], context)
        label = aliases.get(normalized_label, normalized_label)
        image_key = relative.as_posix()
        records.append(
            {
                "sample_id": hashlib.sha256(
                    f"{source['id']}\0{image_key}".encode("utf-8")
                ).hexdigest(),
                "image_uri": str(path.resolve()),
                "image_key": image_key,
                "label": label,
                "source_id": source["id"],
                "source_role": source["role"],
                "split": None,
                "fold": None,
                "group_id": None,
                "content_hash": _file_sha256(path, hash_cache),
                "metadata": {},
            }
        )
    if not records:
        raise DataError(f"source {source['id']}: no supported images below {root}")
    return records


def _deduplicate(
    records: list[dict[str, Any]], policy: str, dataset_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sample_ids: dict[str, int] = {}
    identities: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        sample_id = record["sample_id"]
        identity = (record["source_id"], record["image_key"])
        if sample_id in sample_ids:
            raise DataError(
                f"dataset {dataset_id}: duplicate sample_id {sample_id!r} "
                f"at records {sample_ids[sample_id]} and {index}"
            )
        if identity in identities:
            raise DataError(
                f"dataset {dataset_id}: duplicate source_id/image_key {identity!r}"
            )
        sample_ids[sample_id] = index
        identities[identity] = index
    if policy not in {"error", "keep_first", "keep_last"}:
        raise DataError(f"dataset {dataset_id}: invalid duplicate_policy {policy!r}")
    by_hash: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        by_hash.setdefault(record["content_hash"], []).append(index)
    duplicate_groups = [indices for indices in by_hash.values() if len(indices) > 1]
    if duplicate_groups and policy == "error":
        examples = [
            [records[index]["sample_id"] for index in group]
            for group in duplicate_groups[:5]
        ]
        raise DataError(
            f"dataset {dataset_id}: duplicate image content detected: {examples}"
        )
    dropped: set[int] = set()
    report: list[dict[str, Any]] = []
    for group in duplicate_groups:
        keep = group[0] if policy == "keep_first" else group[-1]
        for index in group:
            if index == keep:
                continue
            dropped.add(index)
            report.append(
                {
                    "reason": "content_hash",
                    "dropped_sample_id": records[index]["sample_id"],
                    "kept_sample_id": records[keep]["sample_id"],
                    "content_hash": records[index]["content_hash"],
                }
            )
    return [
        record for index, record in enumerate(records) if index not in dropped
    ], report


def _stratified_counts(size: int, ratios: list[tuple[str, float]]) -> dict[str, int]:
    raw = [(name, size * ratio) for name, ratio in ratios]
    result = {name: math.floor(value) for name, value in raw}
    remaining = size - sum(result.values())
    order = sorted(
        range(len(raw)),
        key=lambda index: (-(raw[index][1] - math.floor(raw[index][1])), index),
    )
    for index in order[:remaining]:
        result[raw[index][0]] += 1
    return result


def _rank(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{sample_id}".encode("utf-8")).hexdigest()


def _derived_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256(
        "\0".join((str(base_seed), *parts)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def _apply_split(
    frame: pd.DataFrame,
    config: dict[str, Any],
    dataset_id: str,
    base_seed: int,
) -> None:
    strategy = config["strategy"]
    sample_mask = frame["source_role"].eq("sample")
    if strategy == "predefined":
        mapping = config.get("mapping")
        if not isinstance(mapping, dict) or not mapping:
            raise DataError(f"dataset {dataset_id}: predefined split needs mapping")
        invalid_roles = sorted(
            {str(value) for value in mapping.values()} - {"train", "validation", "test"}
        )
        if invalid_roles:
            raise DataError(
                f"dataset {dataset_id}: invalid predefined roles {invalid_roles}"
            )
        for index in frame.index[sample_mask]:
            raw = frame.at[index, "split"]
            if raw is None or pd.isna(raw):
                raise DataError(
                    f"dataset {dataset_id}: sample {frame.at[index, 'sample_id']} has no split"
                )
            key = str(raw)
            if key not in mapping:
                raise DataError(
                    f"dataset {dataset_id}: unknown predefined split {key!r}"
                )
            frame.at[index, "split"] = mapping[key]
        return
    if strategy == "fold_column":
        validation = {str(value) for value in config.get("validation_folds", [])}
        test = {str(value) for value in config.get("test_folds", [])}
        if validation & test:
            raise DataError(
                f"dataset {dataset_id}: validation_folds and test_folds overlap"
            )
        remaining = config.get("remaining", "train")
        if remaining not in {"train", "validation", "test"}:
            raise DataError(
                f"dataset {dataset_id}: invalid remaining fold role {remaining!r}"
            )
        for index in frame.index[sample_mask]:
            fold = frame.at[index, "fold"]
            if fold is None or pd.isna(fold):
                raise DataError(
                    f"dataset {dataset_id}: sample {frame.at[index, 'sample_id']} has no fold"
                )
            key = str(fold)
            frame.at[index, "split"] = (
                "validation"
                if key in validation
                else "test"
                if key in test
                else remaining
            )
        return
    ratios_config = config.get("ratios")
    if not isinstance(ratios_config, dict) or not ratios_config:
        raise DataError(f"dataset {dataset_id}: stratified_hash needs ratios")
    ratios = [(str(name), float(value)) for name, value in ratios_config.items()]
    invalid_roles = sorted(
        {name for name, _ in ratios} - {"train", "validation", "test"}
    )
    if invalid_roles:
        raise DataError(
            f"dataset {dataset_id}: invalid stratified split roles {invalid_roles}"
        )
    if any(value < 0 for _, value in ratios) or not math.isclose(
        sum(value for _, value in ratios), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise DataError(f"dataset {dataset_id}: split ratios must sum to 1")
    if frame.loc[sample_mask, "group_id"].notna().any():
        raise DataError(
            f"dataset {dataset_id}: stratified_hash is forbidden when group_id exists"
        )
    seed = int(config.get("seed", _derived_seed(base_seed, dataset_id, "split")))
    for _, group in frame.loc[sample_mask].groupby("label", sort=True):
        ordered = sorted(
            group.index, key=lambda index: _rank(seed, frame.at[index, "sample_id"])
        )
        counts = _stratified_counts(len(ordered), ratios)
        offset = 0
        for name, _ in ratios:
            for index in ordered[offset : offset + counts[name]]:
                frame.at[index, "split"] = name
            offset += counts[name]


def _filter_view(
    frame: pd.DataFrame, filters: dict[str, Any], context: str
) -> pd.DataFrame:
    result = frame
    for field, expected in filters.items():
        if field not in FILTER_FIELDS:
            raise DataError(f"{context}: unsupported filter field {field!r}")
        values = expected if isinstance(expected, list) else [expected]
        result = result[result[field].isin(values)]
    return result.copy()


def _sample_view(
    frame: pd.DataFrame,
    sampling: dict[str, Any] | None,
    base_seed: int,
    context: str,
) -> pd.DataFrame:
    if not sampling or sampling.get("type", "all") == "all":
        return frame.copy()
    unknown = sorted(set(sampling) - {"type", "count", "seed"})
    if unknown:
        raise DataError(f"{context}: unknown sampling fields: {unknown}")
    if sampling.get("type") != "per_label":
        raise DataError(f"{context}: sampling.type must be all or per_label")
    count = sampling.get("count")
    if not isinstance(count, int) or count <= 0:
        raise DataError(f"{context}: per_label sampling count must be positive")
    seed = int(sampling.get("seed", _derived_seed(base_seed, context, "sampling")))
    selected: list[int] = []
    for label, group in frame.groupby("label", sort=True):
        if len(group) < count:
            raise DataError(
                f"{context}: label {label!r} has {len(group)} samples, needs {count}"
            )
        selected.extend(
            sorted(
                group.index,
                key=lambda index: _rank(seed, frame.at[index, "sample_id"]),
            )[:count]
        )
    return frame.loc[selected].copy()


def _build_views(
    frame: pd.DataFrame,
    configs: list[dict[str, Any]],
    base_seed: int,
    dataset_id: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, str], dict[str, str]]:
    pending = {config["id"]: config for config in configs}
    views: dict[str, pd.DataFrame] = {}
    usages: dict[str, str] = {}
    parents: dict[str, str] = {}
    while pending:
        progressed = False
        for view_id, config in list(pending.items()):
            parent = config["from"]
            if parent != "all" and parent not in views:
                continue
            source = frame if parent == "all" else views[parent]
            context = f"dataset {dataset_id} view {view_id}"
            filtered = _filter_view(source, config.get("filters", {}), context)
            sampled = _sample_view(filtered, config.get("sampling"), base_seed, context)
            if sampled.empty:
                raise DataError(f"{context}: view is empty")
            views[view_id] = sampled.reset_index(drop=True)
            usages[view_id] = config["usage"]
            parents[view_id] = parent
            pending.pop(view_id)
            progressed = True
        if not progressed:
            unresolved = {
                view_id: config["from"] for view_id, config in pending.items()
            }
            raise DataError(
                f"dataset {dataset_id}: view reference cycle or missing parent: {unresolved}"
            )
    return views, usages, parents


def _build_label_map(
    config: dict[str, Any],
    views: dict[str, pd.DataFrame],
    aliases: dict[str, str],
    dataset_id: str,
) -> dict[str, int]:
    reference = config["reference_view"]
    if reference not in views:
        raise DataError(
            f"dataset {dataset_id}: label reference view {reference!r} does not exist"
        )
    if config["mode"] == "explicit":
        labels = [
            aliases.get(
                _normalize_label(value, f"dataset {dataset_id} explicit label"), value
            )
            for value in config["values"]
        ]
    else:
        labels = sorted(set(views[reference]["label"].astype(str)))
    labels = [
        _normalize_label(label, f"dataset {dataset_id} label map") for label in labels
    ]
    if len(labels) != len(set(labels)):
        raise DataError(f"dataset {dataset_id}: label map contains duplicates")
    if not labels:
        raise DataError(f"dataset {dataset_id}: label map is empty")
    return {label: index for index, label in enumerate(labels)}


def _validate_label_space(
    views: dict[str, pd.DataFrame],
    usages: dict[str, str],
    label_map: dict[str, int],
    dataset_id: str,
) -> None:
    known = set(label_map)
    unknown_labels: set[str] = set()
    for view_id, frame in views.items():
        labels = set(frame["label"].astype(str))
        usage = usages[view_id]
        if usage == "unknown_test":
            unknown_labels |= labels
        elif usage in {"train", "validation", "test", "prototype"}:
            extra = sorted(labels - known)
            if extra:
                raise DataError(
                    f"dataset {dataset_id} view {view_id}: labels outside known space: {extra}"
                )
            if usage in {"train", "test", "prototype"}:
                missing = sorted(known - labels)
                if missing:
                    raise DataError(
                        f"dataset {dataset_id} view {view_id}: missing known labels: {missing}"
                    )
    overlap = sorted(known & unknown_labels)
    if overlap:
        raise DataError(
            f"dataset {dataset_id}: known and unknown label spaces overlap: {overlap}"
        )


def _dataset_hash(frame: pd.DataFrame) -> str:
    fields = (
        "sample_id",
        "content_hash",
        "label",
        "split",
        "group_id",
        "source_role",
    )
    rows = [
        {field: None if pd.isna(row[field]) else row[field] for field in fields}
        for _, row in frame.sort_values("sample_id").iterrows()
    ]
    return sha256_json(rows)


def _view_hash(frame: pd.DataFrame, sampling: dict[str, Any] | None) -> str:
    rows = [
        {"sample_id": row.sample_id, "label": row.label}
        for row in frame.sort_values("sample_id").itertuples()
    ]
    return sha256_json({"samples": rows, "sampling": sampling or {"type": "all"}})


def materialize_dataset(config: dict[str, Any], base_seed: int) -> MaterializedDataset:
    dataset_id = config["id"]
    aliases = _resolved_aliases(config["labels"].get("aliases", {}), dataset_id)
    hash_cache: dict[Path, str] = {}
    records: list[dict[str, Any]] = []
    for source in config["sources"]:
        adapter = (
            _folder_records if source["type"] == "folder_label" else _table_records
        )
        records.extend(adapter(source, aliases, hash_cache))
    records, duplicates = _deduplicate(
        records, config.get("duplicate_policy", "error"), dataset_id
    )
    frame = pd.DataFrame(records, columns=CANONICAL_COLUMNS)
    _apply_split(frame, config["split"], dataset_id, base_seed)
    views, usages, parents = _build_views(frame, config["views"], base_seed, dataset_id)
    label_map = _build_label_map(config["labels"], views, aliases, dataset_id)
    _validate_label_space(views, usages, label_map, dataset_id)
    view_config = {item["id"]: item for item in config["views"]}
    hashes = {
        "dataset": _dataset_hash(frame),
        "split": sha256_json(
            [
                {
                    "sample_id": row.sample_id,
                    "split": None if pd.isna(row.split) else row.split,
                }
                for row in frame.sort_values("sample_id").itertuples()
            ]
        ),
        "label_map": sha256_json(
            [{"index": index, "label": label} for label, index in label_map.items()]
        ),
    }
    hashes.update(
        {
            f"view:{view_id}": _view_hash(view, view_config[view_id].get("sampling"))
            for view_id, view in views.items()
        }
    )
    summary = {
        "dataset_id": dataset_id,
        "samples": len(frame),
        "sources": {
            str(key): int(value)
            for key, value in frame["source_id"].value_counts().sort_index().items()
        },
        "roles": {
            str(key): int(value)
            for key, value in frame["source_role"].value_counts().sort_index().items()
        },
        "splits": {
            str(key): int(value)
            for key, value in frame["split"]
            .fillna("<null>")
            .value_counts()
            .sort_index()
            .items()
        },
        "labels": len(label_map),
        "views": {view_id: len(view) for view_id, view in views.items()},
        "duplicates_removed": len(duplicates),
    }
    return MaterializedDataset(
        dataset_id=dataset_id,
        records=frame,
        views=views,
        view_usages=usages,
        view_parents=parents,
        label_map=label_map,
        hashes=hashes,
        summary=summary,
        duplicates=duplicates,
    )


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DependencyError(
            "Parquet artifacts require pyarrow; install requirements.txt"
        ) from exc
    arrays = []
    for name in frame.columns:
        if name == "metadata":

            def metadata_items(item):
                if item is None:
                    return {}
                if isinstance(item, dict):
                    return item
                if isinstance(item, (list, tuple)):
                    return dict(item)
                raise DataError(
                    f"metadata must be an object, got {type(item).__name__}"
                )

            values = [
                {
                    str(key): json.dumps(value, ensure_ascii=False, default=str)
                    for key, value in metadata_items(item).items()
                }
                for item in frame[name]
            ]
            arrays.append(pa.array(values, type=pa.map_(pa.string(), pa.string())))
        elif name in {"split", "fold", "group_id"}:
            arrays.append(
                pa.array(
                    [None if pd.isna(value) else str(value) for value in frame[name]],
                    type=pa.string(),
                )
            )
        else:
            arrays.append(pa.array(frame[name].tolist()))
    pq.write_table(pa.Table.from_arrays(arrays, names=list(frame.columns)), path)


def validate_bound_inputs(
    experiment: dict[str, Any], datasets: dict[str, MaterializedDataset]
) -> None:
    role_groups = {
        "train": "train",
        "validation": "validation",
        "calibration": "validation",
        "test": "test",
        "query": "test",
        "prototype": "prototype",
        "support": "prototype",
        "known": "known",
        "unknown": "unknown",
        "unknown_test": "unknown",
    }
    usage_groups = {
        "train": "train",
        "validation": "validation",
        "test": "test",
        "prototype": "prototype",
        "unknown_test": "unknown",
        "analysis": "analysis",
    }
    bound: list[tuple[str, str, pd.DataFrame]] = []
    for stage in experiment["stages"]:
        for role, reference in stage["inputs"].items():
            if not isinstance(reference, str) or "." not in reference:
                raise DataError(
                    f"stage {stage['id']} input {role}: expected <dataset>.<view>"
                )
            alias, view_id = reference.split(".", 1)
            if alias not in datasets:
                continue
            if view_id not in datasets[alias].views:
                raise DataError(
                    f"stage {stage['id']} input {role}: unknown binding {reference!r}"
                )
            usage = datasets[alias].view_usages[view_id]
            expected = role_groups.get(role)
            if (
                expected in {"train", "validation", "test", "prototype"}
                and usage != expected
            ):
                raise DataError(
                    f"stage {stage['id']} input {role}: view usage is {usage!r}, "
                    f"expected {expected!r}"
                )
            if expected == "unknown" and usage != "unknown_test":
                raise DataError(
                    f"stage {stage['id']} input {role}: unknown input must use "
                    "an unknown_test view"
                )
            if expected == "known" and usage not in {"validation", "test"}:
                raise DataError(
                    f"stage {stage['id']} input {role}: known input must use "
                    "a validation or test view"
                )
            bound.append(
                (
                    role_groups.get(role, usage_groups[usage]),
                    reference,
                    datasets[alias].views[view_id],
                )
            )
    incompatible = {
        frozenset(("train", "validation")),
        frozenset(("train", "test")),
        frozenset(("validation", "test")),
        frozenset(("prototype", "test")),
        frozenset(("prototype", "unknown")),
        frozenset(("known", "unknown")),
        frozenset(("test", "unknown")),
        frozenset(("validation", "unknown")),
        frozenset(("train", "unknown")),
    }
    for index, (left_role, left_ref, left) in enumerate(bound):
        for right_role, right_ref, right in bound[index + 1 :]:
            if frozenset((left_role, right_role)) not in incompatible:
                continue
            sample_overlap = set(left["sample_id"]) & set(right["sample_id"])
            content_overlap = set(left["content_hash"]) & set(right["content_hash"])
            left_groups = set(left["group_id"].dropna().astype(str))
            right_groups = set(right["group_id"].dropna().astype(str))
            group_overlap = left_groups & right_groups
            if sample_overlap or content_overlap or group_overlap:
                raise DataError(
                    f"input leakage {left_ref} ({left_role}) ↔ {right_ref} ({right_role}): "
                    f"sample={len(sample_overlap)}, content={len(content_overlap)}, "
                    f"group={len(group_overlap)}"
                )
            if "unknown" in {left_role, right_role} and (
                {left_role, right_role} & {"known", "train", "test", "validation"}
            ):
                label_overlap = set(left["label"]) & set(right["label"])
                if label_overlap:
                    raise DataError(
                        f"known/unknown label overlap between {left_ref} and {right_ref}: "
                        f"{sorted(label_overlap)}"
                    )
