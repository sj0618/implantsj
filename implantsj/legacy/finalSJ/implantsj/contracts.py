"""Shared label and file contracts."""

from __future__ import annotations

import math
import unicodedata
from pathlib import Path
from typing import Iterable


CLOSED_LABELS: tuple[str, ...] = (
    "IS I",
    "IS II",
    "IS III",
    "EB",
    "Branemark",
    "Astra",
    "Xive",
    "Implantium",
    "Superline",
    "UF",
    "UF II",
    "Any ridge",
    "Anyone internal",
    "Anyone external",
    "Exfeel external",
    "TS standard",
    "TS standard plus",
    "Bone level",
    "Luna",
    "GS II",
    "SS II",
    "TS III",
    "US II",
    "US III",
    "Hexplant",
    "Internal",
    "IT",
)

# Documentation-only grouping; training and evaluation use labels.json instead.
MANUFACTURERS: dict[str, tuple[str, ...]] = {
    "Neobiotech": ("IS I", "IS II", "IS III", "EB"),
    "Nobel Biocare": ("Branemark",),
    "Dentsply Sirona": ("Astra", "Xive"),
    "Dentium": ("Implantium", "Superline"),
    "DIO": ("UF", "UF II"),
    "MegaGen": ("Any ridge", "Anyone internal", "Anyone external", "Exfeel external"),
    "Osstem": (
        "TS standard",
        "TS standard plus",
        "GS II",
        "SS II",
        "TS III",
        "US II",
        "US III",
        "Hexplant",
    ),
    "Straumann": ("Bone level",),
    "Other": ("Luna", "Internal", "IT"),
}


def clean_label(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(value)).strip().split())


def label_key(value: object) -> str:
    return clean_label(value).casefold()


def read_label_list(path: str | Path | None) -> tuple[str, ...]:
    if path is None:
        return CLOSED_LABELS
    labels = tuple(
        clean_label(line)
        for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines()
        if clean_label(line)
    )
    if not labels or len({label_key(label) for label in labels}) != len(labels):
        raise ValueError("closed label file must contain unique, non-empty labels")
    return labels


def validate_partition(
    all_labels: Iterable[str],
    closed: Iterable[str],
    opened: Iterable[str],
    *,
    strict: bool,
) -> None:
    all_list = list(all_labels)
    closed_list = list(closed)
    open_list = list(opened)
    duplicate_groups = {
        name: values
        for name, values in {
            "all": all_list,
            "closed": closed_list,
            "open": open_list,
        }.items()
        if len({label_key(value) for value in values}) != len(values)
    }
    closed_keys = {label_key(value) for value in closed_list}
    open_keys = {label_key(value) for value in open_list}
    all_keys = {label_key(value) for value in all_list}
    missing = sorted(all_keys - closed_keys - open_keys)
    extra = sorted((closed_keys | open_keys) - all_keys)
    overlap = sorted(closed_keys & open_keys)
    count_error = strict and (
        len(all_list) != 42 or len(closed_list) != 27 or len(open_list) != 15
    )
    if duplicate_groups or missing or extra or overlap or count_error:
        raise ValueError(
            "invalid label partition: "
            f"counts={len(all_list)}={len(closed_list)}+{len(open_list)}, "
            f"duplicates={sorted(duplicate_groups)}, missing={missing}, "
            f"extra={extra}, overlap={overlap}"
        )
