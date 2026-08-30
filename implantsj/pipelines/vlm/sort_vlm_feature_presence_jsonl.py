#!/usr/bin/env python3
"""Sort VLM feature-presence JSONL back to dataset discovery order.

The generator can write JSONL records in completion order when --workers > 1.
This script rebuilds the original discovery order from --input, adds
original_idx to each record, and rewrites the JSONL in that order.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from generate_vlm_feature_presence import discover_images


def generator_is_running(output_path: Path) -> bool:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,cmd="],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return False
    needle = str(output_path)
    for line in result.stdout.splitlines():
        if "generate_vlm_feature_presence.py" in line and needle in line:
            return True
    return False


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sort VLM JSONL by original dataset order.")
    parser.add_argument("--input", default="data/7label", help="Original image input used by the generator.")
    parser.add_argument("--jsonl", default="data/manifests/vlm_feature_presence_7label_all.jsonl")
    parser.add_argument("--path-column", default="path")
    parser.add_argument("--in-place", action="store_true", default=True)
    parser.add_argument("--output", default=None, help="Optional output path. Defaults to rewriting --jsonl.")
    parser.add_argument("--backup", action="store_true", default=True)
    parser.add_argument("--no-backup", dest="backup", action="store_false")
    parser.add_argument("--force", action="store_true", help="Allow sorting even if generator appears to be running.")
    args = parser.parse_args()

    input_path = Path(args.input)
    jsonl_path = Path(args.jsonl)
    output_path = Path(args.output) if args.output else jsonl_path

    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)
    if generator_is_running(jsonl_path) and not args.force:
        raise RuntimeError(
            f"Generator still appears to be writing {jsonl_path}. "
            "Wait for it to finish, or pass --force if you are certain it is stopped."
        )

    reference = discover_images(input_path, args.path_column)
    order = {str(row["path"]): idx for idx, row in enumerate(reference)}
    unknown_base = len(order) + 1_000_000

    rows = load_jsonl(jsonl_path)
    missing = 0
    for seq, row in enumerate(rows):
        idx = order.get(str(row.get("path", "")))
        if idx is None:
            idx = unknown_base + seq
            missing += 1
        row["original_idx"] = idx

    rows.sort(key=lambda row: (int(row.get("original_idx", unknown_base)), str(row.get("path", ""))))

    backup_path = None
    if args.backup and output_path == jsonl_path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = jsonl_path.with_suffix(jsonl_path.suffix + f".bak_{stamp}")
        shutil.copy2(jsonl_path, backup_path)

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    write_jsonl(tmp_path, rows)
    tmp_path.replace(output_path)

    print(f"sorted_rows={len(rows)} reference_rows={len(reference)} missing_paths={missing}")
    print(f"wrote={output_path}")
    if backup_path:
        print(f"backup={backup_path}")


if __name__ == "__main__":
    main()
