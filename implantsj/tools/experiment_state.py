from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


FIELDS = ("DATASET", "MODEL", "CHECKPOINT")


def load_state(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {field: "" for field in FIELDS}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment state must be a JSON object")
    return {field: str(payload.get(field, "")) for field in FIELDS}


def save_state(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {field: str(values.get(field, "")) for field in FIELDS}
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def emit_shell_records(values: dict[str, str]) -> None:
    for field in FIELDS:
        sys.stdout.buffer.write(field.encode("utf-8") + b"\0")
        sys.stdout.buffer.write(values[field].encode("utf-8") + b"\0")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    load_parser = subparsers.add_parser("load")
    load_parser.add_argument("path", type=Path)
    save_parser = subparsers.add_parser("save")
    save_parser.add_argument("path", type=Path)
    save_parser.add_argument("dataset")
    save_parser.add_argument("model")
    save_parser.add_argument("checkpoint")
    args = parser.parse_args()
    if args.command == "load":
        emit_shell_records(load_state(args.path))
    else:
        save_state(
            args.path,
            {"DATASET": args.dataset, "MODEL": args.model, "CHECKPOINT": args.checkpoint},
        )


if __name__ == "__main__":
    main()
