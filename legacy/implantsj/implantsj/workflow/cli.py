"""Checkpoint inspection kept separate from explicit experiment scripts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .engine import inspect_checkpoint
from .schema import WorkflowError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="implantsj-workflow")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "inspect-checkpoint", help="inspect checkpoint structure and metadata"
    )
    inspect.add_argument("checkpoint")
    inspect.add_argument("--catalog")
    inspect.add_argument("--id", dest="checkpoint_id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = inspect_checkpoint(
            Path(args.checkpoint),
            Path(args.catalog) if args.catalog else None,
            args.checkpoint_id,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        return 130
    except WorkflowError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 10


if __name__ == "__main__":
    raise SystemExit(main())
