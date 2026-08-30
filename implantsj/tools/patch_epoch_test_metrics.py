#!/usr/bin/env python3
"""Compatibility check for the old epoch-test-metrics patch command.

The former script modified training source to evaluate the held-out test split
after every epoch. That behavior leaks test information into model iteration,
so this command is intentionally read-only now.
"""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "src/supervised.py": "train_supervised",
    "src/metric_learning.py": "train_metric",
}


def epoch_loop_source(path: Path, function_name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    loop = next(node for node in function.body if isinstance(node, ast.For))
    return ast.unparse(loop)


def main() -> None:
    violations = []
    for relative, function_name in TARGETS.items():
        source = epoch_loop_source(ROOT / relative, function_name)
        if "'test'" in source or "test_accuracy" in source:
            violations.append(relative)
    if violations:
        raise RuntimeError(f"per-epoch test evaluation remains in {violations}")
    print("PER_EPOCH_TEST_METRICS_DISABLED")


if __name__ == "__main__":
    main()
