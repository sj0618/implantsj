from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.experiment_state import load_state, save_state


def test_experiment_state_round_trip_is_json_not_shell(tmp_path: Path) -> None:
    path = tmp_path / ".exp_state"
    values = {
        "DATASET": "data set; touch should-not-run",
        "MODEL": "vit$(echo unsafe)",
        "CHECKPOINT": "/tmp/check point.pt",
    }
    save_state(path, values)
    assert load_state(path) == values
    assert json.loads(path.read_text(encoding="utf-8")) == values


def test_experiment_state_rejects_legacy_shell_code(tmp_path: Path) -> None:
    path = tmp_path / ".exp_state"
    path.write_text("DATASET=$(touch unsafe)\n", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_state(path)
