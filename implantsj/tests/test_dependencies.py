from __future__ import annotations

import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.setup.dependencies import BASE_PACKAGES, IMPORT_CHECKS, PACKAGE_IMPORTS


def test_dependency_lists_are_deduplicated_and_clean() -> None:
    assert BASE_PACKAGES == list(PACKAGE_IMPORTS)
    assert IMPORT_CHECKS == ['torch', 'torchvision', *PACKAGE_IMPORTS.values(), 'src']
    assert {'plotly', 'pyyaml', 'tqdm', 'ultralytics'} <= set(BASE_PACKAGES)
    assert {'plotly', 'yaml', 'tqdm', 'ultralytics'} <= set(IMPORT_CHECKS)
    assert len(BASE_PACKAGES) == len(set(BASE_PACKAGES))
    assert len(IMPORT_CHECKS) == len(set(IMPORT_CHECKS))
    requirements = {
        re.split(r"[<>=!~]", line, maxsplit=1)[0].strip().casefold()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert requirements == {name.casefold() for name in BASE_PACKAGES}
