from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.setup.dependencies import BASE_PACKAGES, IMPORT_CHECKS


def test_dependency_lists_are_deduplicated_and_clean() -> None:
    assert BASE_PACKAGES == [
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'umap-learn',
        'Pillow',
        'timm',
        'openai',
        'python-dotenv',
    ]
    assert IMPORT_CHECKS == [
        'torch',
        'torchvision',
        'timm',
        'numpy',
        'pandas',
        'sklearn',
        'matplotlib',
        'umap',
        'PIL',
        'openai',
        'dotenv',
        'src',
    ]
    assert 'PyYAML' not in BASE_PACKAGES
    assert 'tqdm' not in BASE_PACKAGES
    assert 'yaml' not in IMPORT_CHECKS
    assert 'tqdm' not in IMPORT_CHECKS
    assert len(BASE_PACKAGES) == len(set(BASE_PACKAGES))
    assert len(IMPORT_CHECKS) == len(set(IMPORT_CHECKS))
