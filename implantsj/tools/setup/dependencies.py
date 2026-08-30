from __future__ import annotations

# Single installation/check source for every feature exposed by run.sh.
# torch/torchvision remain separate because their wheel index depends on CUDA.
PACKAGE_IMPORTS = {
    'timm': 'timm',
    'safetensors': 'safetensors',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'scikit-learn': 'sklearn',
    'Pillow': 'PIL',
    'matplotlib': 'matplotlib',
    'umap-learn': 'umap',
    'plotly': 'plotly',
    'pyyaml': 'yaml',
    'tqdm': 'tqdm',
    'ultralytics': 'ultralytics',
    'openai': 'openai',
    'python-dotenv': 'dotenv',
}

BASE_PACKAGES = list(PACKAGE_IMPORTS)

IMPORT_CHECKS = [
    'torch',
    'torchvision',
    *PACKAGE_IMPORTS.values(),
    'src',
]
