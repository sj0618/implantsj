from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path('/workspace/data')
DEFAULT_OUTPUT_ROOT = Path('/workspace/implant_outputs')
DEFAULT_LARGE_DATASET = DEFAULT_DATA_ROOT / 'large_multiclass'
DEFAULT_MANIFEST_DIR = DEFAULT_DATA_ROOT / 'manifests'
DEFAULT_LARGE_MANIFEST = DEFAULT_MANIFEST_DIR / 'large_multiclass.csv'

