import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "legacy/ai_hub_pipeline_20260908"


def test_restored_pipeline_source_coverage():
    assert len(list((ROOT / "src").rglob("*.py"))) == 65
    assert len(list((ROOT / "configs").glob("*.yaml"))) == 11
    scripts = list((ROOT / "scripts").rglob("*.sh"))
    assert len(scripts) == 16
    for path in scripts:
        assert path.stat().st_mode & 0o111, path
        subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True, text=True)


def test_restored_pipeline_wiring_and_options():
    result = subprocess.run(
        [sys.executable, "src/verify_pipeline.py"], cwd=ROOT,
        check=True, capture_output=True, text=True, timeout=90,
    )
    assert "WARN 0   FAIL 0" in result.stdout


def test_archived_docker_copy_sources_exist():
    for line in (ROOT / "Dockerfile").read_text().splitlines():
        if line.startswith("COPY "):
            for source in line.split()[1:-1]:
                assert (ROOT / source).exists(), source


def test_archived_oas_and_five_centers_are_not_five_shot():
    code = '''
import sys
sys.path.insert(0, 'src')
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.covariance import OAS
from metric.prototypes import build_prototypes
from metric.scoring import build_scorer
rng = np.random.default_rng(42)
x = np.vstack([rng.normal(i * 3, .5, (12, 4)) for i in range(2)])
y = np.repeat(['a', 'b'], 12)
classes = ['a', 'b']
p = build_prototypes(x, y, classes, 'mean')
s = build_scorer('mahalanobis', {'mahalanobis': {'covariance': 'oas'}}, 42).fit(x, y, classes)
centers = np.vstack([p[c] for c in classes])
precision = OAS().fit(x - centers[np.repeat([0, 1], 12)]).precision_
actual = s.score(x[:3], p)
expected = cdist(x[:3], centers, metric='mahalanobis', VI=precision)
assert np.allclose(actual, expected, atol=1e-8), (actual, expected)
multi = build_prototypes(x, y, classes, 'kmeans_multi', kmeans_clusters=5)
assert all(v.shape == (5, 4) for v in multi.values())
assert len(x) == 24
'''
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True, text=True)
