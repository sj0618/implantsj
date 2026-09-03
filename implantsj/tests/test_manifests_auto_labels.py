#!/usr/bin/env python3
"""--closed-labels auto 가 데이터에 있는 클래스를 그대로 쓰는지 확인한다.

src/manifests.py 는 원래 42 라벨(closed 27 + open 15) 계약을 강제한다. 그래서
클래스 수가 다른 데이터셋에는 쓸 수 없었다. auto 는 그 목록을 데이터에서 읽는다.
여기서는 클래스 5개짜리 데이터를 만들어 두 경우를 비교한다.

  auto 없이  -> 없는 closed 라벨을 요구하며 실패해야 한다
  auto 주면  -> 5개를 그대로 읽어 manifest 를 만들어야 한다
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
N_CLASSES = 5
PER_CLASS = 4


def make_imagefolder(root: Path, tag: str) -> None:
    for i in range(N_CLASSES):
        d = root / f"Class{chr(65 + i)}"
        d.mkdir(parents=True, exist_ok=True)
        for k in range(PER_CLASS):
            Image.new("RGB", (12, 12), (20 + i * 40, 60, 90)).save(d / f"{tag}_{i}_{k}.jpg")


def run(train: Path, test: Path, out: Path, auto: bool) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "src.manifests",
           "--train-root", str(train), "--test-root", str(test),
           "--output-dir", str(out)]
    if auto:
        cmd += ["--closed-labels", "auto"]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="manifests_auto_") as tmp:
        tmp_dir = Path(tmp)
        train, test = tmp_dir / "train", tmp_dir / "test"
        make_imagefolder(train, "tr")
        make_imagefolder(test, "te")

        without = run(train, test, tmp_dir / "out_without", auto=False)
        assert without.returncode != 0, (
            "auto 없이 5개 클래스 데이터가 통과하면 안 된다. 42-라벨 계약이 사라졌다."
        )

        with_auto = run(train, test, tmp_dir / "out_auto", auto=True)
        assert with_auto.returncode == 0, (
            f"auto 인데 실패했다.\nstdout:\n{with_auto.stdout}\nstderr:\n{with_auto.stderr}"
        )
        assert f"클래스 {N_CLASSES}개" in with_auto.stdout, with_auto.stdout

        # 만들어진 manifest 의 실제 클래스 수가 데이터와 같아야 한다.
        csvs = sorted((tmp_dir / "out_auto").rglob("*.csv"))
        assert csvs, "manifest CSV 가 하나도 없다"
        labels: set[str] = set()
        for c in csvs:
            df = pd.read_csv(c)
            if "label" in df.columns:
                labels |= set(df["label"].astype(str))
        assert len(labels) == N_CLASSES, f"기대 {N_CLASSES}개, 실제 {len(labels)}개: {sorted(labels)}"

    print(f"[test_manifests_auto_labels] OK  auto 로 클래스 {N_CLASSES}개 인식")


if __name__ == "__main__":
    main()
