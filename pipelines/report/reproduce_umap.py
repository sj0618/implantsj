"""Rebuild UMAP artifacts in a new directory, preserving the original inputs."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from tools.reproduce_reports import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=9)
    args = parser.parse_args()
    sources = sorted(args.source.rglob("tsne_*.npz"))
    if len(sources) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} t-SNE feature archives, found {len(sources)}")
    for source in sources:
        if not source.with_suffix(".csv").is_file():
            raise FileNotFoundError(source.with_suffix(".csv"))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from pipelines.visualize.visualize_umap_from_tsne import process_artifact

    records = []
    for source in sources:
        target = args.output_dir / source.relative_to(args.source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        shutil.copyfile(source.with_suffix(".csv"), target.with_suffix(".csv"))
        process_artifact(target, seed=42, pca_dim=50, n_neighbors=15, min_dist=0.1, overwrite=False)
        records.append({"source": str(source), "sha256": digest(source),
                        "metadata_sha256": digest(source.with_suffix(".csv")), "copy": str(target)})
    (args.output_dir / "index.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
