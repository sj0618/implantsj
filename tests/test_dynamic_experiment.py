from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import (  # noqa: E402
    EmbeddingTable,
    build_prototypes,
    build_parser as build_evaluation_parser,
    classification_metrics,
    evaluate,
    predict_42way,
)
from src.experiment_data import PKBatchSampler  # noqa: E402
from src.manifests import CLOSED_LABELS, build_manifests, build_parser  # noqa: E402
from src.model_adapters import (  # noqa: E402
    ModelAdapter,
    ModelSpec,
    checkpoint_payload,
    create_adapter,
    load_initial_weights,
    register_adapter,
)
from src.training import build_parser as build_training_parser  # noqa: E402


OPEN_LABELS = [f"Open {index:02d}" for index in range(1, 16)]
ALL_LABELS = list(CLOSED_LABELS) + OPEN_LABELS


def embedding_table(
    features: np.ndarray, labels: list[str], prefix: str
) -> EmbeddingTable:
    count = len(labels)
    return EmbeddingTable(
        features=np.asarray(features, dtype=np.float32),
        labels=np.asarray(labels, dtype=str),
        paths=np.asarray(
            [f"/{prefix}/{index}.png" for index in range(count)], dtype=str
        ),
        splits=np.full(count, prefix, dtype=str),
        sets=np.asarray(
            ["closed" if label in CLOSED_LABELS else "open" for label in labels],
            dtype=str,
        ),
        metadata={"prefix": prefix},
    )


class ManifestContractTests(unittest.TestCase):
    def test_closed_label_config_matches_code(self):
        configured = (ROOT / "configs" / "closed_labels.txt").read_text().splitlines()
        self.assertEqual(configured, list(CLOSED_LABELS))

    def test_fixed_27_open_complement_and_test_preserved(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            rows = []
            expected_test = set()
            for label in ALL_LABELS:
                for index in range(4):
                    rows.append(
                        {
                            "path": f"images/{label}/train_{index}.png",
                            "label": label,
                            "split": "train",
                        }
                    )
                for index in range(2):
                    path = str(
                        (root / "images" / label / f"test_{index}.png").resolve()
                    )
                    expected_test.add(path)
                    rows.append(
                        {
                            "path": f"images/{label}/test_{index}.png",
                            "label": label,
                            "split": "test",
                        }
                    )
            source = root / "source.csv"
            pd.DataFrame(rows).to_csv(source, index=False)
            output = root / "out"
            args = build_parser().parse_args(
                [
                    "--source-manifest",
                    str(source),
                    "--output-dir",
                    str(output),
                    "--valid-fraction",
                    "0.25",
                ]
            )
            build_manifests(args)
            labels = json.loads((output / "labels.json").read_text())
            self.assertEqual(labels["closed"], list(CLOSED_LABELS))
            self.assertEqual(len(labels["open"]), 15)
            self.assertEqual(set(labels["open"]), set(OPEN_LABELS))
            test = pd.read_csv(output / "test.csv")
            train = pd.read_csv(output / "train.csv")
            valid = pd.read_csv(output / "valid.csv")
            self.assertEqual(set(test.path), expected_test)
            self.assertTrue(
                set(test.path).isdisjoint(set(train.path) | set(valid.path))
            )
            self.assertEqual(set(valid.origin_split), {"train"})
            self.assertEqual(len(set(valid.label)), 42)

    def test_duplicate_paths_are_rejected_before_splitting(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.csv"
            pd.DataFrame(
                [
                    {"path": "same.png", "label": "IS I", "split": "train"},
                    {"path": "same.png", "label": "IS I", "split": "test"},
                ]
            ).to_csv(source, index=False)
            args = build_parser().parse_args(
                [
                    "--source-manifest",
                    str(source),
                    "--output-dir",
                    str(root / "out"),
                    "--allow-incomplete",
                ]
            )
            with self.assertRaisesRegex(ValueError, "duplicate image paths"):
                build_manifests(args)

    def test_custom_closed_label_contract(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            closed = [f"Closed {index}" for index in range(7)]
            opened = [f"Open {index}" for index in range(3)]
            rows = [
                {
                    "path": f"images/{label}/{split}_{index}.png",
                    "label": label,
                    "split": split,
                }
                for label in closed + opened
                for split, count in (("train", 4), ("test", 2))
                for index in range(count)
            ]
            source = root / "source.csv"
            pd.DataFrame(rows).to_csv(source, index=False)
            closed_path = root / "closed.txt"
            closed_path.write_text("\n".join(closed) + "\n", encoding="utf-8")
            output = root / "out"
            args = build_parser().parse_args(
                [
                    "--source-manifest",
                    str(source),
                    "--output-dir",
                    str(output),
                    "--closed-labels",
                    str(closed_path),
                    "--valid-fraction",
                    "0.25",
                ]
            )
            build_manifests(args)
            labels = json.loads((output / "labels.json").read_text())
            audit = json.loads((output / "audit.json").read_text())
            self.assertEqual(labels["closed"], closed)
            self.assertEqual(set(labels["open"]), set(opened))
            self.assertEqual(labels["contract"], {"total": 10, "closed": 7, "open": 3})
            self.assertFalse(audit["strict_42_class_contract"])


class PrototypeTests(unittest.TestCase):
    def setUp(self):
        features = np.repeat(np.eye(42, dtype=np.float32), 3, axis=0)
        labels = [label for label in ALL_LABELS for _ in range(3)]
        self.table = embedding_table(features, labels, "support")

    def test_all_prototype_types_cover_all_42(self):
        for method in ("gt", "mean", "ss"):
            prototypes, membership = build_prototypes(
                self.table, ALL_LABELS, method, seed=11, ss_count=2
            )
            self.assertEqual(prototypes.shape, (42, 42))
            self.assertEqual(set(membership.label), set(ALL_LABELS))

    def test_ss_is_deterministic(self):
        first, first_members = build_prototypes(
            self.table, ALL_LABELS, "ss", seed=99, ss_count=2
        )
        second, second_members = build_prototypes(
            self.table, ALL_LABELS, "ss", seed=99, ss_count=2
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_members.path.tolist(), second_members.path.tolist())

    def test_missing_prototype_label_is_rejected(self):
        incomplete = embedding_table(
            self.table.features[:-3], self.table.labels[:-3].tolist(), "support"
        )
        with self.assertRaisesRegex(ValueError, "coverage"):
            build_prototypes(incomplete, ALL_LABELS, "mean")

    def test_euclidean_and_cosine_can_choose_different_labels(self):
        prototypes = np.asarray([[100.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        query = np.asarray([[1.0, 0.0]], dtype=np.float32)
        euclidean, _, _ = predict_42way(
            query, prototypes, ["direction", "near"], "euclidean"
        )
        cosine, _, _ = predict_42way(query, prototypes, ["direction", "near"], "cosine")
        self.assertEqual(euclidean.tolist(), ["near"])
        self.assertEqual(cosine.tolist(), ["direction"])

    def test_group_metrics_slice_the_same_predictions(self):
        truth = np.asarray(ALL_LABELS, dtype=str)
        predictions = truth.copy()
        predictions[0] = OPEN_LABELS[0]
        all_metrics = classification_metrics(truth, predictions, ALL_LABELS)
        closed_metrics = classification_metrics(truth, predictions, CLOSED_LABELS)
        open_metrics = classification_metrics(truth, predictions, OPEN_LABELS)
        self.assertEqual(all_metrics["n_samples"], 42)
        self.assertEqual(closed_metrics["n_samples"], 27)
        self.assertEqual(open_metrics["n_samples"], 15)
        self.assertAlmostEqual(open_metrics["accuracy"], 1.0)
        self.assertLess(closed_metrics["accuracy"], 1.0)


class FullEvaluationTests(unittest.TestCase):
    def test_one_model_query_produces_18_primary_rows(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            labels_path = root / "labels.json"
            labels_path.write_text(
                json.dumps(
                    {
                        "all": ALL_LABELS,
                        "closed": list(CLOSED_LABELS),
                        "open": OPEN_LABELS,
                    }
                )
            )
            base = np.eye(42, dtype=np.float32)
            support = embedding_table(
                np.repeat(base, 2, axis=0),
                [x for x in ALL_LABELS for _ in range(2)],
                "support",
            )
            gt = embedding_table(base, ALL_LABELS, "gt")
            query = embedding_table(base, ALL_LABELS, "test")
            support.save(root / "support.npz")
            gt.save(root / "gt.npz")
            query.save(root / "query.npz")
            args = argparse.Namespace(
                output_dir=str(root / "result"),
                labels_json=str(labels_path),
                support_embeddings=str(root / "support.npz"),
                gt_embeddings=str(root / "gt.npz"),
                query_embeddings=str(root / "query.npz"),
                allow_noncanonical_labels=False,
                allow_incomplete_query=False,
                prototypes="gt,mean,ss",
                distances="euclidean,cosine",
                ss_count=1,
                ss_fraction=None,
                seed=2026,
                model_name="Fake-ViT",
                query_name="test",
            )
            metrics_path = evaluate(args)
            metrics = pd.read_csv(metrics_path)
            self.assertEqual(len(metrics), 18)
            self.assertEqual(set(metrics.group), {"all", "closed", "open"})
            self.assertTrue((metrics.accuracy == 1.0).all())
            predictions = pd.read_csv(root / "result" / "predictions.csv.gz")
            self.assertEqual(len(predictions), 42 * 3 * 2)
            per_class = pd.read_csv(root / "result" / "per_class_metrics.csv")
            self.assertEqual(len(per_class), 42 * 3 * 2)


class SupConSamplerTests(unittest.TestCase):
    def test_each_selected_class_has_distinct_cross_image_positives(self):
        labels = ["A"] * 3 + ["B"] * 3 + ["C"] * 3
        sampler = PKBatchSampler(labels, batch_size=4, samples_per_class=2, seed=7)
        batch = next(iter(sampler))
        selected_labels = [labels[index] for index in batch]
        counts = pd.Series(selected_labels).value_counts().to_dict()
        self.assertEqual(sorted(counts.values()), [2, 2])
        self.assertEqual(len(batch), len(set(batch)))


class InitialCheckpointTests(unittest.TestCase):
    def test_safetensors_sidecar_metadata_is_loaded(self):
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classifier.safetensors"
            save_file(torch.nn.Linear(3, 2).state_dict(), str(path))
            path.with_suffix(".json").write_text(
                json.dumps({"num_classes": 2, "labels": ["A", "B"]})
            )
            payload = checkpoint_payload(path, checkpoint_format="safetensors")
            self.assertEqual(payload["num_classes"], 2)
            self.assertEqual(payload["labels"], ["A", "B"])
            self.assertIn("state_dict", payload)

    def test_training_initialization_loads_safetensors(self):
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "initial.safetensors"
            source = torch.nn.Linear(3, 2)
            with torch.no_grad():
                source.weight.fill_(2.0)
                source.bias.fill_(3.0)
            save_file(source.state_dict(), str(path))
            target = torch.nn.Linear(3, 2)
            spec = ModelSpec(
                name="tiny",
                initial_checkpoint=str(path),
                checkpoint_format="safetensors",
            )
            load_initial_weights(target, spec, "ce")
            self.assertTrue(torch.equal(target.weight, source.weight))
            self.assertTrue(torch.equal(target.bias, source.bias))

    def test_classifier_safetensors_uses_explicit_model_metadata(self):
        import timm
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classifier.safetensors"
            source = timm.create_model("resnet10t", pretrained=False, num_classes=2)
            save_file(
                {
                    key: value.detach().contiguous()
                    for key, value in source.state_dict().items()
                },
                str(path),
            )
            spec = ModelSpec(
                name="tiny-classifier",
                adapter="classifier_checkpoint",
                architecture="resnet10t",
                checkpoint=str(path),
                checkpoint_format="safetensors",
                extra={"num_classes": 2, "labels": ["A", "B"]},
            )
            adapter = create_adapter(spec, "cpu").load()
            self.assertEqual(adapter.checkpoint_metadata["labels"], ["A", "B"])


class AdapterSwapTests(unittest.TestCase):
    def test_custom_adapter_is_registry_swappable(self):
        class FakeAdapter(ModelAdapter):
            def load(self):
                self.model = object()
                return self

            def encode_batch(self, images):
                return images

        register_adapter("unit_fake", FakeAdapter, replace=True)
        spec = ModelSpec(
            name="anything", adapter="unit_fake", architecture="not-hard-coded"
        )
        adapter = create_adapter(spec, "cpu")
        self.assertIsInstance(adapter, FakeAdapter)


class DynamicTrainingCliTests(unittest.TestCase):
    def test_components_are_selected_by_explicit_arguments(self):
        args = build_training_parser().parse_args(
            [
                "--experiment-name",
                "vit_triplet_sgd",
                "--model",
                "vit_small_patch16_224",
                "--checkpoint",
                "/tmp/start.safetensors",
                "--checkpoint-format",
                "safetensors",
                "--train-data",
                "/tmp/train.csv",
                "--valid-data",
                "/tmp/valid.csv",
                "--output-dir",
                "/tmp/run",
                "--loss",
                "triplet",
                "--optimizer",
                "sgd",
                "--scheduler",
                "step",
                "--save-format",
                "ckpt",
            ]
        )
        self.assertEqual(args.model, "vit_small_patch16_224")
        self.assertEqual(args.loss, "triplet")
        self.assertEqual(args.optimizer, "sgd")
        self.assertEqual(args.scheduler, "step")
        self.assertEqual(args.save_format, "ckpt")
        self.assertEqual(args.epochs, 15)


class EntryPointTests(unittest.TestCase):
    def test_root_dispatcher_runs_src_training(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "run.py"), "experiment", "--help"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertIn("--model MODEL", result.stdout)
        self.assertIn("--loss", result.stdout)

    def test_root_dispatcher_exposes_tsne_output_dir(self):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "run.py"),
                "visualize",
                "tsne",
                "--help",
            ],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertIn("--output-dir OUTPUT_DIR", result.stdout)

    def test_embedding_cli_accepts_explicit_checkpoint(self):
        args = build_evaluation_parser().parse_args(
            [
                "embed",
                "--model",
                "vit_base_patch16_224",
                "--adapter",
                "classifier_checkpoint",
                "--checkpoint",
                "/tmp/model.safetensors",
                "--checkpoint-format",
                "safetensors",
                "--manifest",
                "/tmp/test.csv",
                "--output",
                "/tmp/test.npz",
            ]
        )
        self.assertEqual(args.checkpoint, "/tmp/model.safetensors")
        self.assertEqual(args.checkpoint_format, "safetensors")

    def test_metric_diagnosis_package_is_removed(self):
        self.assertFalse((ROOT / "src" / "metric_diagnosis").exists())
        self.assertFalse((ROOT / "metric_diagnosis").exists())


if __name__ == "__main__":
    unittest.main()
