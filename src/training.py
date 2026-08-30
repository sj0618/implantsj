"""Train one configurable experiment from explicit command-line arguments."""

from __future__ import annotations

import argparse
import os
from typing import Sequence

from .experiment_data import DATA_MODULES
from .experiment import Experiment, ExperimentConfig
from .losses import LOSS_STRATEGIES
from .optimization import OPTIMIZERS, SCHEDULERS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name")
    parser.add_argument("--data", choices=sorted(DATA_MODULES), default="manifest")
    parser.add_argument(
        "--model",
        required=True,
        help="timm model architecture, for example vit_base_patch16_224",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use timm pretrained weights when --checkpoint is omitted",
    )
    parser.add_argument("--checkpoint", help="initial ckpt or safetensors weights")
    parser.add_argument(
        "--checkpoint-format",
        choices=["auto", "ckpt", "safetensors"],
        default="auto",
        help="input checkpoint format",
    )
    parser.add_argument(
        "--train-data", "--train-manifest", dest="train_data", required=True
    )
    parser.add_argument(
        "--valid-data", "--valid-manifest", dest="valid_data", required=True
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scope", choices=["closed", "all"], default="all")
    parser.add_argument("--loss", choices=sorted(LOSS_STRATEGIES), default="ce")
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="adamw")
    parser.add_argument("--scheduler", choices=SCHEDULERS, default="cosine")
    parser.add_argument(
        "--save-format", choices=["ckpt", "safetensors"], default="safetensors"
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler-step-size", type=int, default=10)
    parser.add_argument("--scheduler-gamma", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--triplet-margin", type=float, default=0.2)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--projection-hidden-dim", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    return parser


def train(args: argparse.Namespace):
    return Experiment(ExperimentConfig.from_namespace(args)).run()


def main(argv: Sequence[str] | None = None) -> int:
    output = train(build_parser().parse_args(argv))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
