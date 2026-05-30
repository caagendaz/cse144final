"""Command-line entry point for the improved ensemble training run."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from image_ensemble.config import default_colab_config
from image_ensemble.runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kaggle-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/Kaggle"),
        help="Directory containing train/ and test/ folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the directory used for checkpoints and submissions.",
    )
    parser.add_argument("--epochs", type=int, help="Override maximum epochs per fold.")
    parser.add_argument("--folds", type=int, help="Override the number of folds.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use two folds and two epochs as a pipeline smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = default_colab_config(args.kaggle_dir)
    if args.output_dir:
        config = replace(config, output_dir=args.output_dir)
    if args.epochs:
        config = replace(config, final_epochs=args.epochs)
    if args.folds:
        config = replace(config, num_folds=args.folds)
    if args.quick:
        config = replace(config, num_folds=2, final_epochs=2, blend_search_iterations=300)
    run_experiment(config)


if __name__ == "__main__":
    main()
