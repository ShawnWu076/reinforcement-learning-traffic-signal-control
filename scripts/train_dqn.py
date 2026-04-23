#!/usr/bin/env python3
"""Train a DQN controller and compare it against baseline policies."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_rl.config import apply_overrides, load_config, parse_override_strings
from traffic_rl.experiments import train_and_evaluate_dqn
from traffic_rl.visualization import generate_experiment_plots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/checkpoints/dqn_policy.pt",
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="results/dqn_summary.json",
        help="Path to save training and evaluation metrics.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="results/plots/dqn",
        help="Directory to save training/evaluation plots.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values, e.g. --set training.learning_rate=0.0005",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    args = parser.parse_args()

    overrides = parse_override_strings(args.overrides)
    config = apply_overrides(load_config(PROJECT_ROOT / args.config), overrides)

    checkpoint_path = PROJECT_ROOT / args.checkpoint
    output_path = PROJECT_ROOT / args.summary_output

    summary = train_and_evaluate_dqn(
        config=config,
        checkpoint_path=checkpoint_path,
        summary_path=output_path,
        verbose=True,
    )

    if not args.no_plots:
        try:
            plot_paths = generate_experiment_plots(summary, PROJECT_ROOT / args.plot_dir)
            print("Saved plots:")
            for plot_path in plot_paths:
                print(f"  {plot_path}")
        except ModuleNotFoundError as error:
            print(f"Skipping plots: {error}")


if __name__ == "__main__":
    main()
