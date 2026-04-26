#!/usr/bin/env python3
"""Train a DQN controller and compare it against baseline policies."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_rl.config import apply_overrides, load_config, parse_override_strings
from traffic_rl.experiments import train_and_evaluate_dqn, train_and_evaluate_dqn_multiseed
from traffic_rl.visualization import generate_experiment_plots

DEFAULT_CONFIG = "configs/default.yaml"
DEFAULT_CHECKPOINT = "results/checkpoints/dqn_policy.pt"
DEFAULT_SUMMARY_OUTPUT = "results/dqn_summary.json"
DEFAULT_PLOT_DIR = "results/plots/dqn"
DEFAULT_MULTI_SEED_SUMMARY_OUTPUT = "results/dqn_multiseed_summary.json"
DEFAULT_MULTI_SEED_OUTPUT_DIR = "results/multiseed"

PROFILE_DEFAULTS = {
    "1x1": {
        "config": "configs/default.yaml",
        "checkpoint": "results/experiments/1x1/dqn_policy.pt",
        "summary_output": "results/experiments/1x1/dqn_summary.json",
        "plot_dir": "results/plots/experiments/1x1",
        "multiseed_summary_output": "results/experiments/1x1/dqn_multiseed_summary.json",
        "multiseed_output_dir": "results/experiments/1x1/multiseed",
    },
    "2x2": {
        "config": "configs/grid_2x2.yaml",
        "checkpoint": "results/experiments/2x2/dqn_policy.pt",
        "summary_output": "results/experiments/2x2/dqn_summary.json",
        "plot_dir": "results/plots/experiments/2x2",
        "multiseed_summary_output": "results/experiments/2x2/dqn_multiseed_summary.json",
        "multiseed_output_dir": "results/experiments/2x2/multiseed",
    },
}


def parse_seed_list(raw_value: str | None) -> list[int]:
    """Parse comma-separated seed values from CLI input."""
    if raw_value is None or not raw_value.strip():
        return []
    seeds = []
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        seeds.append(int(item))
    if not seeds:
        raise ValueError("--seeds must include at least one integer seed")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_DEFAULTS),
        default=None,
        help="Use standard paths for a 1x1 or 2x2 experiment.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default=DEFAULT_SUMMARY_OUTPUT,
        help="Path to save training and evaluation metrics.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=DEFAULT_PLOT_DIR,
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
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seed list for a multi-seed DQN run, e.g. 7,17,27.",
    )
    parser.add_argument(
        "--multiseed-summary-output",
        type=str,
        default=DEFAULT_MULTI_SEED_SUMMARY_OUTPUT,
        help="Path to save aggregated multi-seed metrics.",
    )
    parser.add_argument(
        "--multiseed-output-dir",
        type=str,
        default=DEFAULT_MULTI_SEED_OUTPUT_DIR,
        help="Directory for per-seed checkpoints and summaries.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    args = parser.parse_args()

    profile_defaults = PROFILE_DEFAULTS.get(args.profile, {})
    config_path_arg = (
        profile_defaults.get("config", args.config)
        if args.config == DEFAULT_CONFIG
        else args.config
    )
    checkpoint_arg = (
        profile_defaults.get("checkpoint", args.checkpoint)
        if args.checkpoint == DEFAULT_CHECKPOINT
        else args.checkpoint
    )
    summary_output_arg = (
        profile_defaults.get("summary_output", args.summary_output)
        if args.summary_output == DEFAULT_SUMMARY_OUTPUT
        else args.summary_output
    )
    plot_dir_arg = (
        profile_defaults.get("plot_dir", args.plot_dir)
        if args.plot_dir == DEFAULT_PLOT_DIR
        else args.plot_dir
    )
    multiseed_summary_output_arg = (
        profile_defaults.get("multiseed_summary_output", args.multiseed_summary_output)
        if args.multiseed_summary_output == DEFAULT_MULTI_SEED_SUMMARY_OUTPUT
        else args.multiseed_summary_output
    )
    multiseed_output_dir_arg = (
        profile_defaults.get("multiseed_output_dir", args.multiseed_output_dir)
        if args.multiseed_output_dir == DEFAULT_MULTI_SEED_OUTPUT_DIR
        else args.multiseed_output_dir
    )

    overrides = parse_override_strings(args.overrides)
    config = apply_overrides(load_config(PROJECT_ROOT / config_path_arg), overrides)
    seeds = parse_seed_list(args.seeds)

    if seeds:
        train_and_evaluate_dqn_multiseed(
            config=config,
            seeds=seeds,
            output_dir=PROJECT_ROOT / multiseed_output_dir_arg,
            summary_path=PROJECT_ROOT / multiseed_summary_output_arg,
            verbose=True,
        )
        return

    checkpoint_path = PROJECT_ROOT / checkpoint_arg
    output_path = PROJECT_ROOT / summary_output_arg

    summary = train_and_evaluate_dqn(
        config=config,
        checkpoint_path=checkpoint_path,
        summary_path=output_path,
        verbose=True,
    )

    if not args.no_plots:
        try:
            plot_paths = generate_experiment_plots(summary, PROJECT_ROOT / plot_dir_arg)
            print("Saved plots:")
            for plot_path in plot_paths:
                print(f"  {plot_path}")
        except ModuleNotFoundError as error:
            print(f"Skipping plots: {error}")


if __name__ == "__main__":
    main()
