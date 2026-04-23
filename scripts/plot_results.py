#!/usr/bin/env python3
"""Generate plots from an experiment or tuning summary JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_rl.visualization import generate_experiment_plots, plot_tuning_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Path to a training summary or tuning summary JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/plots/manual",
        help="Directory to save plots.",
    )
    args = parser.parse_args()

    summary_path = PROJECT_ROOT / args.summary
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if "training_history" in summary and "evaluation_results" in summary:
        plot_paths = generate_experiment_plots(summary, output_dir)
    elif "ranked_trials" in summary or "trials" in summary:
        plot_paths = [str(plot_tuning_results(summary, output_dir / "tuning_overview.png"))]
    else:
        raise ValueError("Unsupported summary format")

    print("Saved plots:")
    for plot_path in plot_paths:
        print(f"  {plot_path}")


if __name__ == "__main__":
    main()
