#!/usr/bin/env python3
"""Run a lightweight hyperparameter search for the DQN controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised in lean runtime envs
    yaml = None

from traffic_rl.config import apply_overrides, load_config, parse_override_strings
from traffic_rl.experiment import run_dqn_experiment, save_experiment_summary
from traffic_rl.tuning import build_trial_overrides, extract_objective_score, sort_trials
from traffic_rl.visualization import generate_experiment_plots, plot_tuning_results

DEFAULT_CONFIG = "configs/default.yaml"
DEFAULT_OUTPUT = "results/tuning/tuning_summary.json"
DEFAULT_TRIALS_DIR = "results/tuning/trials"
DEFAULT_CHECKPOINTS_DIR = "results/tuning/checkpoints"
DEFAULT_PLOT_DIR = "results/plots/tuning"

PROFILE_DEFAULTS = {
    "1x1": {
        "config": "configs/default.yaml",
        "output": "results/tuning/1x1/tuning_summary.json",
        "trials_dir": "results/tuning/1x1/trials",
        "checkpoints_dir": "results/tuning/1x1/checkpoints",
        "plot_dir": "results/plots/tuning/1x1",
    },
    "2x2": {
        "config": "configs/grid_2x2.yaml",
        "output": "results/tuning/2x2/tuning_summary.json",
        "trials_dir": "results/tuning/2x2/trials",
        "checkpoints_dir": "results/tuning/2x2/checkpoints",
        "plot_dir": "results/plots/tuning/2x2",
    },
}


def _dump_config(path: Path, config: dict[str, object]) -> None:
    if yaml is not None:
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_DEFAULTS),
        default=None,
        help="Use standard paths for a 1x1 or 2x2 tuning run.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to YAML config with a tuning section.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Path to write the tuning summary JSON.",
    )
    parser.add_argument(
        "--trials-dir",
        type=str,
        default=DEFAULT_TRIALS_DIR,
        help="Directory where per-trial summaries are stored.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default=DEFAULT_CHECKPOINTS_DIR,
        help="Directory where per-trial checkpoints are stored.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=DEFAULT_PLOT_DIR,
        help="Directory for tuning plots and best-trial plots.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override base config values, e.g. --set training.episodes=120",
    )
    args = parser.parse_args()

    profile_defaults = PROFILE_DEFAULTS.get(args.profile, {})
    config_path_arg = (
        profile_defaults.get("config", args.config)
        if args.config == DEFAULT_CONFIG
        else args.config
    )
    output_arg = (
        profile_defaults.get("output", args.output)
        if args.output == DEFAULT_OUTPUT
        else args.output
    )
    trials_dir_arg = (
        profile_defaults.get("trials_dir", args.trials_dir)
        if args.trials_dir == DEFAULT_TRIALS_DIR
        else args.trials_dir
    )
    checkpoints_dir_arg = (
        profile_defaults.get("checkpoints_dir", args.checkpoints_dir)
        if args.checkpoints_dir == DEFAULT_CHECKPOINTS_DIR
        else args.checkpoints_dir
    )
    plot_dir_arg = (
        profile_defaults.get("plot_dir", args.plot_dir)
        if args.plot_dir == DEFAULT_PLOT_DIR
        else args.plot_dir
    )

    base_config = load_config(PROJECT_ROOT / config_path_arg)
    base_config = apply_overrides(base_config, parse_override_strings(args.overrides))

    tuning_config = base_config.get("tuning")
    if not isinstance(tuning_config, dict):
        raise ValueError("Config must define a top-level tuning section")

    objective = dict(tuning_config.get("objective", {}))
    objective.setdefault("regime", "nonstationary")
    objective.setdefault("metric", "average_wait_time_seconds")
    objective.setdefault("policy", "dqn")
    objective.setdefault("mode", "min")

    trials_dir = PROJECT_ROOT / trials_dir_arg
    checkpoints_dir = PROJECT_ROOT / checkpoints_dir_arg
    plot_dir = PROJECT_ROOT / plot_dir_arg
    trials_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    trial_overrides = build_trial_overrides(tuning_config)
    trial_records: list[dict[str, object]] = []
    best_summary: dict[str, object] | None = None

    print(f"Running {len(trial_overrides)} tuning trial(s)...\n")

    for index, overrides in enumerate(trial_overrides, start=1):
        trial_name = f"trial_{index:03d}"
        trial_config = apply_overrides(base_config, overrides)
        trial_checkpoint = checkpoints_dir / f"{trial_name}.pt"
        trial_summary_path = trials_dir / f"{trial_name}.json"

        print(f"[{index}/{len(trial_overrides)}] {trial_name} overrides={overrides}")
        summary = run_dqn_experiment(
            config=trial_config,
            checkpoint_path=trial_checkpoint,
            verbose=False,
        )
        save_experiment_summary(summary, trial_summary_path)

        score = extract_objective_score(summary, objective)
        trial_records.append(
            {
                "trial_id": index,
                "trial_name": trial_name,
                "objective_score": score,
                "overrides": overrides,
                "summary_path": str(trial_summary_path),
                "checkpoint": str(trial_checkpoint),
                "training_overview": summary["training_overview"],
                "evaluation_results": summary["evaluation_results"],
            }
        )

        if best_summary is None:
            best_summary = summary

        mode = str(objective["mode"])
        if best_summary is not None:
            best_score = extract_objective_score(best_summary, objective)
            is_better = score < best_score if mode == "min" else score > best_score
            if is_better:
                best_summary = summary

        print(f"    objective={score:.4f}")

    ranked_trials = sort_trials(trial_records, mode=str(objective["mode"]))
    best_trial = ranked_trials[0]
    best_config = apply_overrides(base_config, best_trial["overrides"])
    best_config_path = trials_dir.parent / "best_config.yaml"

    _dump_config(best_config_path, best_config)

    tuning_summary = {
        "config": base_config,
        "objective": objective,
        "trials": trial_records,
        "ranked_trials": ranked_trials,
        "best_trial": best_trial,
        "best_config_path": str(best_config_path),
    }

    output_path = PROJECT_ROOT / output_arg
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(tuning_summary, file, indent=2)

    try:
        plot_tuning_results(tuning_summary, plot_dir / "tuning_overview.png")

        best_trial_summary_path = Path(best_trial["summary_path"])
        with best_trial_summary_path.open("r", encoding="utf-8") as file:
            best_trial_summary = json.load(file)
        generate_experiment_plots(best_trial_summary, plot_dir / "best_trial")
    except ModuleNotFoundError as error:
        print(f"Skipping plots: {error}")

    print(f"\nSaved tuning summary to {output_path}")
    print(f"Saved best config to {best_config_path}")
    print(f"Best trial: {best_trial['trial_name']} objective={best_trial['objective_score']:.4f}")


if __name__ == "__main__":
    main()
