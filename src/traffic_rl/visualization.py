"""Plotting helpers for experiment and tuning summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:  # pragma: no cover - exercised when optional plotting deps are installed
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - exercised in lean runtime envs
    plt = None


def _require_matplotlib():
    if plt is None:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install requirements-optional.txt to enable figures."
        )
    return plt


def _moving_average(values: list[float], window: int = 10) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=float)

    smoothed = []
    for index in range(len(values)):
        start = max(index - window + 1, 0)
        smoothed.append(float(np.mean(values[start : index + 1])))
    return np.asarray(smoothed, dtype=float)


def _prepare_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_training_history(
    training_history: list[dict[str, float]],
    output_path: str | Path,
) -> Path:
    """Plot core training metrics over episodes."""
    if not training_history:
        raise ValueError("training_history must not be empty")

    pyplot = _require_matplotlib()
    path = _prepare_output_path(output_path)
    episodes = [int(item["episode"]) + 1 for item in training_history]

    metrics = [
        ("total_reward", "Total Reward"),
        ("average_queue_length", "Average Queue Length"),
        ("average_wait_time_seconds", "Average Wait Time (s)"),
        ("throughput_per_step", "Throughput / Step"),
        ("epsilon", "Epsilon"),
        ("mean_loss", "Mean TD Loss"),
    ]

    figure, axes = pyplot.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, metrics):
        values = [float(item.get(key, np.nan)) for item in training_history]
        axis.plot(episodes, values, color="#4C6EF5", alpha=0.35, linewidth=1.3, label="raw")

        valid_pairs = [
            (episode, value)
            for episode, value in zip(episodes, values)
            if not np.isnan(value)
        ]
        if valid_pairs:
            valid_episodes = [episode for episode, _ in valid_pairs]
            valid_values = [value for _, value in valid_pairs]
            axis.plot(
                valid_episodes,
                _moving_average(valid_values, window=min(10, len(valid_values))),
                color="#1034A6",
                linewidth=2.0,
                label="moving avg",
            )

        axis.set_title(title)
        axis.set_xlabel("Episode")
        axis.grid(alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    figure.suptitle("DQN Training Curves", fontsize=15)
    figure.savefig(path, dpi=180)
    pyplot.close(figure)
    return path


def plot_evaluation_results(
    evaluation_results: Mapping[str, Mapping[str, Mapping[str, float]]],
    output_path: str | Path,
) -> Path:
    """Plot evaluation comparisons across regimes and policies."""
    if not evaluation_results:
        raise ValueError("evaluation_results must not be empty")

    pyplot = _require_matplotlib()
    path = _prepare_output_path(output_path)
    regimes = list(evaluation_results)
    policies = list(next(iter(evaluation_results.values())))
    metrics = [
        ("average_queue_length", "Average Queue Length"),
        ("average_wait_time_seconds", "Average Wait Time (s)"),
        ("throughput_per_step", "Throughput / Step"),
    ]
    colors = ["#1B9AAA", "#EF476F", "#FFD166", "#073B4C"]

    figure, axes = pyplot.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
    x = np.arange(len(regimes))
    width = 0.18 if policies else 0.6

    for axis, (metric_key, metric_title) in zip(axes, metrics):
        for index, policy in enumerate(policies):
            values = [
                float(evaluation_results[regime][policy][metric_key])
                for regime in regimes
            ]
            offset = (index - (len(policies) - 1) / 2) * width
            axis.bar(
                x + offset,
                values,
                width=width,
                label=policy,
                color=colors[index % len(colors)],
            )

        axis.set_title(metric_title)
        axis.set_xticks(x)
        axis.set_xticklabels(regimes, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.2)

    figure.legend(loc="upper center", ncol=len(policies), frameon=False)
    figure.suptitle("Policy Comparison Across Traffic Regimes", fontsize=15)
    figure.savefig(path, dpi=180)
    pyplot.close(figure)
    return path


def plot_tuning_results(
    tuning_summary: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Plot the ranked objective scores from hyperparameter tuning."""
    trials = list(tuning_summary.get("ranked_trials", tuning_summary.get("trials", [])))
    if not trials:
        raise ValueError("tuning summary does not contain any trials")

    objective = tuning_summary["objective"]
    objective_label = f"{objective.get('policy', 'dqn')} / {objective['regime']} / {objective['metric']}"
    mode = str(objective.get("mode", "min"))
    path = _prepare_output_path(output_path)

    pyplot = _require_matplotlib()
    figure, axes = pyplot.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    trial_labels = [f"T{int(trial['trial_id']):02d}" for trial in trials]
    scores = [float(trial["objective_score"]) for trial in trials]
    axes[0].bar(trial_labels, scores, color="#2A9D8F")
    axes[0].set_title(f"Objective Score by Trial ({mode})")
    axes[0].set_ylabel(objective["metric"])
    axes[0].grid(axis="y", alpha=0.2)

    lines = [f"Objective: {objective_label}", ""]
    for rank, trial in enumerate(trials[:5], start=1):
        lines.append(
            f"#{rank} T{int(trial['trial_id']):02d} score={trial['objective_score']:.3f}"
        )
        for dotted_path, value in trial["overrides"].items():
            lines.append(f"  {dotted_path}={value}")
        lines.append("")

    axes[1].axis("off")
    axes[1].text(
        0.0,
        1.0,
        "\n".join(lines).rstrip(),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )
    axes[1].set_title("Top Trial Settings")

    figure.suptitle("DQN Hyperparameter Search", fontsize=15)
    figure.savefig(path, dpi=180)
    pyplot.close(figure)
    return path


def generate_experiment_plots(summary: Mapping[str, Any], output_dir: str | Path) -> list[str]:
    """Generate the standard training and evaluation figures for one experiment."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    generated = [
        str(plot_training_history(summary["training_history"], directory / "training_curves.png")),
        str(plot_evaluation_results(summary["evaluation_results"], directory / "evaluation_comparison.png")),
    ]
    return generated
