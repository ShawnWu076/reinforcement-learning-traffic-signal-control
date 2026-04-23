"""Convenience wrappers around the shared DQN experiment pipeline."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .experiments import train_and_evaluate_dqn


def summarize_training_history(training_history: list[Mapping[str, float]]) -> dict[str, float]:
    """Build a compact overview for quick inspection and tuning tables."""
    if not training_history:
        return {}

    best_episode = max(training_history, key=lambda item: float(item["total_reward"]))
    final_episode = training_history[-1]

    overview = {
        "episodes": float(len(training_history)),
        "best_reward_episode": float(best_episode["episode"]),
        "best_total_reward": float(best_episode["total_reward"]),
        "final_total_reward": float(final_episode["total_reward"]),
        "final_average_queue_length": float(final_episode["average_queue_length"]),
        "final_average_wait_time_seconds": float(final_episode["average_wait_time_seconds"]),
        "final_throughput_per_step": float(final_episode["throughput_per_step"]),
        "final_epsilon": float(final_episode["epsilon"]),
    }

    if "mean_loss" in final_episode:
        overview["final_mean_loss"] = float(final_episode["mean_loss"])
    if "invalid_switch_count" in final_episode:
        overview["final_invalid_switch_count"] = float(final_episode["invalid_switch_count"])

    return overview


def run_dqn_experiment(
    config: Mapping[str, Any],
    checkpoint_path: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run one DQN experiment and add a compact training overview."""
    if checkpoint_path is None:
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = train_and_evaluate_dqn(
                config=config,
                checkpoint_path=Path(temp_dir) / "dqn_policy.pt",
                summary_path=None,
                verbose=verbose,
            )
        payload["checkpoint"] = None
    else:
        resolved_checkpoint = Path(checkpoint_path)
        payload = train_and_evaluate_dqn(
            config=config,
            checkpoint_path=resolved_checkpoint,
            summary_path=None,
            verbose=verbose,
        )
        payload["checkpoint"] = str(resolved_checkpoint)

    payload["config"] = deepcopy(dict(config))
    payload["training_overview"] = summarize_training_history(payload.get("training_history", []))
    return payload


def save_experiment_summary(summary: Mapping[str, Any], output_path: str | Path) -> Path:
    """Persist an experiment summary as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
