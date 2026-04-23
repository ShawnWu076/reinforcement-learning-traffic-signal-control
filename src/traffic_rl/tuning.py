"""Helpers for simple hyperparameter search over DQN configs."""

from __future__ import annotations

import itertools
import random
from typing import Any, Mapping


def build_trial_overrides(tuning_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand the configured search space into a list of trial overrides."""
    search_space = tuning_config.get("search_space", {})
    if not search_space:
        raise ValueError("tuning.search_space must define at least one parameter")

    normalized_space: dict[str, list[Any]] = {}
    for dotted_path, candidates in dict(search_space).items():
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"Search space for '{dotted_path}' must be a non-empty list")
        normalized_space[dotted_path] = candidates

    keys = list(normalized_space)
    combinations = [
        dict(zip(keys, values))
        for values in itertools.product(*(normalized_space[key] for key in keys))
    ]

    fixed_overrides = dict(tuning_config.get("fixed_overrides", {}))
    search_type = str(tuning_config.get("search_type", "grid"))
    seed = int(tuning_config.get("seed", 0))
    max_trials = tuning_config.get("max_trials")
    rng = random.Random(seed)

    if search_type == "grid":
        if max_trials is not None and int(max_trials) < len(combinations):
            rng.shuffle(combinations)
            combinations = combinations[: int(max_trials)]
    elif search_type == "random":
        rng.shuffle(combinations)
        if max_trials is not None:
            combinations = combinations[: int(max_trials)]
    else:
        raise ValueError("tuning.search_type must be either 'grid' or 'random'")

    return [{**fixed_overrides, **combo} for combo in combinations]


def extract_objective_score(
    experiment_summary: Mapping[str, Any],
    objective_config: Mapping[str, Any],
) -> float:
    """Read the objective value from evaluation results."""
    regime = str(objective_config.get("regime", "nonstationary"))
    metric = str(objective_config.get("metric", "average_wait_time_seconds"))
    policy = str(objective_config.get("policy", "dqn"))

    return float(experiment_summary["evaluation_results"][regime][policy][metric])


def sort_trials(
    trials: list[dict[str, Any]],
    mode: str,
) -> list[dict[str, Any]]:
    """Sort trials according to the objective direction."""
    reverse = mode == "max"
    return sorted(trials, key=lambda item: item["objective_score"], reverse=reverse)
