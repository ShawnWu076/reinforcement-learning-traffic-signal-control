"""Factories for creating traffic environments and matching baseline policies."""

from __future__ import annotations

from typing import Any, Mapping

from .baselines import (
    FixedCycleController,
    GridFixedCycleController,
    GridMaxPressureController,
    GridQueueThresholdController,
    MaxPressureController,
    QueueThresholdController,
)
from .config import build_env_kwargs
from .env import AdaptiveTrafficSignalEnv
from .grid_env import GridTrafficSignalEnv


def resolve_network_type(environment_config: Mapping[str, Any]) -> str:
    """Resolve the environment family from config."""
    return str(environment_config.get("network_type", "1x1")).lower()


def make_environment(
    environment_config: Mapping[str, Any],
    arrival_schedule: list[dict[str, Any]],
    seed: int | None = None,
) -> AdaptiveTrafficSignalEnv | GridTrafficSignalEnv:
    """Create a 1x1 or 2x2 environment from the shared config shape."""
    env_kwargs = build_env_kwargs(environment_config, arrival_schedule)
    network_type = resolve_network_type(environment_config)

    if network_type in {"1x1", "single", "single_intersection"}:
        return AdaptiveTrafficSignalEnv(**env_kwargs, seed=seed)
    if network_type in {"2x2", "grid", "grid_2x2"}:
        return GridTrafficSignalEnv(
            **env_kwargs,
            grid_shape=environment_config.get("grid_shape", [2, 2]),
            intersection_ids=environment_config.get("intersection_ids"),
            seed=seed,
        )

    raise ValueError(
        "environment.network_type must be one of '1x1', 'single', '2x2', or 'grid'"
    )


def make_baseline_policies(
    env: AdaptiveTrafficSignalEnv | GridTrafficSignalEnv,
) -> dict[str, Any]:
    """Build baseline policies that match the active environment action space."""
    if isinstance(env, GridTrafficSignalEnv):
        return {
            "fixed_cycle": GridFixedCycleController(
                cycle_length=10,
                intersection_count=env.intersection_count,
                observation_variant=env.observation_variant,
            ),
            "queue_threshold": GridQueueThresholdController(
                threshold=5.0,
                min_green=3,
                intersection_count=env.intersection_count,
                observation_variant=env.observation_variant,
            ),
            "max_pressure": GridMaxPressureController(
                min_green=2,
                intersection_count=env.intersection_count,
                observation_variant=env.observation_variant,
            ),
        }

    return {
        "fixed_cycle": FixedCycleController(cycle_length=10),
        "queue_threshold": QueueThresholdController(threshold=5.0, min_green=3),
        "max_pressure": MaxPressureController(min_green=2),
    }
