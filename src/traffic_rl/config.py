"""Helpers for loading experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_env_kwargs(
    environment_config: Mapping[str, Any],
    arrival_schedule: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create environment kwargs from a config section and a schedule."""
    return {
        "arrival_schedule": arrival_schedule,
        "episode_length": int(environment_config.get("episode_length", 200)),
        "step_seconds": int(environment_config.get("step_seconds", 3)),
        "yellow_time": int(environment_config.get("yellow_time", 1)),
        "max_departures_per_step": int(environment_config.get("max_departures_per_step", 4)),
        "reward_mode": environment_config.get("reward_mode", "queue"),
        "switch_penalty": float(environment_config.get("switch_penalty", 2.0)),
    }
