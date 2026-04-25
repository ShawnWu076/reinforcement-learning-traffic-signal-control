"""Baseline controllers for adaptive traffic signal control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .env import KEEP_ACTION, SWITCH_ACTION, resolve_switch_allowed
from .grid_env import encode_grid_action


def _extract_state(
    observation: np.ndarray,
    info: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    return {
        "q_n": float(observation[0]),
        "q_s": float(observation[1]),
        "q_e": float(observation[2]),
        "q_w": float(observation[3]),
        "phase": int(round(float(observation[4]))),
        "phase_duration": float(observation[5]),
        "switch_allowed": float(resolve_switch_allowed(observation, info)),
    }


def _infer_grid_observation_stride(
    observation: np.ndarray,
    intersection_count: int,
    observation_variant: str | None = None,
) -> int:
    if observation_variant == "minimal":
        return 6
    if observation_variant == "full":
        return 13
    if observation.shape[0] == intersection_count * 13:
        return 13
    if observation.shape[0] == intersection_count * 6:
        return 6
    raise ValueError("grid observation length does not match the intersection count")


def _extract_grid_state(
    observation: np.ndarray,
    intersection_index: int,
    intersection_count: int,
    info: Mapping[str, Any] | None = None,
    observation_variant: str | None = None,
) -> dict[str, float]:
    stride = _infer_grid_observation_stride(
        observation,
        intersection_count=intersection_count,
        observation_variant=observation_variant,
    )
    start = intersection_index * stride
    local_observation = observation[start : start + stride]

    local_info: dict[str, Any] = {}
    if info is not None:
        intersection_ids = list(info.get("intersection_ids", []))
        if len(intersection_ids) > intersection_index:
            intersection_id = intersection_ids[intersection_index]
            allowed_by_intersection = info.get(
                "next_switch_allowed_by_intersection",
                info.get("switch_allowed_by_intersection", {}),
            )
            if isinstance(allowed_by_intersection, Mapping):
                local_info["next_switch_allowed"] = bool(
                    allowed_by_intersection.get(intersection_id, True)
                )

    return _extract_state(local_observation, local_info or None)


@dataclass
class FixedCycleController:
    """Switch every fixed number of active green steps."""

    cycle_length: int = 10
    name: str = "fixed_cycle"

    def act(self, observation: np.ndarray, info: Mapping[str, Any] | None = None) -> int:
        state = _extract_state(observation, info)
        if state["switch_allowed"] < 0.5:
            return KEEP_ACTION
        if state["phase_duration"] >= self.cycle_length:
            return SWITCH_ACTION
        return KEEP_ACTION


@dataclass
class QueueThresholdController:
    """Switch if the opposite direction is much more congested."""

    threshold: float = 5.0
    min_green: int = 3
    name: str = "queue_threshold"

    def act(self, observation: np.ndarray, info: Mapping[str, Any] | None = None) -> int:
        state = _extract_state(observation, info)
        ns_queue = state["q_n"] + state["q_s"]
        ew_queue = state["q_e"] + state["q_w"]

        if state["switch_allowed"] < 0.5:
            return KEEP_ACTION
        if state["phase_duration"] < self.min_green:
            return KEEP_ACTION

        if state["phase"] == 0 and (ew_queue - ns_queue) > self.threshold:
            return SWITCH_ACTION
        if state["phase"] == 1 and (ns_queue - ew_queue) > self.threshold:
            return SWITCH_ACTION
        return KEEP_ACTION


@dataclass
class MaxPressureController:
    """Always favor the larger queue once minimum green is satisfied."""

    min_green: int = 2
    name: str = "max_pressure"

    def act(self, observation: np.ndarray, info: Mapping[str, Any] | None = None) -> int:
        state = _extract_state(observation, info)
        ns_queue = state["q_n"] + state["q_s"]
        ew_queue = state["q_e"] + state["q_w"]

        if state["switch_allowed"] < 0.5:
            return KEEP_ACTION
        if state["phase_duration"] < self.min_green:
            return KEEP_ACTION

        if state["phase"] == 0 and ew_queue > ns_queue:
            return SWITCH_ACTION
        if state["phase"] == 1 and ns_queue > ew_queue:
            return SWITCH_ACTION
        return KEEP_ACTION


@dataclass
class GridFixedCycleController:
    """Apply fixed-cycle logic independently at every grid intersection."""

    cycle_length: int = 10
    intersection_count: int = 4
    observation_variant: str = "full"
    name: str = "grid_fixed_cycle"

    def act(self, observation: np.ndarray, info: Mapping[str, Any] | None = None) -> int:
        actions = []
        for index in range(self.intersection_count):
            state = _extract_grid_state(
                observation,
                intersection_index=index,
                intersection_count=self.intersection_count,
                info=info,
                observation_variant=self.observation_variant,
            )
            if state["switch_allowed"] < 0.5:
                actions.append(KEEP_ACTION)
            elif state["phase_duration"] >= self.cycle_length:
                actions.append(SWITCH_ACTION)
            else:
                actions.append(KEEP_ACTION)
        return encode_grid_action(actions)


@dataclass
class GridQueueThresholdController:
    """Switch each grid signal when the opposite axis is sufficiently busier."""

    threshold: float = 5.0
    min_green: int = 3
    intersection_count: int = 4
    observation_variant: str = "full"
    name: str = "grid_queue_threshold"

    def act(self, observation: np.ndarray, info: Mapping[str, Any] | None = None) -> int:
        actions = []
        for index in range(self.intersection_count):
            state = _extract_grid_state(
                observation,
                intersection_index=index,
                intersection_count=self.intersection_count,
                info=info,
                observation_variant=self.observation_variant,
            )
            ns_queue = state["q_n"] + state["q_s"]
            ew_queue = state["q_e"] + state["q_w"]

            action = KEEP_ACTION
            if state["switch_allowed"] >= 0.5 and state["phase_duration"] >= self.min_green:
                if state["phase"] == 0 and (ew_queue - ns_queue) > self.threshold:
                    action = SWITCH_ACTION
                elif state["phase"] == 1 and (ns_queue - ew_queue) > self.threshold:
                    action = SWITCH_ACTION
            actions.append(action)
        return encode_grid_action(actions)


@dataclass
class GridMaxPressureController:
    """Apply local max-pressure logic at every grid intersection."""

    min_green: int = 2
    intersection_count: int = 4
    observation_variant: str = "full"
    name: str = "grid_max_pressure"

    def act(self, observation: np.ndarray, info: Mapping[str, Any] | None = None) -> int:
        actions = []
        for index in range(self.intersection_count):
            state = _extract_grid_state(
                observation,
                intersection_index=index,
                intersection_count=self.intersection_count,
                info=info,
                observation_variant=self.observation_variant,
            )
            ns_queue = state["q_n"] + state["q_s"]
            ew_queue = state["q_e"] + state["q_w"]

            action = KEEP_ACTION
            if state["switch_allowed"] >= 0.5 and state["phase_duration"] >= self.min_green:
                if state["phase"] == 0 and ew_queue > ns_queue:
                    action = SWITCH_ACTION
                elif state["phase"] == 1 and ns_queue > ew_queue:
                    action = SWITCH_ACTION
            actions.append(action)
        return encode_grid_action(actions)
