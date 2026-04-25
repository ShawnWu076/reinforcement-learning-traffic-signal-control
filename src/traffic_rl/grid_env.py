"""Multi-intersection grid environment for centralized traffic signal control."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Mapping, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .env import (
    DIRECTIONS,
    EW_DIRECTIONS,
    KEEP_ACTION,
    NS_DIRECTIONS,
    OBSERVATION_VARIANTS,
    SWITCH_ACTION,
)


def encode_grid_action(local_actions: Sequence[int]) -> int:
    """Encode per-intersection keep/switch decisions into one discrete action."""
    encoded = 0
    for index, action in enumerate(local_actions):
        if int(action) not in {KEEP_ACTION, SWITCH_ACTION}:
            raise ValueError("local grid actions must be 0 (keep) or 1 (switch)")
        encoded |= int(action) << index
    return encoded


def decode_grid_action(action: int, intersection_count: int) -> np.ndarray:
    """Decode one discrete joint action into per-intersection keep/switch decisions."""
    if intersection_count <= 0:
        raise ValueError("intersection_count must be > 0")
    action_dim = 2**intersection_count
    if int(action) < 0 or int(action) >= action_dim:
        raise ValueError(f"action must be in [0, {action_dim - 1}]")
    return np.asarray(
        [(int(action) >> index) & 1 for index in range(intersection_count)],
        dtype=np.int64,
    )


def build_grid_action_mask(switch_allowed: Sequence[bool]) -> np.ndarray:
    """Build a mask over all joint actions from local switch permissions."""
    allowed = [bool(value) for value in switch_allowed]
    action_dim = 2 ** len(allowed)
    mask = np.zeros(action_dim, dtype=np.float32)

    for action in range(action_dim):
        local_actions = decode_grid_action(action, len(allowed))
        is_valid = all(
            local_action == KEEP_ACTION or allowed[index]
            for index, local_action in enumerate(local_actions)
        )
        mask[action] = 1.0 if is_valid else 0.0

    return mask


@dataclass(frozen=True)
class GridScheduleSegment:
    """Arrival-rate segment active until a given time step."""

    until_step: int
    rates: dict[str, dict[str, float]]


@dataclass
class GridTrafficMetrics:
    """Episode-level metrics tracked by the grid environment."""

    total_reward: float = 0.0
    total_departed: int = 0
    total_served: int = 0
    total_wait_time_steps: float = 0.0
    internal_transfer_count: int = 0
    switch_count: int = 0
    switch_requested_count: int = 0
    switch_applied_count: int = 0
    invalid_switch_count: int = 0
    queue_history: list[int] = field(default_factory=list)
    reward_history: list[float] = field(default_factory=list)
    throughput_history: list[int] = field(default_factory=list)
    switch_requested_history: list[int] = field(default_factory=list)
    switch_applied_history: list[int] = field(default_factory=list)
    invalid_switch_history: list[int] = field(default_factory=list)

    def summary(self, step_seconds: int, intersection_count: int) -> dict[str, float]:
        """Aggregate metrics for reporting."""
        avg_queue = float(np.mean(self.queue_history)) if self.queue_history else 0.0
        max_queue = int(max(self.queue_history)) if self.queue_history else 0
        avg_wait_steps = (
            self.total_wait_time_steps / self.total_departed if self.total_departed else 0.0
        )
        throughput_per_step = (
            self.total_departed / len(self.queue_history) if self.queue_history else 0.0
        )
        avg_switches_per_intersection = (
            self.switch_applied_count / intersection_count if intersection_count else 0.0
        )

        return {
            "total_reward": float(self.total_reward),
            "average_queue_length": avg_queue,
            "maximum_queue_length": float(max_queue),
            "throughput_per_step": float(throughput_per_step),
            "total_departed": float(self.total_departed),
            "total_served": float(self.total_served),
            "average_wait_time_steps": float(avg_wait_steps),
            "average_wait_time_seconds": float(avg_wait_steps * step_seconds),
            "switch_count": float(self.switch_count),
            "switch_requested_count": float(self.switch_requested_count),
            "switch_applied_count": float(self.switch_applied_count),
            "invalid_switch_count": float(self.invalid_switch_count),
            "internal_transfer_count": float(self.internal_transfer_count),
            "average_switches_per_intersection": float(avg_switches_per_intersection),
        }


class GridTrafficSignalEnv(gym.Env):
    """Centralized 2D grid environment with one binary signal action per intersection."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        arrival_schedule: Sequence[Mapping[str, Any]],
        grid_shape: Sequence[int] = (2, 2),
        intersection_ids: Sequence[str] | None = None,
        episode_length: int = 200,
        step_seconds: int = 3,
        min_green_time: int = 2,
        yellow_time: int = 1,
        max_departures_per_step: int = 4,
        recent_arrival_window: int = 5,
        reward_mode: str = "queue",
        switch_penalty: float = 2.0,
        observation_variant: str = "full",
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        if len(tuple(grid_shape)) != 2:
            raise ValueError("grid_shape must contain [rows, columns]")
        rows, columns = int(grid_shape[0]), int(grid_shape[1])
        if rows <= 0 or columns <= 0:
            raise ValueError("grid_shape rows and columns must be > 0")
        if reward_mode not in {"queue", "waiting"}:
            raise ValueError("reward_mode must be either 'queue' or 'waiting'")
        if episode_length <= 0:
            raise ValueError("episode_length must be > 0")
        if step_seconds <= 0:
            raise ValueError("step_seconds must be > 0")
        if min_green_time < 0:
            raise ValueError("min_green_time must be >= 0")
        if yellow_time < 0:
            raise ValueError("yellow_time must be >= 0")
        if max_departures_per_step <= 0:
            raise ValueError("max_departures_per_step must be > 0")
        if recent_arrival_window <= 0:
            raise ValueError("recent_arrival_window must be > 0")
        if observation_variant not in OBSERVATION_VARIANTS:
            raise ValueError(
                f"observation_variant must be one of {sorted(OBSERVATION_VARIANTS)}"
            )
        if render_mode not in {None, "human"}:
            raise ValueError("render_mode must be None or 'human'")

        super().__init__()
        self.grid_shape = (rows, columns)
        self.intersection_ids = self._resolve_intersection_ids(intersection_ids)
        self.intersection_count = len(self.intersection_ids)
        self.id_to_position = {
            intersection_id: divmod(index, columns)
            for index, intersection_id in enumerate(self.intersection_ids)
        }
        self.position_to_id = {
            position: intersection_id for intersection_id, position in self.id_to_position.items()
        }

        self.arrival_schedule = self._normalize_schedule(arrival_schedule)
        self.episode_length = int(episode_length)
        self.step_seconds = int(step_seconds)
        self.min_green_time = int(min_green_time)
        self.yellow_time = int(yellow_time)
        self.max_departures_per_step = int(max_departures_per_step)
        self.recent_arrival_window = int(recent_arrival_window)
        self.reward_mode = reward_mode
        self.switch_penalty = float(switch_penalty)
        self.observation_variant = observation_variant
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)

        self.local_observation_dim = 13 if self.observation_variant == "full" else 6
        self.observation_dim = self.local_observation_dim * self.intersection_count
        self.action_dim = 2**self.intersection_count
        self.action_space = spaces.Discrete(self.action_dim)
        self.observation_space = self._build_observation_space()

        self.step_count = 0
        self.current_phase: dict[str, int] = {}
        self.phase_duration: dict[str, int] = {}
        self.yellow_remaining: dict[str, int] = {}
        self.pending_phase: dict[str, int | None] = {}
        self.queues: dict[str, dict[str, Deque[int]]] = {}
        self.recent_arrivals: dict[str, Deque[np.ndarray]] = {}
        self.metrics = GridTrafficMetrics()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the episode state and return the initial observation."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        options = {} if options is None else dict(options)
        initial_phases = self._resolve_initial_phases(options.get("initial_phases"))

        self.step_count = 0
        self.current_phase = dict(initial_phases)
        self.phase_duration = {intersection_id: 0 for intersection_id in self.intersection_ids}
        self.yellow_remaining = {intersection_id: 0 for intersection_id in self.intersection_ids}
        self.pending_phase = {intersection_id: None for intersection_id in self.intersection_ids}
        self.queues = {
            intersection_id: {direction: deque() for direction in DIRECTIONS}
            for intersection_id in self.intersection_ids
        }
        self.recent_arrivals = {
            intersection_id: deque(maxlen=self.recent_arrival_window)
            for intersection_id in self.intersection_ids
        }
        self.metrics = GridTrafficMetrics()

        self._load_initial_queues(options.get("initial_queues"))

        observation = self._get_observation()
        return observation, self._build_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the grid simulator by one step."""
        if self.step_count >= self.episode_length:
            raise RuntimeError("episode is done; call reset() before calling step() again")

        local_actions = decode_grid_action(int(action), self.intersection_count)
        switch_allowed = self._switch_allowed_map()
        switch_requested = {
            intersection_id: bool(local_actions[index] == SWITCH_ACTION)
            for index, intersection_id in enumerate(self.intersection_ids)
        }
        invalid_switch = {
            intersection_id: bool(
                switch_requested[intersection_id] and not switch_allowed[intersection_id]
            )
            for intersection_id in self.intersection_ids
        }
        switch_applied = {intersection_id: False for intersection_id in self.intersection_ids}
        in_yellow = {intersection_id: False for intersection_id in self.intersection_ids}

        waiting_increment = self._total_queue_length()
        self._age_queued_vehicles()

        arrivals = self._sample_arrivals()
        for intersection_id in self.intersection_ids:
            arrival_vector = np.asarray(
                [arrivals[intersection_id][direction] for direction in DIRECTIONS],
                dtype=np.float32,
            )
            self.recent_arrivals[intersection_id].append(arrival_vector)

        departures = {
            intersection_id: {direction: 0 for direction in DIRECTIONS}
            for intersection_id in self.intersection_ids
        }
        network_exits = 0
        internal_transfers: list[tuple[str, str, int]] = []

        for intersection_id in self.intersection_ids:
            if self.yellow_remaining[intersection_id] > 0:
                in_yellow[intersection_id] = True
                self._consume_yellow_step(intersection_id)
            elif switch_requested[intersection_id] and switch_allowed[intersection_id]:
                switch_applied[intersection_id] = True
                self.metrics.switch_count += 1
                if self.yellow_time > 0:
                    self.pending_phase[intersection_id] = 1 - self.current_phase[intersection_id]
                    self.yellow_remaining[intersection_id] = self.yellow_time
                    in_yellow[intersection_id] = True
                    self._consume_yellow_step(intersection_id)
                else:
                    self.current_phase[intersection_id] = 1 - self.current_phase[intersection_id]
                    self.phase_duration[intersection_id] = 0
                    served = self._serve_intersection(intersection_id)
                    departures[intersection_id] = served["departures"]
                    network_exits += served["network_exits"]
                    internal_transfers.extend(served["internal_transfers"])
                    self.phase_duration[intersection_id] += 1
            else:
                served = self._serve_intersection(intersection_id)
                departures[intersection_id] = served["departures"]
                network_exits += served["network_exits"]
                internal_transfers.extend(served["internal_transfers"])
                self.phase_duration[intersection_id] += 1

        for intersection_id, direction, age in internal_transfers:
            self.queues[intersection_id][direction].append(age)
        for intersection_id, direction_rates in arrivals.items():
            for direction, count in direction_rates.items():
                self.queues[intersection_id][direction].extend([0] * count)

        applied_count = sum(int(value) for value in switch_applied.values())
        queue_length = self._total_queue_length()
        reward_signal = -queue_length if self.reward_mode == "queue" else -waiting_increment
        reward = float(reward_signal - self.switch_penalty * applied_count)

        self.metrics.total_reward += reward
        self.metrics.switch_requested_count += sum(int(value) for value in switch_requested.values())
        self.metrics.switch_applied_count += applied_count
        self.metrics.invalid_switch_count += sum(int(value) for value in invalid_switch.values())
        self.metrics.queue_history.append(queue_length)
        self.metrics.reward_history.append(reward)
        self.metrics.throughput_history.append(network_exits)
        self.metrics.switch_requested_history.append(
            sum(int(value) for value in switch_requested.values())
        )
        self.metrics.switch_applied_history.append(applied_count)
        self.metrics.invalid_switch_history.append(
            sum(int(value) for value in invalid_switch.values())
        )

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.episode_length
        observation = self._get_observation()
        info = self._build_info(
            arrivals=arrivals,
            departures=departures,
            switch_requested=switch_requested,
            switch_allowed=switch_allowed,
            switch_applied=switch_applied,
            invalid_switch=invalid_switch,
            in_yellow=in_yellow,
            local_actions=local_actions,
        )
        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        """Print a compact text representation of the current grid state."""
        queue_summary = {
            intersection_id: sum(len(self.queues[intersection_id][direction]) for direction in DIRECTIONS)
            for intersection_id in self.intersection_ids
        }
        print(
            "[GridTrafficSignalEnv] "
            f"step={self.step_count} "
            f"phases={self.current_phase} "
            f"queues={queue_summary} "
            f"switch_allowed={self._switch_allowed_map()} "
            f"yellow_remaining={self.yellow_remaining}"
        )

    def summarize(self) -> dict[str, float]:
        """Return a summary of the current episode metrics."""
        return self.metrics.summary(
            step_seconds=self.step_seconds,
            intersection_count=self.intersection_count,
        )

    def local_observation_slice(self, intersection_index: int) -> slice:
        """Return the observation slice for one intersection."""
        if intersection_index < 0 or intersection_index >= self.intersection_count:
            raise ValueError("intersection_index is out of range")
        start = intersection_index * self.local_observation_dim
        return slice(start, start + self.local_observation_dim)

    def _resolve_intersection_ids(self, intersection_ids: Sequence[str] | None) -> list[str]:
        rows, columns = self.grid_shape
        expected_count = rows * columns
        if intersection_ids is None:
            return [f"I{index}" for index in range(expected_count)]

        resolved = [str(intersection_id) for intersection_id in intersection_ids]
        if len(resolved) != expected_count:
            raise ValueError("intersection_ids length must match rows * columns")
        if len(set(resolved)) != len(resolved):
            raise ValueError("intersection_ids must be unique")
        return resolved

    def _resolve_initial_phases(self, raw_phases: Any) -> dict[str, int]:
        if raw_phases is None:
            return {intersection_id: 0 for intersection_id in self.intersection_ids}
        if isinstance(raw_phases, Mapping):
            phases = {
                intersection_id: int(raw_phases.get(intersection_id, 0))
                for intersection_id in self.intersection_ids
            }
        else:
            if len(raw_phases) != self.intersection_count:
                raise ValueError("initial_phases must match the intersection count")
            phases = {
                intersection_id: int(raw_phases[index])
                for index, intersection_id in enumerate(self.intersection_ids)
            }
        for phase in phases.values():
            if phase not in {0, 1}:
                raise ValueError("initial phases must be 0 or 1")
        return phases

    def _load_initial_queues(self, raw_queues: Any) -> None:
        if raw_queues is None:
            return
        if not isinstance(raw_queues, Mapping):
            raise ValueError("initial_queues must be a mapping of intersection ids")

        for intersection_id, direction_counts in raw_queues.items():
            if intersection_id not in self.queues:
                raise ValueError(f"unknown intersection id in initial_queues: {intersection_id}")
            if not isinstance(direction_counts, Mapping):
                raise ValueError("each initial queue entry must map directions to counts")
            for direction, count in direction_counts.items():
                if direction not in DIRECTIONS:
                    raise ValueError(f"unknown direction in initial_queues: {direction}")
                count_int = int(count)
                if count_int < 0:
                    raise ValueError("initial queue lengths must be >= 0")
                self.queues[intersection_id][direction].extend([0] * count_int)

    def _normalize_schedule(
        self,
        schedule: Sequence[Mapping[str, Any]],
    ) -> list[GridScheduleSegment]:
        if not schedule:
            raise ValueError("arrival_schedule must contain at least one segment")

        normalized: list[GridScheduleSegment] = []
        previous_until = -1
        for segment in schedule:
            until_step = int(segment["until_step"])
            if until_step <= previous_until:
                raise ValueError("arrival_schedule must be sorted by increasing until_step")
            if until_step < 0:
                raise ValueError("arrival_schedule until_step values must be >= 0")

            rates = {
                intersection_id: {direction: 0.0 for direction in DIRECTIONS}
                for intersection_id in self.intersection_ids
            }
            raw_rates = segment.get("rates", {})
            if not isinstance(raw_rates, Mapping):
                raise ValueError("arrival_schedule rates must be a mapping")

            for key, value in raw_rates.items():
                key_text = str(key)
                if key_text in self.intersection_ids:
                    if not isinstance(value, Mapping):
                        raise ValueError("nested grid rates must map directions to values")
                    for direction in DIRECTIONS:
                        rates[key_text][direction] = float(value.get(direction, 0.0))
                elif key_text in DIRECTIONS:
                    rate = float(value)
                    for intersection_id in self.intersection_ids:
                        rates[intersection_id][key_text] = rate
                elif "." in key_text:
                    intersection_id, direction = key_text.split(".", maxsplit=1)
                    if intersection_id not in self.intersection_ids:
                        raise ValueError(f"unknown intersection id in rates: {intersection_id}")
                    if direction not in DIRECTIONS:
                        raise ValueError(f"unknown direction in rates: {direction}")
                    rates[intersection_id][direction] = float(value)
                else:
                    raise ValueError(
                        "grid rates must use direction keys, intersection keys, or 'intersection.direction'"
                    )

            for intersection_id, direction_rates in rates.items():
                for direction, rate in direction_rates.items():
                    if not np.isfinite(rate) or rate < 0.0:
                        raise ValueError(
                            f"arrival rate for {intersection_id}.{direction} must be finite and >= 0"
                        )

            previous_until = until_step
            normalized.append(GridScheduleSegment(until_step=until_step, rates=rates))

        return normalized

    def _build_info(self, **extra: Any) -> dict[str, Any]:
        switch_allowed = self._switch_allowed_map()
        switch_allowed_vector = [
            switch_allowed[intersection_id] for intersection_id in self.intersection_ids
        ]
        action_mask = build_grid_action_mask(switch_allowed_vector)
        queue_lengths = self._queue_lengths_by_intersection()

        info: dict[str, Any] = {
            "step": self.step_count,
            "network_type": "2x2",
            "intersection_ids": list(self.intersection_ids),
            "grid_shape": list(self.grid_shape),
            "queue_length": self._total_queue_length(),
            "queue_lengths_by_intersection": queue_lengths,
            "arrival_rates": self._current_arrival_rates(),
            "phases": dict(self.current_phase),
            "phase_durations": dict(self.phase_duration),
            "switch_allowed_by_intersection": switch_allowed,
            "next_switch_allowed_by_intersection": switch_allowed,
            "action_mask": action_mask,
            "yellow_remaining_by_intersection": dict(self.yellow_remaining),
            "pending_phase_by_intersection": dict(self.pending_phase),
            "observation_variant": self.observation_variant,
        }
        info.update(extra)
        return info

    def _switch_allowed_map(self) -> dict[str, bool]:
        return {
            intersection_id: bool(
                self.yellow_remaining[intersection_id] == 0
                and self.phase_duration[intersection_id] >= self.min_green_time
            )
            for intersection_id in self.intersection_ids
        }

    def _consume_yellow_step(self, intersection_id: str) -> None:
        if self.yellow_remaining[intersection_id] <= 0:
            raise RuntimeError("_consume_yellow_step called when not in yellow")

        self.yellow_remaining[intersection_id] -= 1
        if self.yellow_remaining[intersection_id] == 0:
            pending_phase = self.pending_phase[intersection_id]
            if pending_phase is None:
                raise RuntimeError("yellow ended but pending_phase is missing")
            self.current_phase[intersection_id] = pending_phase
            self.pending_phase[intersection_id] = None
            self.phase_duration[intersection_id] = 0

    def _serve_intersection(self, intersection_id: str) -> dict[str, Any]:
        departures = {direction: 0 for direction in DIRECTIONS}
        network_exits = 0
        internal_transfers: list[tuple[str, str, int]] = []
        active_directions = (
            NS_DIRECTIONS if self.current_phase[intersection_id] == 0 else EW_DIRECTIONS
        )

        for direction in active_directions:
            departures[direction] = min(
                len(self.queues[intersection_id][direction]),
                self.max_departures_per_step,
            )
            for _ in range(departures[direction]):
                age = self.queues[intersection_id][direction].popleft()
                self.metrics.total_served += 1
                downstream = self._downstream_approach(intersection_id, direction)
                if downstream is None:
                    self.metrics.total_wait_time_steps += age
                    self.metrics.total_departed += 1
                    network_exits += 1
                else:
                    target_intersection, target_direction = downstream
                    internal_transfers.append((target_intersection, target_direction, age))
                    self.metrics.internal_transfer_count += 1

        return {
            "departures": departures,
            "network_exits": network_exits,
            "internal_transfers": internal_transfers,
        }

    def _downstream_approach(self, intersection_id: str, direction: str) -> tuple[str, str] | None:
        row, column = self.id_to_position[intersection_id]
        if direction == "N":
            position = (row + 1, column)
            target_direction = "N"
        elif direction == "S":
            position = (row - 1, column)
            target_direction = "S"
        elif direction == "E":
            position = (row, column - 1)
            target_direction = "E"
        elif direction == "W":
            position = (row, column + 1)
            target_direction = "W"
        else:
            raise ValueError(f"unknown direction: {direction}")

        target_intersection = self.position_to_id.get(position)
        if target_intersection is None:
            return None
        return target_intersection, target_direction

    def _total_queue_length(self) -> int:
        return sum(
            len(self.queues[intersection_id][direction])
            for intersection_id in self.intersection_ids
            for direction in DIRECTIONS
        )

    def _queue_lengths_by_intersection(self) -> dict[str, dict[str, int]]:
        return {
            intersection_id: {
                direction: len(self.queues[intersection_id][direction])
                for direction in DIRECTIONS
            }
            for intersection_id in self.intersection_ids
        }

    def _current_arrival_rates(self) -> dict[str, dict[str, float]]:
        for segment in self.arrival_schedule:
            if self.step_count < segment.until_step:
                return {
                    intersection_id: dict(direction_rates)
                    for intersection_id, direction_rates in segment.rates.items()
                }
        return {
            intersection_id: dict(direction_rates)
            for intersection_id, direction_rates in self.arrival_schedule[-1].rates.items()
        }

    def _sample_arrivals(self) -> dict[str, dict[str, int]]:
        rates = self._current_arrival_rates()
        return {
            intersection_id: {
                direction: int(self.rng.poisson(rates[intersection_id][direction]))
                for direction in DIRECTIONS
            }
            for intersection_id in self.intersection_ids
        }

    def _age_queued_vehicles(self) -> None:
        for intersection_id in self.intersection_ids:
            for direction in DIRECTIONS:
                self.queues[intersection_id][direction] = deque(
                    age + 1 for age in self.queues[intersection_id][direction]
                )

    def _recent_arrival_means(self, intersection_id: str) -> np.ndarray:
        if not self.recent_arrivals[intersection_id]:
            return np.zeros(4, dtype=np.float32)
        stacked = np.stack(list(self.recent_arrivals[intersection_id]), axis=0)
        return np.mean(stacked, axis=0).astype(np.float32)

    def _get_observation(self) -> np.ndarray:
        normalized_step = float(self.step_count) / float(self.episode_length)
        normalized_step = float(min(max(normalized_step, 0.0), 1.0))
        values: list[float] = []

        for intersection_id in self.intersection_ids:
            local_values: list[float] = [
                len(self.queues[intersection_id]["N"]),
                len(self.queues[intersection_id]["S"]),
                len(self.queues[intersection_id]["E"]),
                len(self.queues[intersection_id]["W"]),
                self.current_phase[intersection_id],
                self.phase_duration[intersection_id],
            ]
            if self.observation_variant == "full":
                recent_arrivals = self._recent_arrival_means(intersection_id)
                local_values.extend(
                    [
                        float(self._switch_allowed_map()[intersection_id]),
                        self.yellow_remaining[intersection_id],
                        normalized_step,
                        recent_arrivals[0],
                        recent_arrivals[1],
                        recent_arrivals[2],
                        recent_arrivals[3],
                    ]
                )
            values.extend(local_values)

        return np.asarray(values, dtype=np.float32)

    def _build_observation_space(self) -> spaces.Box:
        if self.observation_variant == "minimal":
            local_high = np.asarray(
                [
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    1.0,
                    np.inf,
                ],
                dtype=np.float32,
            )
        else:
            local_high = np.asarray(
                [
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    1.0,
                    np.inf,
                    1.0,
                    float(self.yellow_time),
                    1.0,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                ],
                dtype=np.float32,
            )
        high = np.tile(local_high, self.intersection_count)
        return spaces.Box(
            low=np.zeros(self.observation_dim, dtype=np.float32),
            high=high.astype(np.float32),
            dtype=np.float32,
        )
