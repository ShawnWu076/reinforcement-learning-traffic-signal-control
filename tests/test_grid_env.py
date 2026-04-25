"""Tests for the 2x2 grid traffic environment."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_rl.baselines import GridMaxPressureController
from traffic_rl.env import KEEP_ACTION, SWITCH_ACTION, build_action_mask
from traffic_rl.grid_env import (
    GridTrafficSignalEnv,
    build_grid_action_mask,
    decode_grid_action,
    encode_grid_action,
)


ZERO_GRID_SCHEDULE = [
    {
        "until_step": 8,
        "rates": {
            "A": {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
            "B": {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
            "C": {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
            "D": {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        },
    }
]


class GridTrafficSignalEnvTest(unittest.TestCase):
    def test_action_encoding_round_trip(self) -> None:
        local_actions = [KEEP_ACTION, SWITCH_ACTION, KEEP_ACTION, SWITCH_ACTION]
        encoded = encode_grid_action(local_actions)

        self.assertEqual(encoded, 10)
        np.testing.assert_array_equal(decode_grid_action(encoded, 4), np.asarray(local_actions))

    def test_reset_returns_grid_observation_and_action_mask(self) -> None:
        env = GridTrafficSignalEnv(
            arrival_schedule=ZERO_GRID_SCHEDULE,
            grid_shape=(2, 2),
            intersection_ids=["A", "B", "C", "D"],
            episode_length=8,
            min_green_time=2,
        )

        observation, info = env.reset(seed=0)

        self.assertEqual(observation.shape, (52,))
        self.assertEqual(env.action_dim, 16)
        self.assertTrue(env.observation_space.contains(observation))
        np.testing.assert_array_equal(
            info["action_mask"],
            build_grid_action_mask([False, False, False, False]),
        )
        np.testing.assert_array_equal(
            build_action_mask(observation, info=info, action_dim=env.action_dim),
            info["action_mask"],
        )

    def test_all_joint_actions_are_valid_after_min_green(self) -> None:
        env = GridTrafficSignalEnv(
            arrival_schedule=ZERO_GRID_SCHEDULE,
            grid_shape=(2, 2),
            intersection_ids=["A", "B", "C", "D"],
            episode_length=8,
            min_green_time=1,
        )

        env.reset(seed=0)
        observation, _, _, _, info = env.step(KEEP_ACTION)

        self.assertEqual(float(np.sum(info["action_mask"])), 16.0)
        self.assertEqual(float(observation[6]), 1.0)

    def test_internal_transfer_moves_vehicle_to_downstream_intersection(self) -> None:
        env = GridTrafficSignalEnv(
            arrival_schedule=ZERO_GRID_SCHEDULE,
            grid_shape=(2, 2),
            intersection_ids=["A", "B", "C", "D"],
            episode_length=4,
            min_green_time=0,
            yellow_time=0,
        )

        env.reset(
            seed=0,
            options={
                "initial_phases": {"A": 1, "B": 1, "C": 0, "D": 0},
                "initial_queues": {"A": {"W": 1}},
            },
        )
        _, _, _, _, info = env.step(KEEP_ACTION)
        summary = env.summarize()

        self.assertEqual(info["queue_lengths_by_intersection"]["B"]["W"], 1)
        self.assertEqual(summary["total_served"], 1.0)
        self.assertEqual(summary["total_departed"], 0.0)
        self.assertEqual(summary["internal_transfer_count"], 1.0)

    def test_grid_max_pressure_returns_joint_action(self) -> None:
        env = GridTrafficSignalEnv(
            arrival_schedule=ZERO_GRID_SCHEDULE,
            grid_shape=(2, 2),
            intersection_ids=["A", "B", "C", "D"],
            episode_length=4,
            min_green_time=0,
        )
        observation, info = env.reset(
            seed=0,
            options={
                "initial_queues": {
                    "A": {"E": 4, "W": 4},
                    "C": {"N": 5, "S": 5},
                }
            },
        )
        controller = GridMaxPressureController(
            min_green=0,
            intersection_count=4,
            observation_variant="full",
        )

        action = controller.act(observation, info=info)
        local_actions = decode_grid_action(action, 4)

        self.assertEqual(local_actions[0], SWITCH_ACTION)
        self.assertEqual(local_actions[2], KEEP_ACTION)


if __name__ == "__main__":
    unittest.main()
