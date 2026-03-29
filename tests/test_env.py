"""Smoke tests for the traffic RL starter project."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_rl.baselines import FixedCycleController, MaxPressureController
from traffic_rl.env import AdaptiveTrafficSignalEnv, KEEP_ACTION, SWITCH_ACTION


ZERO_SCHEDULE = [
    {
        "until_step": 5,
        "rates": {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
    }
]


class AdaptiveTrafficSignalEnvTest(unittest.TestCase):
    def test_reset_returns_expected_observation_shape(self) -> None:
        env = AdaptiveTrafficSignalEnv(arrival_schedule=ZERO_SCHEDULE, episode_length=5)
        observation, info = env.reset(seed=0)

        self.assertEqual(observation.shape, (10,))
        self.assertEqual(info["queue_length"], 0)
        self.assertEqual(info["phase"], 0)

    def test_episode_finishes_after_expected_steps(self) -> None:
        env = AdaptiveTrafficSignalEnv(
            arrival_schedule=[
                {
                    "until_step": 3,
                    "rates": {"N": 1.0, "S": 0.0, "E": 0.0, "W": 0.0},
                }
            ],
            episode_length=3,
        )

        env.reset(seed=0)
        done = False
        steps = 0

        while not done:
            _, _, done, _ = env.step(KEEP_ACTION)
            steps += 1

        summary = env.summarize()
        self.assertEqual(steps, 3)
        self.assertIn("average_queue_length", summary)
        self.assertGreaterEqual(summary["maximum_queue_length"], 0.0)

    def test_switch_action_changes_phase(self) -> None:
        env = AdaptiveTrafficSignalEnv(
            arrival_schedule=ZERO_SCHEDULE,
            episode_length=2,
            yellow_time=1,
        )

        env.reset(seed=0)
        self.assertEqual(env.current_phase, 0)
        _, _, _, info = env.step(SWITCH_ACTION)

        self.assertEqual(env.current_phase, 1)
        self.assertEqual(info["switch_cooldown"], 0)

    def test_fixed_cycle_switches_when_cycle_reached(self) -> None:
        controller = FixedCycleController(cycle_length=3)
        observation = np.asarray([0, 0, 0, 0, 0, 3, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.assertEqual(controller.act(observation), SWITCH_ACTION)

    def test_max_pressure_prefers_busier_direction(self) -> None:
        controller = MaxPressureController(min_green=2)
        observation = np.asarray([1, 1, 5, 5, 0, 4, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.assertEqual(controller.act(observation), SWITCH_ACTION)


if __name__ == "__main__":
    unittest.main()
