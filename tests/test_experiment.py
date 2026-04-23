"""Smoke test for the reusable experiment runner."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_rl.experiment import run_dqn_experiment


SMALL_SCHEDULE = [
    {
        "until_step": 6,
        "rates": {"N": 0.6, "S": 0.6, "E": 0.4, "W": 0.4},
    }
]


class ExperimentRunnerTest(unittest.TestCase):
    def test_run_dqn_experiment_returns_expected_summary(self) -> None:
        config = {
            "environment": {
                "train_schedule_name": "tiny_train",
                "episode_length": 6,
                "step_seconds": 3,
                "min_green_time": 1,
                "yellow_time": 1,
                "max_departures_per_step": 2,
                "recent_arrival_window": 3,
                "observation_variant": "full",
                "reward_mode": "queue",
                "switch_penalty": 1.0,
                "train_schedule": SMALL_SCHEDULE,
                "evaluation_regimes": {
                    "tiny_eval": SMALL_SCHEDULE,
                },
            },
            "training": {
                "episodes": 3,
                "gamma": 0.95,
                "learning_rate": 0.001,
                "batch_size": 2,
                "buffer_size": 32,
                "hidden_dims": [16],
                "start_epsilon": 0.8,
                "end_epsilon": 0.1,
                "epsilon_decay_steps": 12,
                "warmup_steps": 1,
                "update_frequency": 1,
                "target_sync_steps": 4,
                "seed": 3,
                "device": "cpu",
                "log_interval_episodes": 10,
            },
            "evaluation": {
                "episodes_per_regime": 2,
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "tiny_checkpoint.pt"
            summary = run_dqn_experiment(
                config=config,
                checkpoint_path=checkpoint_path,
                verbose=False,
            )

            self.assertTrue(checkpoint_path.exists())

        self.assertEqual(len(summary["training_history"]), 3)
        self.assertIn("training_overview", summary)
        self.assertIn("tiny_eval", summary["evaluation_results"])


if __name__ == "__main__":
    unittest.main()
