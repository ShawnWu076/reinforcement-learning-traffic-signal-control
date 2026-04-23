"""Tests for config overrides and tuning helpers."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_rl.config import apply_overrides, parse_override_strings
from traffic_rl.tuning import build_trial_overrides, extract_objective_score, sort_trials


class ConfigAndTuningHelpersTest(unittest.TestCase):
    def test_parse_and_apply_overrides(self) -> None:
        config = {
            "training": {
                "learning_rate": 0.001,
                "hidden_dims": [128, 128],
            }
        }

        overrides = parse_override_strings(
            [
                "training.learning_rate=0.0005",
                "training.hidden_dims=[256, 128]",
            ]
        )
        updated = apply_overrides(config, overrides)

        self.assertEqual(updated["training"]["learning_rate"], 0.0005)
        self.assertEqual(updated["training"]["hidden_dims"], [256, 128])
        self.assertEqual(config["training"]["learning_rate"], 0.001)

    def test_build_trial_overrides_merges_fixed_values(self) -> None:
        tuning_config = {
            "search_type": "grid",
            "fixed_overrides": {"training.episodes": 50},
            "search_space": {
                "training.learning_rate": [0.001, 0.0005],
                "training.gamma": [0.95, 0.99],
            },
        }

        trials = build_trial_overrides(tuning_config)

        self.assertEqual(len(trials), 4)
        self.assertIn("training.episodes", trials[0])
        self.assertIn("training.learning_rate", trials[0])
        self.assertIn("training.gamma", trials[0])

    def test_extract_and_sort_objective_scores(self) -> None:
        summary = {
            "evaluation_results": {
                "nonstationary": {
                    "dqn": {"average_wait_time_seconds": 12.5}
                }
            }
        }

        score = extract_objective_score(
            summary,
            {
                "regime": "nonstationary",
                "policy": "dqn",
                "metric": "average_wait_time_seconds",
            },
        )

        ranked = sort_trials(
            [
                {"trial_id": 1, "objective_score": 14.0},
                {"trial_id": 2, "objective_score": 11.0},
            ],
            mode="min",
        )

        self.assertEqual(score, 12.5)
        self.assertEqual(ranked[0]["trial_id"], 2)


if __name__ == "__main__":
    unittest.main()
