#!/usr/bin/env python3
"""Render JSON experiment summaries as readable text tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _format_float(value: float) -> str:
    return f"{value:.2f}"


def _render_baseline_summary(summary: dict[str, dict[str, dict[str, float]]]) -> str:
    lines = ["Baseline summary", ""]
    for regime_name, policy_results in summary.items():
        lines.append(f"Regime: {regime_name}")
        lines.append("policy            avg_queue  avg_wait_s  throughput  switches  invalid")
        for policy_name, metrics in policy_results.items():
            lines.append(
                f"{policy_name:16s}  "
                f"{_format_float(metrics['average_queue_length']):>9s}  "
                f"{_format_float(metrics['average_wait_time_seconds']):>10s}  "
                f"{_format_float(metrics['throughput_per_step']):>10s}  "
                f"{_format_float(metrics['switch_count']):>8s}  "
                f"{_format_float(metrics.get('invalid_switch_count', 0.0)):>7s}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def _render_dqn_summary(summary: dict[str, object]) -> str:
    metadata = summary.get("metadata", {})
    training_history = summary.get("training_history", [])
    evaluation_results = summary.get("evaluation_results", {})
    checkpoint = summary.get("checkpoint", "")

    lines = ["DQN summary", ""]
    lines.append(f"Checkpoint: {checkpoint}")
    if metadata:
        lines.append(
            "Run metadata: "
            f"study={metadata.get('study_name', 'default')}, "
            f"variant={metadata.get('variant_name', 'default')}, "
            f"seed={metadata.get('seed', 'n/a')}, "
            f"train_schedule={metadata.get('train_schedule_name', 'train_schedule')}, "
            f"obs={metadata.get('observation_variant', 'full')}, "
            f"reward={metadata.get('reward_mode', 'queue')}, "
            f"switch_penalty={_format_float(float(metadata.get('switch_penalty', 0.0)))}, "
            f"double_dqn={metadata.get('double_dqn', 'n/a')}"
        )
    lines.append(f"Training episodes recorded: {len(training_history)}")
    if training_history:
        final_episode = training_history[-1]
        lines.append(
            "Final training episode: "
            f"reward={_format_float(final_episode['total_reward'])}, "
            f"avg_queue={_format_float(final_episode['average_queue_length'])}, "
            f"avg_wait_s={_format_float(final_episode['average_wait_time_seconds'])}, "
            f"epsilon={_format_float(final_episode['epsilon'])}"
        )
    lines.append("")

    if evaluation_results:
        lines.append("Evaluation")
        lines.append("regime              avg_queue  avg_wait_s  throughput  switches  invalid")
        for regime_name, policy_results in evaluation_results.items():
            dqn_metrics = policy_results["dqn"]
            lines.append(
                f"{regime_name:18s}  "
                f"{_format_float(dqn_metrics['average_queue_length']):>9s}  "
                f"{_format_float(dqn_metrics['average_wait_time_seconds']):>10s}  "
                f"{_format_float(dqn_metrics['throughput_per_step']):>10s}  "
                f"{_format_float(dqn_metrics['switch_count']):>8s}  "
                f"{_format_float(dqn_metrics.get('invalid_switch_count', 0.0)):>7s}"
            )
    return "\n".join(lines)


def _render_multiseed_summary(summary: dict[str, object]) -> str:
    metadata = summary.get("metadata", {})
    aggregate = summary.get("aggregate", {})
    seeds = summary.get("seeds", [])
    training = aggregate.get("final_training_episode", {})
    per_regime = aggregate.get("per_regime", {})

    lines = ["Multi-seed DQN summary", ""]
    lines.append(f"Seeds: {seeds}")
    if metadata:
        lines.append(
            "Run metadata: "
            f"network={metadata.get('network_type', 'n/a')}, "
            f"train_schedule={metadata.get('train_schedule_name', 'train_schedule')}, "
            f"double_dqn={metadata.get('double_dqn', 'n/a')}, "
            f"runs={metadata.get('run_count', len(seeds))}"
        )
    if training:
        lines.append(
            "Final training episode mean/std: "
            f"reward={_format_float(training['total_reward']['mean'])}"
            f"+/-{_format_float(training['total_reward']['std'])}, "
            f"avg_queue={_format_float(training['average_queue_length']['mean'])}"
            f"+/-{_format_float(training['average_queue_length']['std'])}, "
            f"avg_wait_s={_format_float(training['average_wait_time_seconds']['mean'])}"
            f"+/-{_format_float(training['average_wait_time_seconds']['std'])}"
        )
    lines.append("")

    if per_regime:
        lines.append("DQN evaluation mean/std")
        lines.append("regime              avg_queue       avg_wait_s      throughput      switches")
        for regime_name, policy_results in per_regime.items():
            dqn_metrics = policy_results["dqn"]
            lines.append(
                f"{regime_name:18s}  "
                f"{_format_float(dqn_metrics['average_queue_length']['mean']):>6s}"
                f"+/-{_format_float(dqn_metrics['average_queue_length']['std']):<6s}  "
                f"{_format_float(dqn_metrics['average_wait_time_seconds']['mean']):>6s}"
                f"+/-{_format_float(dqn_metrics['average_wait_time_seconds']['std']):<6s}  "
                f"{_format_float(dqn_metrics['throughput_per_step']['mean']):>6s}"
                f"+/-{_format_float(dqn_metrics['throughput_per_step']['std']):<6s}  "
                f"{_format_float(dqn_metrics['switch_count']['mean']):>6s}"
                f"+/-{_format_float(dqn_metrics['switch_count']['std']):<6s}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=str, help="Path to a JSON output from baseline or DQN scripts.")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    with summary_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if "aggregate" in payload and "runs" in payload:
        print(_render_multiseed_summary(payload))
    elif "training_history" in payload and "evaluation_results" in payload:
        print(_render_dqn_summary(payload))
    else:
        print(_render_baseline_summary(payload))


if __name__ == "__main__":
    main()
