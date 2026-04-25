# Notebook Status

These notebooks are analysis and presentation layers on top of the JSON files written by the scripts.

Current notebooks:

- `01_project_overview.ipynb`: project-level dashboard for goals, implemented pieces, current results, and next steps.
- `02_baseline_comparison.ipynb`: loads `results/baseline_summary.json` and compares the heuristic controllers across regimes.
- `03_dqn_training_analysis.ipynb`: loads `results/dqn_summary.json`, inspects DQN training behavior, and compares DQN against baselines.

Recommended workflow:

```bash
python3 scripts/run_baselines.py --config configs/default.yaml
python3 scripts/train_dqn.py --config configs/default.yaml
python3 scripts/tune_dqn.py --config configs/default.yaml
```

Then open the notebooks to inspect, explain, and export the results for a report or presentation.
