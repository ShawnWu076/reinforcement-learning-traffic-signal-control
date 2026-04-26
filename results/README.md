# Results

This directory stores generated experiment artifacts.

Expected outputs:

- `baseline_summary.json` from `scripts/run_baselines.py`
- `dqn_summary.json` from `scripts/train_dqn.py`
- `dqn_multiseed_summary.json` from `scripts/train_dqn.py --seeds 7,17,27`
- `checkpoints/dqn_policy.pt` from `scripts/train_dqn.py`
- `multiseed/seed_*/dqn_summary.json` from multi-seed DQN runs
- `ablations/ablation_summary.json` from `scripts/run_ablations.py`
- `../figures/*.png` from `scripts/plot_ablations.py`

Checkpoints created before the 13D observation update are not compatible with
the current model input shape and should be regenerated.

Quick inspection:

```bash
python3 scripts/summarize_results.py results/baseline_summary.json
python3 scripts/summarize_results.py results/dqn_summary.json
```

Multi-seed DQN:

```bash
python3 scripts/train_dqn.py --config configs/default.yaml --seeds 7,17,27 --no-plots
```

Ablation workflow:

```bash
python3 scripts/run_ablations.py --config configs/ablations.yaml
python3 scripts/plot_ablations.py results/ablations/ablation_summary.json
```

The default ablation config now includes reward, state, switch-penalty, and
train-distribution generalization comparisons.
