# RL Assignment 2 - Master Documentation

This is the single consolidated guide for the full assignment.

It merges and normalizes content from:

- README and quick-reference docs
- Part 01 and Part 02 implementation/writeup docs
- Bonus DDPG docs and comparison reports
- Final status/submission summaries

Use this file as the primary reference.

## 1. Project Scope

This repository contains:

- Part 01: Vanilla DQN baseline for racing
- Part 02: Advanced DQN with improved environment and learning techniques
- Bonus: DDPG with Stable-Baselines3 for continuous-control style behavior

Primary folder:

- `RLI_17_A0/`

Core code:

- `RLI_17_A0/dqn_vanilla/Pyrace_RL_DQN.py`
- `RLI_17_A0/dqn_vanilla/Pyrace_RL_DQN_Advanced.py`
- `RLI_17_A0/dqn_vanilla/Pyrace_RL_DDPG_SB3.py`
- `RLI_17_A0/dqn_vanilla/Pyrace_RL_DQN_SharpTurns.py`

Environment code:

- `RLI_17_A0/gym_race/envs/race_env.py`
- `RLI_17_A0/gym_race/envs/pyrace_2d.py`

## 2. What Was Implemented

### 2.1 Part 01: Vanilla DQN

Implemented:

- Feed-forward Q-network
- Replay buffer
- Epsilon-greedy exploration with decay
- Train/eval CLI flow
- Model checkpoint save/load

Typical reported behavior:

- Convergence around ~3000 episodes
- Reward range around ~200-500 in baseline runs

### 2.2 Part 02: Advanced DQN + Environment Improvements

Environment upgrades (`Pyrace-v3`):

- Observation: 5 discrete-style radar features -> 7 continuous features
- Added speed and checkpoint distance signals
- Action space: +brake action (4 actions total)
- Reward shaping: denser progress/speed feedback and penalties

Advanced learning techniques (documented across reports):

- Double DQN
- Dueling architecture
- Prioritized replay
- N-step returns
- Soft target updates
- Huber loss and gradient clipping
- Reward normalization

Typical reported behavior:

- Faster convergence (~1500 episodes in many runs)
- Higher rewards (commonly around ~2100+ in best runs)

### 2.3 Bonus: DDPG (Stable-Baselines3)

Implemented:

- SB3 DDPG training/evaluation pipeline
- Continuous-action wrapper mapped onto race controls
- Saved models and evaluation flow

Typical reported behavior:

- Very fast wall-clock training in documented runs
- Final evaluation values reported around ~2021.96 in strongest run
- Additional docs discuss stability-performance trade-offs from hyperparameter changes

## 3. Unified Setup

Run from repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r RLI_17_A0\dqn_vanilla\requirements.txt
```

Optional quick environment check:

```powershell
python -c "import torch, gymnasium, pygame, numpy; print('ok')"
```

## 4. How To Run Everything

### 4.1 Smoke tests

```powershell
python RLI_17_A0\dqn_vanilla\smoke_test_dqn.py
python RLI_17_A0\dqn_vanilla\smoke_test_ddpg_sb3.py
```

### 4.2 Train Part 01 (Vanilla DQN)

```powershell
python RLI_17_A0\dqn_vanilla\Pyrace_RL_DQN.py --mode train --env-id Pyrace-v1 --episodes 3000 --max-steps 2000 --model-dir RLI_17_A0\dqn_vanilla\models_DQN_v01
```

### 4.3 Evaluate Part 01

```powershell
python RLI_17_A0\dqn_vanilla\Pyrace_RL_DQN.py --mode eval --env-id Pyrace-v1 --model-path RLI_17_A0\dqn_vanilla\models_DQN_v01\dqn_best.pt --eval-episodes 10 --render
```

### 4.4 Train Part 02 (Advanced DQN)

Recommended:

```powershell
python RLI_17_A0\dqn_vanilla\Pyrace_RL_DQN_Advanced.py --mode train --env-id Pyrace-v3 --episodes 3000 --max-steps 2000 --normalize-reward --model-dir RLI_17_A0\dqn_vanilla\models_DQN_v03_part2
```

Longer run:

```powershell
python RLI_17_A0\dqn_vanilla\Pyrace_RL_DQN_Advanced.py --mode train --env-id Pyrace-v3 --episodes 6000 --max-steps 2000 --normalize-reward --model-dir RLI_17_A0\dqn_vanilla\models_DQN_adv_6k
```

### 4.5 Evaluate Part 02

```powershell
python RLI_17_A0\dqn_vanilla\Pyrace_RL_DQN_Advanced.py --mode eval --env-id Pyrace-v3 --model-path RLI_17_A0\dqn_vanilla\models_DQN_v03_part2\dqn_adv_best.pt --eval-episodes 10 --render
```

### 4.6 Bonus: Train DDPG

```powershell
python RLI_17_A0\dqn_vanilla\Pyrace_RL_DDPG_SB3.py --mode train --timesteps 30000 --model-dir RLI_17_A0\dqn_vanilla\models_DDPG_sb3_v01
```

### 4.7 Bonus: Evaluate DDPG

```powershell
python RLI_17_A0\dqn_vanilla\Pyrace_RL_DDPG_SB3.py --mode eval --model-path RLI_17_A0\dqn_vanilla\models_DDPG_sb3_v01\ddpg_best.pt --episodes 5
```

### 4.8 Bonus: Compare DDPG vs DQN

```powershell
python RLI_17_A0\dqn_vanilla\Pyrace_RL_DDPG_SB3.py --mode compare
```

### 4.9 Run preset experiments

```powershell
python RLI_17_A0\dqn_vanilla\run_experiment.py --preset fast
python RLI_17_A0\dqn_vanilla\run_experiment.py --preset stable
python RLI_17_A0\dqn_vanilla\run_experiment.py --preset long
```

## 5. Analysis and Visualization

Notebook-based comparison:

```powershell
jupyter notebook RLI_17_A0\PART_01_02_EVALUATION.ipynb
```

TensorBoard examples:

```powershell
tensorboard --logdir RLI_17_A0\dqn_vanilla\models_DQN_v03_part2
tensorboard --logdir RLI_17_A0\dqn_vanilla\models_DDPG_sb3_v01\tb_logs
```

## 6. Practical Interpretation of Results

Use this summary when writing final conclusions:

- Part 01 validates baseline DQN implementation correctness.
- Part 02 demonstrates the largest performance gains from richer observations and shaped rewards, plus advanced DQN stabilization techniques.
- Bonus DDPG demonstrates a different control-learning style with strong speed/simplicity trade-offs.
- DDPG improvement reports indicate a common RL trade-off: aggressive settings can produce higher peaks; conservative settings can produce smoother training.

## 6.1 Current Best Checkpoint (Sharp-Turns Variant)

For submission/demo consistency, the current best checkpoint to use is:

- `RLI_17_A0/dqn_vanilla/models_DQN_sharp_6k_v01/dqn_adv_final.pt`

Recent evaluation run on this checkpoint produced consistently strong rewards
(~4246 per episode over the observed rendered episodes).

Run evaluation (recommended command):

```powershell
python RLI_17_A0\dqn_vanilla\Pyrace_RL_DQN_SharpTurns.py --mode eval --env-id Pyrace-v4 --model-path RLI_17_A0\dqn_vanilla\models_DQN_sharp_6k_v01\dqn_adv_final.pt --eval-episodes 10 --render
```

If you want to train again into the same folder:

```powershell
python RLI_17_A0\dqn_vanilla\Pyrace_RL_DQN_SharpTurns.py --mode train --env-id Pyrace-v4 --episodes 6000 --max-steps 2000 --normalize-reward --model-dir RLI_17_A0\dqn_vanilla\models_DQN_sharp_6k_v01
```

## 7. Troubleshooting

If pygame/display issues happen on headless systems:

- Run training without render flags.
- Keep evaluation render optional.

If learning is unstable:

- Lower learning rate.
- Increase training steps.
- Use checkpoint resume and compare multiple seeds.

If model path errors occur:

- Confirm the model file exists in the selected `--model-dir`.
- Use absolute or repository-root-relative paths consistently.

## 8. Document Consolidation Notes

This file supersedes the previously scattered Markdown set as the primary run/explanation guide.

If you want an even cleaner repo later, the remaining status/comparison docs can also be archived into a `docs/archive/` folder.
