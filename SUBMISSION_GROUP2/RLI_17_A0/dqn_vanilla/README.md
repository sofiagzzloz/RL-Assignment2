# RLI 17 A0 - DQN Training (Part 01 + Part 02)

This folder includes:

- A **vanilla DQN** solution for `Pyrace-v1` (Part 01 baseline)
- An **advanced DQN** pipeline for `Pyrace-v3` (Part 02 performance)

Vanilla DQN replaces the tabular Q-table approach with:

- A feed-forward neural network for Q-value approximation
- Experience replay buffer
- Epsilon-greedy exploration
- Optional target network (disabled by default to keep it "vanilla")

Advanced DQN adds multiple techniques aimed at speed and stability:

- Double DQN targets
- Dueling network architecture
- Prioritized Experience Replay (PER)
- n-step returns
- Soft target updates
- Huber loss + gradient clipping
- Optional reward normalization

## Files

- `Pyrace_RL_DQN.py` → Train/evaluate DQN agent
- `Pyrace_RL_DQN_Advanced.py` → Train/evaluate advanced DQN agent
- `run_experiment.py` → One-command preset launcher (`fast`, `stable`, `long`)
- `smoke_test_dqn.py` → Tiny sanity harness (2 very short episodes)
- `requirements.txt` → Minimal dependencies for this part

## Setup

Run commands from this folder (`RLI_17_A0/dqn_vanilla`) or repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r RLI_17_A0\dqn_vanilla\requirements.txt
```

## Quick smoke test

```powershell
python RLI_17_A0\dqn_vanilla\smoke_test_dqn.py
```

## Train

```powershell
python RLI_17_A0\dqn_vanilla\Pyrace_RL_DQN.py --mode train --env-id Pyrace-v1 --episodes 2000 --max-steps 2000 --model-dir dqn_vanilla/models_DQN_v01
```

Useful options:

- `--render` to show the game
- `--use-target-network` to enable optional target net updates
- `--save-every 100` to checkpoint periodically

## Train advanced model for fast driving (Part 02)

Recommended first run:

```powershell
python RLI_17_A0\dqn_vanilla\Pyrace_RL_DQN_Advanced.py --mode train --env-id Pyrace-v3 --episodes 3000 --max-steps 2000 --normalize-reward --model-dir dqn_vanilla/models_DQN_adv_v01
```

Train longer than 2k episodes:

```powershell
python RLI_17_A0\dqn_vanilla\Pyrace_RL_DQN_Advanced.py --mode train --env-id Pyrace-v3 --episodes 6000 --max-steps 2000 --normalize-reward --model-dir dqn_vanilla/models_DQN_adv_6k
```

Resume from a checkpoint:

```powershell
python RLI_17_A0\dqn_vanilla\Pyrace_RL_DQN_Advanced.py --mode train --env-id Pyrace-v3 --episodes 6000 --resume-from dqn_vanilla/models_DQN_adv_6k/dqn_adv_ep_3000.pt --resume-episode 3000 --normalize-reward --model-dir dqn_vanilla/models_DQN_adv_6k
```

Preset launcher (quick way to run experiments):

```powershell
python RLI_17_A0\dqn_vanilla\run_experiment.py --preset fast
python RLI_17_A0\dqn_vanilla\run_experiment.py --preset stable
python RLI_17_A0\dqn_vanilla\run_experiment.py --preset long
```

## Evaluate

```powershell
python RLI_17_A0\dqn_vanilla\Pyrace_RL_DQN.py --mode eval --env-id Pyrace-v1 --model-path dqn_vanilla/models_DQN_v01/dqn_best.pt --eval-episodes 10 --render
```

Evaluate advanced model and watch trained car:

```powershell
python RLI_17_A0\dqn_vanilla\Pyrace_RL_DQN_Advanced.py --mode eval --env-id Pyrace-v3 --model-path dqn_vanilla/models_DQN_adv_v01/dqn_adv_best.pt --eval-episodes 10 --render
```

## Notes

- `Pyrace-v1` uses 5 radar-distance features and 3 actions.
- `Pyrace-v3` uses 7 continuous features (5 radars + speed + checkpoint distance) and 4 actions (adds brake).
- `Pyrace-v3` reward is shaped for progress and speed while still penalizing crashes.
