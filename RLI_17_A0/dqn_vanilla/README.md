# RLI 17 A0 - Vanilla DQN (Part 01)

This folder now includes a **vanilla DQN** solution for `Pyrace-v1`, replacing the tabular Q-table approach with:

- A feed-forward neural network for Q-value approximation
- Experience replay buffer
- Epsilon-greedy exploration
- Optional target network (disabled by default to keep it "vanilla")

## Files

- `Pyrace_RL_DQN.py` → Train/evaluate DQN agent
- `smoke_test_dqn.py` → Tiny sanity harness (2 very short episodes)
- `requirements.txt` → Minimal dependencies for this part

## Setup

Run commands from this folder (`RLI_17_A0/dqn_vanilla`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick smoke test

```bash
python smoke_test_dqn.py
```

## Train

```bash
python Pyrace_RL_DQN.py --mode train --episodes 2000 --max-steps 2000 --model-dir dqn_vanilla/models_DQN_v01
```

Useful options:

- `--render` to show the game
- `--use-target-network` to enable optional target net updates
- `--save-every 100` to checkpoint periodically

## Evaluate

```bash
python Pyrace_RL_DQN.py --mode eval --model-path dqn_vanilla/models_DQN_v01/dqn_best.pt --eval-episodes 10 --render
```

## Notes

- Observations are 5 radar-distance features from the environment.
- Actions are discrete (`0: accelerate`, `1: turn left`, `2: turn right`).
- Rewards are provided by `PyRace2D.evaluate()`.
