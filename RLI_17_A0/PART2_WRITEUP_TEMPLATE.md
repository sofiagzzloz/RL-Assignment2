# Assignment 17.00 - Part 01 and Part 02 (What Was Done)

Date updated: 2026-04-02

## Part 01 - Refactor from Q-Table to DQN

Completed work:

- Replaced tabular Q-learning logic with a neural-network DQN agent in [RLI_17_A0/dqn_vanilla/Pyrace_RL_DQN.py](RLI_17_A0/dqn_vanilla/Pyrace_RL_DQN.py).
- Implemented a feed-forward Q-network (MLP) to estimate Q-values from state vectors.
- Implemented replay memory and random minibatch sampling.
- Implemented epsilon-greedy exploration with decay.
- Implemented training and evaluation modes using command-line arguments.
- Added model checkpoint saving/loading (`dqn_best.pt`, periodic checkpoints, `dqn_final.pt`).
- Kept compatibility with the original baseline environment (`Pyrace-v1`).

Code status:

- DQN training pipeline executes end-to-end (verified by smoke test).

## Part 02 - Environment and Learning Improvements

Completed work:

- Added and registered improved environment variant `Pyrace-v3` in [RLI_17_A0/gym_race/**init**.py](RLI_17_A0/gym_race/__init__.py).
- Added `RaceEnvV3` in [RLI_17_A0/gym_race/envs/race_env.py](RLI_17_A0/gym_race/envs/race_env.py).
- Extended simulator behavior in [RLI_17_A0/gym_race/envs/pyrace_2d.py](RLI_17_A0/gym_race/envs/pyrace_2d.py) with:
  - continuous observation mode,
  - extended action mode (including brake),
  - shaped reward mode.

State/action/reward changes in `Pyrace-v3`:

- Observation size changed from 5 discrete radar buckets to 7 continuous features:
  - 5 normalized radar distances,
  - normalized speed,
  - normalized distance to current checkpoint.
- Action space changed from 3 to 4 actions by adding brake.
- Reward changed from sparse crash/goal style to shaped reward with progress, speed term, checkpoint bonus, collision penalty, and goal bonus.

Additional Part 02 model work:

- Implemented advanced DQN pipeline in [RLI_17_A0/dqn_vanilla/Pyrace_RL_DQN_Advanced.py](RLI_17_A0/dqn_vanilla/Pyrace_RL_DQN_Advanced.py) using:
  - Double DQN,
  - Dueling architecture,
  - Prioritized Experience Replay,
  - n-step returns,
  - soft target updates,
  - Huber loss and gradient clipping,
  - optional reward normalization.

## Current Verified Outcome

- Part 01 implementation exists and runs.
- Part 02 implementation exists and runs.
- `Pyrace-v3` evaluation with saved model `models_DQN_v03_part2/dqn_final.pt` produced stable rewards during spot-checking (`2178.18`, `2178.18`).

## Best Model Run Command (visual)

From `RLI_17_A0/dqn_vanilla` with the project virtual environment active:

- `python.exe Pyrace_RL_DQN_SharpTurns.py --mode eval --model-path dqn_vanilla/models_DQN_sharp_6k_v01/dqn_adv_final.pt --eval-episodes 10 --max-steps 1000 --render`
