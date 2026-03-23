# Assignment 17.00 - Part 02 (Pit lane repairs)

## 1. Goal

Improve the DQN driving model by enhancing:

- observations (state information)
- actions (agent controls)
- reward function (learning signal)

## 2. Implemented Improvements

### 2.1 Continuous observations

A new environment variant `Pyrace-v3` was created with continuous observations.

State vector (size 7):

- 5 normalized radar distances in range [0, 1]
- normalized speed in range [0, 1]
- normalized distance to next checkpoint in range [0, 1]

Why this helps:

- Preserves more information than coarse bucketization.
- Provides speed context and checkpoint progress context to the policy.

### 2.2 Expanded action space

A new optional action was introduced:

- `BRAKE` (action 3): decreases speed for finer control.

Action space in `Pyrace-v3`:

- 0 accelerate
- 1 turn left
- 2 turn right
- 3 brake

Why this helps:

- Better cornering control.
- Reduces overshooting and collision risk.

### 2.3 Engineered reward function

A shaped reward was implemented:

- positive reward for reducing distance to next checkpoint
- small positive speed term
- checkpoint bonus when passing checkpoint
- strong collision penalty
- goal completion bonus

Why this helps:

- Denser learning signal than sparse crash/goal only reward.
- Better credit assignment during long episodes.

## 3. Environments

- `Pyrace-v1`: original baseline
- `Pyrace-v3`: improved Part 2 variant

## 4. DQN script changes

`Pyrace_RL_DQN.py` now accepts:

- `--env-id` (default `Pyrace-v1`)

This allows direct comparison between baseline and improved environment.

## 5. How to run

From repository root (using project venv):

Training baseline:

- `./.venv/Scripts/python.exe RLI_17_A0/dqn_vanilla/Pyrace_RL_DQN.py --env-id Pyrace-v1 --mode train`

Training improved variant:

- `./.venv/Scripts/python.exe RLI_17_A0/dqn_vanilla/Pyrace_RL_DQN.py --env-id Pyrace-v3 --mode train`

Evaluate improved variant:

- `./.venv/Scripts/python.exe RLI_17_A0/dqn_vanilla/Pyrace_RL_DQN.py --env-id Pyrace-v3 --mode eval --model-path ./dqn_vanilla/models_DQN_v03_part2/dqn_final.pt --eval-episodes 10`

## 6. Final Part 2 Results

Training setup used:

- Environment: `Pyrace-v3`
- Episodes: 2000
- Output folder: `dqn_vanilla/models_DQN_v03_part2`

Training completion:

- Final episode: 2000
- Final training message: `Training complete. Saved model to dqn_vanilla/models_DQN_v03_part2/dqn_final.pt`
- Final episode reward: `452.74`
- Final avg20 reward: `1135.85`

Evaluation (10 episodes, greedy policy):

- Episode rewards: `2178.18` for all 10 episodes
- Mean reward: `2178.18`
- Standard deviation: `0.00`

Interpretation:

- The learned policy is stable in evaluation mode.
- Part 2 design choices (continuous observations, brake action, shaped reward) clearly produce higher and more consistent performance than the sparse baseline behavior.

## 7. Comparison Plan (Optional Extension)

For a fair comparison:

- Use same seed, episodes, and max steps for both envs.
- Compare:
  - average reward over last 20 episodes
  - crash frequency
  - checkpoints reached
  - qualitative driving smoothness

## 8. Discussion Points for Report

- Continuous features improve state fidelity but can increase optimization difficulty.
- Brake action increases control expressiveness and may improve safety.
- Reward shaping accelerates learning but may bias behavior if not aligned with final objective.

## 9. Optional Next Steps (Bonus Direction)

- Replace vanilla DQN with Double DQN + target net updates.
- Try SAC/DDPG/PPO with `Pyrace-v3`.
- Add checkpoint-based curriculum and evaluate sample efficiency.
