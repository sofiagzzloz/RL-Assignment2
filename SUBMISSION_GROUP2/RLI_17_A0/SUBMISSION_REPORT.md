# RL Assignment 2 - Consolidated Submission Report

Mini-group: 2
Assignment: RLI 17.00 - Pit lane repairs

## 1. Overview

This submission implements all required parts:

1. Part 01: Q-Table baseline migrated to a vanilla DQN using a neural network and replay buffer.
2. Part 02: Model/environment improvements (state, action, reward) with analysis and tested implementation.
3. Bonus: Migration to DDPG using Stable-Baselines3 on a continuous-action wrapper.

Core code is located in `dqn_vanilla/` and the environment in `gym_race/`.

## 2. Part 01 - Vanilla DQN

### 2.1 Objective

Replace tabular Q-learning with a deep Q-network while keeping the same task and workflow.

### 2.2 Implementation

Main file: `dqn_vanilla/Pyrace_RL_DQN.py`

Implemented components:

1. Feed-forward Q-network (PyTorch) mapping state to Q-values.
2. Experience replay buffer for decorrelated training samples.
3. Epsilon-greedy exploration with decay.
4. Training/evaluation CLI modes.

No mandatory target-network dependency is required by the assignment. The script remains a valid vanilla DQN implementation with replay.

### 2.3 Validation

Smoke test file: `dqn_vanilla/smoke_test_dqn.py`

Result: training/evaluation path executes successfully in short-run smoke validation.

## 3. Part 02 - Improvements

### 3.1 Objective

Discuss and implement feasible improvements to observations, actions, and rewards to improve driving performance and learning stability.

### 3.2 Implemented Improvements

Main file: `dqn_vanilla/Pyrace_RL_DQN_Advanced.py`

Algorithmic improvements included:

1. Double DQN target selection.
2. Dueling architecture.
3. Prioritized replay.
4. N-step returns.
5. Soft target updates.
6. Robust loss and stabilization techniques.

Environment-side improvements:

1. Extended action space including brake support.
2. Continuous normalized observations (radars, speed, checkpoint distance).
3. Reward shaping aligned with progress/safety.

Environment code:

1. `gym_race/envs/pyrace_2d.py`
2. `gym_race/envs/race_env.py`
3. `gym_race/__init__.py`

### 3.3 Validation

The advanced setup is integrated and trainable through the provided script and model checkpoints in the submission.

## 4. Bonus - DDPG Migration

### 4.1 Objective

Use a more advanced algorithm suitable for continuous control in a Gymnasium-compatible environment.

### 4.2 Implementation

Main file: `dqn_vanilla/Pyrace_RL_DDPG_SB3.py`

Implemented components:

1. Continuous action wrapper over PyRace environment.
2. DDPG agent via Stable-Baselines3.
3. Train/eval/compare workflow.
4. Model save/load and evaluation path.

### 4.3 Validation

Smoke test file: `dqn_vanilla/smoke_test_ddpg_sb3.py`

Result: 4/4 smoke checks passing (environment creation, model initialization, short training, save/load).

## 5. Included Files

Essential code and report files included in this zip:

1. `README.md` (project overview)
2. `SUBMISSION_REPORT.md` (this consolidated report)
3. `PART_01_02_EVALUATION.ipynb`
4. `Pyrace_performance_analysis.ipynb`
5. `Pyrace_performance_analysis.html`
6. `Pyrace_RL_QTable.py` (baseline reference)
7. `dqn_vanilla/` (DQN, advanced DQN, DDPG, tests, requirements)
8. `gym_race/` (environment implementation)
9. Selected best model checkpoints for reproducibility

Included final sharp-turn model directory:

1. `dqn_vanilla/models_DQN_sharp_6k_v01/dqn_adv_best.pt`
2. `dqn_vanilla/models_DQN_sharp_6k_v01/dqn_adv_final.pt`

## 6. How To Run

Install dependencies:

```bash
pip install -r RLI_17_A0/dqn_vanilla/requirements.txt
```

Quick checks:

```bash
python RLI_17_A0/dqn_vanilla/smoke_test_dqn.py
python RLI_17_A0/dqn_vanilla/smoke_test_ddpg_sb3.py
```

Evaluate final sharp-turn model:

```bash
python RLI_17_A0/dqn_vanilla/Pyrace_RL_DQN_SharpTurns.py --mode eval --env-id Pyrace-v4 --model-path RLI_17_A0/dqn_vanilla/models_DQN_sharp_6k_v01/dqn_adv_best.pt --eval-episodes 10 --render
```

## 7. Notes

1. The submission was cleaned to avoid redundant duplicated markdown documents.
2. Explanations for Part 01, Part 02, and Bonus are consolidated here for easier grading.
3. Additional training checkpoints were intentionally reduced to keep archive size manageable while preserving reproducibility.
