# RLI-17-A0 Assignment Submission

Course: Reinforcement Learning (RLI)
Assignment: 17.00 - Pit lane repairs
Mini-group: 2

## 1. Submission Scope

This submission contains all required components:

1. Part 01: Vanilla DQN migration from the Q-table baseline.
2. Part 02: Environment and algorithm improvements.
3. Bonus: Additional advanced variants, including DDPG and a sharp-turn specialist model.

## 2. Main Deliverables

Top-level files:

1. `SUBMISSION_REPORT.md`: Consolidated explanation for Part 01, Part 02, and Bonus.
2. `PART_01_02_EVALUATION.ipynb`: Notebook for comparisons and evaluation.
3. `Pyrace_performance_analysis.ipynb`: Additional analysis notebook.
4. `Pyrace_performance_analysis.html`: Exported analysis report.
5. `Pyrace_RL_QTable.py`: Original reference baseline.

Implementation directories:

1. `dqn_vanilla/`: DQN, advanced DQN, DDPG, scripts, tests, and selected model checkpoints.
2. `gym_race/`: Environment code and Gymnasium wrappers.

## 3. Part 01

Main script:

1. `dqn_vanilla/Pyrace_RL_DQN.py`

Part 01 includes:

1. Neural network approximation of Q-values.
2. Experience replay.
3. Epsilon-greedy exploration.
4. Train and evaluation modes.

## 4. Part 02

Main script:

1. `dqn_vanilla/Pyrace_RL_DQN_Advanced.py`

Environment updates are implemented in:

1. `gym_race/envs/pyrace_2d.py`
2. `gym_race/envs/race_env.py`
3. `gym_race/__init__.py`

Part 02 includes:

1. Improved observations and action handling.
2. Reward shaping.
3. Advanced DQN techniques for improved convergence and stability.

## 5. Bonus Components

Scripts:

1. `dqn_vanilla/Pyrace_RL_DQN_SharpTurns.py`
2. `dqn_vanilla/Pyrace_RL_DDPG_SB3.py`

Final sharp-turn model directory included in this submission:

1. `dqn_vanilla/models_DQN_sharp_6k_v01/dqn_adv_best.pt`
2. `dqn_vanilla/models_DQN_sharp_6k_v01/dqn_adv_final.pt`
3. `dqn_vanilla/models_DQN_sharp_6k_v01/reward_history_adv.npy`

## 6. Setup

Install dependencies from repository root:

```bash
pip install -r RLI_17_A0/dqn_vanilla/requirements.txt
```

## 7. Verification Commands

Run smoke tests:

```bash
python RLI_17_A0/dqn_vanilla/smoke_test_dqn.py
python RLI_17_A0/dqn_vanilla/smoke_test_ddpg_sb3.py
```

## 8. Evaluation Commands

Evaluate Part 01 model:

```bash
python RLI_17_A0/dqn_vanilla/Pyrace_RL_DQN.py --mode eval --env-id Pyrace-v1 --model-path RLI_17_A0/dqn_vanilla/models_DQN_v03_part2/dqn_best.pt --eval-episodes 5 --render
```

Evaluate Part 02 model:

```bash
python RLI_17_A0/dqn_vanilla/Pyrace_RL_DQN_Advanced.py --mode eval --env-id Pyrace-v3 --model-path RLI_17_A0/dqn_vanilla/models_DQN_v03_part2/dqn_best.pt --eval-episodes 10 --render
```

Evaluate final sharp-turn model:

```bash
python RLI_17_A0/dqn_vanilla/Pyrace_RL_DQN_SharpTurns.py --mode eval --env-id Pyrace-v4 --model-path RLI_17_A0/dqn_vanilla/models_DQN_sharp_6k_v01/dqn_adv_best.pt --eval-episodes 10 --render
```

Evaluate DDPG model:

```bash
python RLI_17_A0/dqn_vanilla/Pyrace_RL_DDPG_SB3.py --mode eval --model-path RLI_17_A0/dqn_vanilla/models_DDPG_sb3_v01/ddpg_best.pt --episodes 5
```
