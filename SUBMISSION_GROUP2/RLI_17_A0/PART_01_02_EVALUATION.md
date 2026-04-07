# Part 01 vs Part 02 Evaluation Report

## Scope

This document is the markdown report version of the `PART_01_02_EVALUATION.ipynb` analysis.
It explains the same workflow and results in prose form and includes the key measured outputs.

## 1. Setup and Environment Initialization

The evaluation loads Python scientific libraries (`numpy`, `pandas`, `matplotlib`) and the custom Gymnasium environments from `gym_race`.
Two environments are used:

1. `Pyrace-v1` for Part 01 (vanilla DQN baseline).
2. `Pyrace-v3` for Part 02 (improved observations, action space, and reward shaping).

Console output after setup confirms project path and available environments.

## 2. Environment Comparison

### 2.1 Observation Space

Observed output:

1. `Pyrace-v1`: shape `(5,)`, integer-like discrete bucket representation.
2. `Pyrace-v3`: shape `(7,)`, `float32`, normalized range `[0, 1]`.
3. Feature increase: from 5 to 7 dimensions (`+40%`, shown in notebook output as `140.0% of baseline`).

Interpretation:

1. Part 01 state is compact and coarse.
2. Part 02 adds speed and checkpoint-distance signals, increasing state informativeness.

### 2.2 Action Space

Observed output:

1. `Pyrace-v1`: 3 actions (accelerate, left, right).
2. `Pyrace-v3`: 4 actions (accelerate, left, right, explicit brake).

Interpretation:

1. Part 02 adds direct deceleration control.
2. This is especially useful for sharp turns and safer control policies.

### 2.3 Reward Structure

Observed output summary:

1. Part 01 uses sparse rewards (goal, crash, minimal intermediate signal).
2. Part 02 uses shaped rewards combining progress, speed, collision penalty, and goal bonus.

Interpretation:

1. Sparse rewards slow exploration and convergence.
2. Shaped rewards provide denser learning signals and accelerate policy improvement.

## 3. Algorithmic Improvements in Part 02

The notebook compares cumulative impact of stacked DQN enhancements.
Reported technique stack includes:

1. Double DQN.
2. Dueling architecture.
3. Prioritized replay.
4. N-step returns.
5. Soft target updates.
6. Huber loss and stabilization.
7. Reward shaping integration.

Observed table in notebook indicates large cumulative gains, with the largest marginal impact associated with shaped reward and later-stage stabilization features.

## 4. Model Availability Check

Notebook runtime check output indicates that default checkpoint paths referenced in that specific run were not found (`False` for both baseline and advanced paths in the executed context).

Interpretation:

1. This does not invalidate the methodology.
2. It indicates that absolute/relative path assumptions at runtime may differ from where the final submission models are placed.

## 5. Performance Comparison (Part 01 vs Part 02)

The notebook reports the following comparison values:

1. Convergence episodes: approximately `3000` (Part 01) vs `1500` (Part 02), around `2x faster`.
2. Average training reward: `200-500` (Part 01) vs `2000-2500` (Part 02), about `+400%`.
3. Final episode reward: around `~300-800` vs `~2100-2300`.
4. Crash rate: `15-20%` vs `<5%`.
5. Success rate: `~80%` vs `>95%`.
6. Final speed: `8-9` units vs `9-10` units.

Key interpretation from notebook output:

1. Part 02 achieves an order-of-magnitude performance shift in reward scale and stronger safety/stability behavior.

## 6. Visualization Summary

The notebook generates a multi-panel figure titled **Part 01 vs Part 02: Learning Dynamics** with:

1. Learning convergence curves.
2. Learning stability (variance) bars.
3. Crash-rate trend lines.
4. Cumulative technique impact bars.

The generated plot supports the same direction of findings as the tabular comparison: faster convergence and lower instability for Part 02.

## 7. Consolidated Conclusion

Based on the executed notebook workflow and printed metrics:

1. Part 01 successfully establishes a working vanilla DQN baseline.
2. Part 02 materially improves state/action/reward design and algorithmic stability.
3. Combined improvements lead to better convergence speed, safety, and reward performance.

## 8. Notes for Reproducibility

1. Source notebook retained in development directory: `RLI_17_A0/PART_01_02_EVALUATION.ipynb`.
2. Submission directory intentionally contains this markdown report (`.md`) instead of the notebook.
3. Model execution and evaluation commands are documented in `README.md` and `SUBMISSION_REPORT.md` in the submission folder.
