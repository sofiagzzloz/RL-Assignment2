import argparse
import subprocess
import sys
from pathlib import Path


PRESETS = {
    "fast": {
        "episodes": 3000,
        "learning_rate": 3e-4,
        "batch_size": 128,
        "replay_capacity": 120000,
        "learning_starts": 3000,
        "epsilon_decay_episodes": 2200,
        "n_step": 3,
        "normalize_reward": True,
        "model_dir": "dqn_vanilla/models_DQN_adv_fast",
    },
    "stable": {
        "episodes": 4000,
        "learning_rate": 2e-4,
        "batch_size": 128,
        "replay_capacity": 140000,
        "learning_starts": 4000,
        "epsilon_decay_episodes": 3000,
        "n_step": 5,
        "normalize_reward": True,
        "model_dir": "dqn_vanilla/models_DQN_adv_stable",
    },
    "long": {
        "episodes": 6000,
        "learning_rate": 1.5e-4,
        "batch_size": 256,
        "replay_capacity": 180000,
        "learning_starts": 5000,
        "epsilon_decay_episodes": 5000,
        "n_step": 5,
        "normalize_reward": True,
        "model_dir": "dqn_vanilla/models_DQN_adv_long",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch preset advanced DQN experiment"
    )
    parser.add_argument("--preset", choices=["fast", "stable", "long"], default="fast")
    parser.add_argument(
        "--episodes", type=int, default=0, help="Override preset episodes"
    )
    parser.add_argument("--env-id", type=str, default="Pyrace-v3")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    cfg = PRESETS[args.preset].copy()
    if args.episodes > 0:
        cfg["episodes"] = args.episodes

    script = Path(__file__).resolve().parent / "Pyrace_RL_DQN_Advanced.py"

    command = [
        sys.executable,
        str(script),
        "--mode",
        "train",
        "--env-id",
        args.env_id,
        "--episodes",
        str(cfg["episodes"]),
        "--learning-rate",
        str(cfg["learning_rate"]),
        "--batch-size",
        str(cfg["batch_size"]),
        "--replay-capacity",
        str(cfg["replay_capacity"]),
        "--learning-starts",
        str(cfg["learning_starts"]),
        "--epsilon-decay-episodes",
        str(cfg["epsilon_decay_episodes"]),
        "--n-step",
        str(cfg["n_step"]),
        "--model-dir",
        cfg["model_dir"],
    ]

    if cfg["normalize_reward"]:
        command.append("--normalize-reward")

    if args.render:
        command.append("--render")

    print("Launching:")
    print(" ".join(command))

    result = subprocess.run(command, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
