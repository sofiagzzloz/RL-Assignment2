import subprocess
import sys
from pathlib import Path


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    train_script = base_dir / "Pyrace_RL_DQN.py"
    model_dir = base_dir / "models_DQN_smoke"

    command = [
        sys.executable,
        str(train_script),
        "--mode",
        "train",
        "--episodes",
        "2",
        "--max-steps",
        "40",
        "--learning-starts",
        "8",
        "--batch-size",
        "8",
        "--replay-capacity",
        "256",
        "--save-every",
        "2",
        "--model-dir",
        str(model_dir),
    ]
    completed = subprocess.run(command, check=False, cwd=str(base_dir))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
