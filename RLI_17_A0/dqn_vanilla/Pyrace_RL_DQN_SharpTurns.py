import argparse
from types import SimpleNamespace

from Pyrace_RL_DQN_Advanced import evaluate, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sharp-turn focused DQN training for Pyrace-v4"
    )
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--env-id", type=str, default="Pyrace-v4")

    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=2000)

    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=3000)

    parser.add_argument("--replay-capacity", type=int, default=140000)
    parser.add_argument("--learning-starts", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument("--n-step", type=int, default=5)
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta-start", type=float, default=0.4)
    parser.add_argument("--per-beta-frames", type=int, default=400000)

    parser.add_argument("--soft-tau", type=float, default=0.005)
    parser.add_argument("--grad-clip", type=float, default=10.0)

    parser.add_argument("--normalize-reward", action="store_true")

    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-every", type=int, default=100)

    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--model-dir", type=str, default="dqn_vanilla/models_DQN_sharp_v01"
    )
    parser.add_argument("--model-path", type=str, default="")

    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument("--resume-episode", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def to_adv_namespace(cli: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        env_id=cli.env_id,
        mode=cli.mode,
        episodes=cli.episodes,
        eval_episodes=cli.eval_episodes,
        max_steps=cli.max_steps,
        learning_rate=cli.learning_rate,
        gamma=cli.gamma,
        epsilon_start=cli.epsilon_start,
        epsilon_end=cli.epsilon_end,
        epsilon_decay_episodes=cli.epsilon_decay_episodes,
        replay_capacity=cli.replay_capacity,
        learning_starts=cli.learning_starts,
        batch_size=cli.batch_size,
        n_step=cli.n_step,
        per_alpha=cli.per_alpha,
        per_beta_start=cli.per_beta_start,
        per_beta_frames=cli.per_beta_frames,
        soft_tau=cli.soft_tau,
        grad_clip=cli.grad_clip,
        normalize_reward=cli.normalize_reward,
        render=cli.render,
        render_every=cli.render_every,
        save_every=cli.save_every,
        model_dir=cli.model_dir,
        model_path=cli.model_path,
        resume_from=cli.resume_from,
        resume_episode=cli.resume_episode,
        seed=cli.seed,
    )


if __name__ == "__main__":
    args = parse_args()
    cfg = to_adv_namespace(args)
    if args.mode == "train":
        train(cfg)
    else:
        evaluate(cfg)
