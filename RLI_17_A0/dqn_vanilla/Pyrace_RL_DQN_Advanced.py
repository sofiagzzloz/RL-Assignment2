import argparse
import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Tuple
import sys

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

os.chdir(PROJECT_DIR)

import gym_race

_ = gym_race


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class RunningNorm:
    def __init__(self, eps: float = 1e-5) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x: float) -> None:
        old_mean = self.mean
        self.count += 1.0
        self.mean += (x - self.mean) / self.count
        self.var += (x - old_mean) * (x - self.mean)

    def normalize(self, x: float) -> float:
        std = max(1e-6, np.sqrt(self.var / max(1.0, self.count - 1.0)))
        return float((x - self.mean) / std)


class NStepBuffer:
    def __init__(self, n_step: int, gamma: float) -> None:
        self.n_step = max(1, n_step)
        self.gamma = gamma
        self.buffer: Deque[Transition] = deque(maxlen=self.n_step)

    def _aggregate(self) -> Optional[Transition]:
        if len(self.buffer) == 0:
            return None

        state = self.buffer[0].state
        action = self.buffer[0].action
        reward = 0.0
        next_state = self.buffer[-1].next_state
        done = self.buffer[-1].done

        for idx, tr in enumerate(self.buffer):
            reward += (self.gamma**idx) * tr.reward
            if tr.done:
                next_state = tr.next_state
                done = True
                break

        return Transition(
            state=state,
            action=action,
            reward=float(reward),
            next_state=next_state,
            done=done,
        )

    def append(self, transition: Transition) -> Optional[Transition]:
        self.buffer.append(transition)
        if len(self.buffer) < self.n_step:
            return None
        out = self._aggregate()
        self.buffer.popleft()
        return out

    def flush(self) -> List[Transition]:
        out: List[Transition] = []
        while len(self.buffer) > 0:
            agg = self._aggregate()
            if agg is not None:
                out.append(agg)
            self.buffer.popleft()
        return out


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 400_000,
        eps: float = 1e-5,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = max(1, beta_frames)
        self.eps = eps

        self.buffer: List[Transition] = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.frames = 1

    def __len__(self) -> int:
        return len(self.buffer)

    def beta(self) -> float:
        return min(
            1.0,
            self.beta_start
            + (1.0 - self.beta_start) * (self.frames / self.beta_frames),
        )

    def add(self, transition: Transition) -> None:
        max_prio = self.priorities.max() if len(self.buffer) > 0 else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]

        probs = prios**self.alpha
        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs_sum

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        beta_t = self.beta()
        self.frames += 1

        weights = (len(self.buffer) * probs[indices]) ** (-beta_t)
        weights = weights / max(1e-6, weights.max())

        states = np.array([t.state for t in samples], dtype=np.float32)
        actions = np.array([t.action for t in samples], dtype=np.int64)
        rewards = np.array([t.reward for t in samples], dtype=np.float32)
        next_states = np.array([t.next_state for t in samples], dtype=np.float32)
        dones = np.array([t.done for t in samples], dtype=np.float32)

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = float(abs(err) + self.eps)


class DuelingQNetwork(nn.Module):
    def __init__(
        self, state_size: int, action_size: int, hidden_size: int = 256
    ) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        value = self.value_stream(feat)
        adv = self.adv_stream(feat)
        q = value + (adv - adv.mean(dim=1, keepdim=True))
        return q


class AdvancedDQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: int,
        replay_capacity: int,
        batch_size: int,
        n_step: int,
        per_alpha: float,
        per_beta_start: float,
        per_beta_frames: int,
        tau: float,
        grad_clip: float,
        seed: int,
    ) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.grad_clip = grad_clip

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = max(1, epsilon_decay)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online_net = DuelingQNetwork(state_size, action_size).to(self.device)
        self.target_net = DuelingQNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        self.buffer = PrioritizedReplayBuffer(
            capacity=replay_capacity,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_frames=per_beta_frames,
        )

        self.n_step = max(1, n_step)
        self.nstep_buffer = NStepBuffer(n_step=self.n_step, gamma=self.gamma)

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        if (not eval_mode) and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def decay_epsilon(self) -> None:
        decay_amount = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay
        self.epsilon = max(self.epsilon_end, self.epsilon - decay_amount)

    def _soft_update(self) -> None:
        for target_param, online_param in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + online_param.data * self.tau
            )

    def remember(self, transition: Transition) -> None:
        reduced = self.nstep_buffer.append(transition)
        if reduced is not None:
            self.buffer.add(reduced)

        if transition.done:
            for extra in self.nstep_buffer.flush():
                self.buffer.add(extra)

    def optimize(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            weights,
        ) = self.buffer.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(
            actions, dtype=torch.int64, device=self.device
        ).unsqueeze(1)
        rewards_t = torch.tensor(
            rewards, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        next_states_t = torch.tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        dones_t = torch.tensor(
            dones, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        weights_t = torch.tensor(
            weights, dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        q_values = self.online_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            next_actions = self.online_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions)
            n_gamma = self.gamma**self.n_step
            targets = rewards_t + n_gamma * (1.0 - dones_t) * next_q

        td_errors = (targets - q_values).detach().squeeze(1).cpu().numpy()
        self.buffer.update_priorities(indices, td_errors)

        loss_each = self.loss_fn(q_values, targets)
        loss = (weights_t * loss_each).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self._soft_update()

        return float(loss.item())

    def save(self, model_path: str) -> None:
        payload = {
            "model_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }
        torch.save(payload, model_path)

    def load(self, model_path: str) -> None:
        payload = torch.load(model_path, map_location=self.device)
        self.online_net.load_state_dict(payload["model_state_dict"])
        target_state = payload.get("target_state_dict", payload["model_state_dict"])
        self.target_net.load_state_dict(target_state)

        if "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])

        self.epsilon = float(payload.get("epsilon", self.epsilon_end))


def train(cfg: argparse.Namespace) -> None:
    env = gym.make(cfg.env_id).unwrapped
    env.set_view(cfg.render)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = AdvancedDQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=cfg.learning_rate,
        gamma=cfg.gamma,
        epsilon_start=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end,
        epsilon_decay=cfg.epsilon_decay_episodes,
        replay_capacity=cfg.replay_capacity,
        batch_size=cfg.batch_size,
        n_step=cfg.n_step,
        per_alpha=cfg.per_alpha,
        per_beta_start=cfg.per_beta_start,
        per_beta_frames=cfg.per_beta_frames,
        tau=cfg.soft_tau,
        grad_clip=cfg.grad_clip,
        seed=cfg.seed,
    )

    os.makedirs(cfg.model_dir, exist_ok=True)

    start_episode = 1
    if cfg.resume_from:
        if not os.path.exists(cfg.resume_from):
            raise FileNotFoundError(
                f"Cannot resume, model not found: {cfg.resume_from}"
            )
        agent.load(cfg.resume_from)
        print(f"Resumed model from: {cfg.resume_from}")
        start_episode = cfg.resume_episode + 1

    reward_norm = RunningNorm()
    best_reward = -float("inf")
    reward_history: List[float] = []

    for episode in range(start_episode, cfg.episodes + 1):
        state, _ = env.reset()
        state = state.astype(np.float32)

        episode_reward = 0.0
        losses: List[float] = []

        for step in range(cfg.max_steps):
            action = agent.select_action(state, eval_mode=False)
            next_state, reward, done, _, info = env.step(action)
            next_state = next_state.astype(np.float32)

            raw_reward = float(reward)
            reward_norm.update(raw_reward)
            train_reward = (
                reward_norm.normalize(raw_reward)
                if cfg.normalize_reward
                else raw_reward
            )

            agent.remember(
                Transition(
                    state=state,
                    action=action,
                    reward=train_reward,
                    next_state=next_state,
                    done=bool(done),
                )
            )

            if len(agent.buffer) >= cfg.learning_starts:
                loss_value = agent.optimize()
                if loss_value > 0.0:
                    losses.append(loss_value)

            episode_reward += raw_reward
            state = next_state

            if cfg.render and (episode % cfg.render_every == 0):
                env.set_msgs(
                    [
                        "ADV DQN TRAIN",
                        f"Episode: {episode}",
                        f"Step: {step}",
                        f"Reward: {episode_reward:.1f}",
                        f"Epsilon: {agent.epsilon:.3f}",
                        f"Check: {info.get('check', 0)}",
                        f"Dist: {info.get('dist', 0):.1f}",
                        f"Speed: {info.get('speed', 0):.1f}",
                    ]
                )
                env.render()

            if done:
                break

        reward_history.append(episode_reward)
        agent.decay_epsilon()

        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(cfg.model_dir, "dqn_adv_best.pt"))

        if episode % cfg.save_every == 0:
            agent.save(os.path.join(cfg.model_dir, f"dqn_adv_ep_{episode}.pt"))

        recent = reward_history[-20:]
        avg20 = float(np.mean(recent)) if recent else episode_reward
        avg_loss = float(np.mean(losses)) if losses else 0.0

        print(
            f"Episode {episode:5d} | reward={episode_reward:10.2f} | avg20={avg20:10.2f} | "
            f"eps={agent.epsilon:.3f} | loss={avg_loss:.4f} | replay={len(agent.buffer)}"
        )

    final_path = os.path.join(cfg.model_dir, "dqn_adv_final.pt")
    agent.save(final_path)
    np.save(
        os.path.join(cfg.model_dir, "reward_history_adv.npy"),
        np.array(reward_history, dtype=np.float32),
    )
    print(f"Training complete. Saved model to {final_path}")


def evaluate(cfg: argparse.Namespace) -> None:
    env = gym.make(cfg.env_id).unwrapped
    env.set_view(cfg.render)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = AdvancedDQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=cfg.learning_rate,
        gamma=cfg.gamma,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay=1,
        replay_capacity=cfg.replay_capacity,
        batch_size=cfg.batch_size,
        n_step=cfg.n_step,
        per_alpha=cfg.per_alpha,
        per_beta_start=cfg.per_beta_start,
        per_beta_frames=cfg.per_beta_frames,
        tau=cfg.soft_tau,
        grad_clip=cfg.grad_clip,
        seed=cfg.seed,
    )

    if not cfg.model_path or not os.path.exists(cfg.model_path):
        raise FileNotFoundError(
            "Please provide an existing --model-path for evaluation."
        )

    agent.load(cfg.model_path)

    rewards: List[float] = []

    for episode in range(1, cfg.eval_episodes + 1):
        state, _ = env.reset()
        state = state.astype(np.float32)
        episode_reward = 0.0

        for step in range(cfg.max_steps):
            action = agent.select_action(state, eval_mode=True)
            next_state, reward, done, _, info = env.step(action)
            state = next_state.astype(np.float32)
            episode_reward += float(reward)

            if cfg.render:
                env.set_msgs(
                    [
                        "ADV DQN EVAL",
                        f"Episode: {episode}",
                        f"Step: {step}",
                        f"Reward: {episode_reward:.1f}",
                        f"Check: {info.get('check', 0)}",
                        f"Dist: {info.get('dist', 0):.1f}",
                        f"Speed: {info.get('speed', 0):.1f}",
                    ]
                )
                env.render()

            if done:
                break

        rewards.append(episode_reward)
        print(f"Eval episode {episode:3d} | reward={episode_reward:10.2f}")

    print(f"Eval mean reward: {np.mean(rewards):.2f} | std: {np.std(rewards):.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Advanced DQN (DDQN + Dueling + PER + n-step) for Pyrace"
    )
    parser.add_argument("--env-id", type=str, default="Pyrace-v3")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")

    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=2000)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=2200)

    parser.add_argument("--replay-capacity", type=int, default=120000)
    parser.add_argument("--learning-starts", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument("--n-step", type=int, default=3)

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
        "--model-dir", type=str, default="dqn_vanilla/models_DQN_adv_v01"
    )
    parser.add_argument("--model-path", type=str, default="")

    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument("--resume-episode", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.mode == "train":
        train(cli_args)
    else:
        evaluate(cli_args)
