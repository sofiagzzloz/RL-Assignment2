import argparse
import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Tuple
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


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones


class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        input_size = state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, action_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: int,
        buffer_size: int,
        batch_size: int,
        use_target_network: bool = False,
        target_update_every: int = 200,
        seed: int = 42,
    ) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = max(1, epsilon_decay)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online_net = QNetwork(state_size, action_size, hidden_sizes=[128, 128]).to(self.device)
        self.target_net = QNetwork(state_size, action_size, hidden_sizes=[128, 128]).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.buffer = ReplayBuffer(buffer_size)

        self.use_target_network = use_target_network
        self.target_update_every = target_update_every
        self.training_steps = 0

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        if (not eval_mode) and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def decay_epsilon(self) -> None:
        decay_amount = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay
        self.epsilon = max(self.epsilon_end, self.epsilon - decay_amount)

    def optimize(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.online_net(states_tensor).gather(1, actions_tensor)

        with torch.no_grad():
            if self.use_target_network:
                next_q_values = self.target_net(next_states_tensor).max(dim=1, keepdim=True)[0]
            else:
                next_q_values = self.online_net(next_states_tensor).max(dim=1, keepdim=True)[0]
            targets = rewards_tensor + self.gamma * (1.0 - dones_tensor) * next_q_values

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.training_steps += 1
        if self.use_target_network and self.training_steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    def save(self, model_path: str) -> None:
        payload = {
            "model_state_dict": self.online_net.state_dict(),
            "epsilon": self.epsilon,
        }
        torch.save(payload, model_path)

    def load(self, model_path: str) -> None:
        payload = torch.load(model_path, map_location=self.device)
        self.online_net.load_state_dict(payload["model_state_dict"])
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.epsilon = float(payload.get("epsilon", self.epsilon_end))


def train(args: argparse.Namespace) -> None:
    env = gym.make("Pyrace-v1").unwrapped
    env.set_view(args.render)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay_episodes,
        buffer_size=args.replay_capacity,
        batch_size=args.batch_size,
        use_target_network=args.use_target_network,
        target_update_every=args.target_update_every,
        seed=args.seed,
    )

    os.makedirs(args.model_dir, exist_ok=True)

    best_reward = -float("inf")
    reward_history: List[float] = []

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset()
        state = state.astype(np.float32)
        episode_reward = 0.0
        losses: List[float] = []

        for step in range(args.max_steps):
            action = agent.select_action(state, eval_mode=False)
            next_state, reward, done, _, info = env.step(action)
            next_state = next_state.astype(np.float32)

            transition = Transition(state=state, action=action, reward=float(reward), next_state=next_state, done=bool(done))
            agent.buffer.add(transition)

            if len(agent.buffer) >= args.learning_starts:
                loss_value = agent.optimize()
                if loss_value > 0.0:
                    losses.append(loss_value)

            episode_reward += float(reward)
            state = next_state

            if args.render and (episode % args.render_every == 0):
                env.set_msgs(
                    [
                        "DQN TRAIN",
                        f"Episode: {episode}",
                        f"Step: {step}",
                        f"Reward: {episode_reward:.1f}",
                        f"Epsilon: {agent.epsilon:.3f}",
                        f"Check: {info.get('check', 0)}",
                        f"Dist: {info.get('dist', 0):.1f}",
                    ]
                )
                env.render()

            if done:
                break

        reward_history.append(episode_reward)
        agent.decay_epsilon()

        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(args.model_dir, "dqn_best.pt"))

        if episode % args.save_every == 0:
            agent.save(os.path.join(args.model_dir, f"dqn_ep_{episode}.pt"))

        recent_rewards = reward_history[-20:]
        avg_recent = float(np.mean(recent_rewards)) if recent_rewards else episode_reward
        avg_loss = float(np.mean(losses)) if losses else 0.0
        print(
            f"Episode {episode:4d} | reward={episode_reward:9.2f} | avg20={avg_recent:9.2f} | "
            f"eps={agent.epsilon:.3f} | loss={avg_loss:.4f} | buffer={len(agent.buffer)}"
        )

    final_model_path = os.path.join(args.model_dir, "dqn_final.pt")
    agent.save(final_model_path)
    np.save(os.path.join(args.model_dir, "reward_history.npy"), np.array(reward_history, dtype=np.float32))
    print(f"Training complete. Saved model to {final_model_path}")


def evaluate(args: argparse.Namespace) -> None:
    env = gym.make("Pyrace-v1").unwrapped
    env.set_view(args.render)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay=1,
        buffer_size=args.replay_capacity,
        batch_size=args.batch_size,
        use_target_network=args.use_target_network,
        target_update_every=args.target_update_every,
        seed=args.seed,
    )

    if not args.model_path or not os.path.exists(args.model_path):
        raise FileNotFoundError("Please provide an existing --model-path for evaluation.")

    agent.load(args.model_path)

    for episode in range(1, args.eval_episodes + 1):
        state, _ = env.reset()
        state = state.astype(np.float32)
        episode_reward = 0.0

        for step in range(args.max_steps):
            action = agent.select_action(state, eval_mode=True)
            next_state, reward, done, _, info = env.step(action)
            state = next_state.astype(np.float32)
            episode_reward += float(reward)

            if args.render:
                env.set_msgs(
                    [
                        "DQN EVAL",
                        f"Episode: {episode}",
                        f"Step: {step}",
                        f"Reward: {episode_reward:.1f}",
                        f"Check: {info.get('check', 0)}",
                        f"Dist: {info.get('dist', 0):.1f}",
                    ]
                )
                env.render()

            if done:
                break

        print(f"Eval episode {episode:3d} | reward={episode_reward:9.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vanilla DQN for Pyrace-v1")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")

    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=2000)

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=1500)

    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--use-target-network", action="store_true")
    parser.add_argument("--target-update-every", type=int, default=200)

    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-every", type=int, default=50)

    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--model-dir", type=str, default="dqn_vanilla/models_DQN_v01")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        evaluate(args)
