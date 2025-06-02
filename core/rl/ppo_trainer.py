# core/rl/ppo_trainer.py

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from collections import deque
from core.rl.gym_env import ODINTradingEnv
from core.logger.logger import logger
import os
from utils.progress_logger import log_model_progress


class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid()  # [position_size, exit_days_scaled]
        )
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        latent = self.shared(x)
        return self.actor(latent), self.critic(latent)


class PPOTrainer:
    def __init__(self, env, lr=3e-4, gamma=0.99):
        self.env = env
        self.gamma = gamma
        self.model = ActorCritic(input_dim=env.observation_space.shape[0])
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.buffer = deque(maxlen=5000)

    def collect_rollout(self, steps=1000):
        obs = self.env.reset()
        for _ in range(steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _ = self.model(obs_tensor)
            action = action.squeeze().numpy()

            next_obs, reward, done, info = self.env.step(action)

            # Optional debug log for reward breakdown
            reason = info.get("reason", "N/A")
            breakdown = info.get("reward_breakdown", {})
            logger.debug(
                f"[RL] Step: {reason} | Reward: {reward:.2f} | "
                f"Profit={breakdown.get('profit', 0):.2f}, "
                f"Missed={breakdown.get('missed_profit', 0):.2f}, "
                f"Penalty={breakdown.get('hold_duration_penalty', 0):.2f}"
            )

            self.buffer.append((obs, action, reward, next_obs, done))
            obs = next_obs
            if done:
                obs = self.env.reset()

    def train_step(self, batch_size=64):
        if len(self.buffer) < batch_size:
            logger.warning("Not enough samples to train.")
            return

        batch = [self.buffer[i] for i in np.random.choice(len(self.buffer), batch_size, replace=False)]
        obs, actions, rewards, next_obs, dones = zip(*batch)
        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        pred_actions, values = self.model(obs)
        _, next_values = self.model(next_obs)
        target_values = rewards + self.gamma * next_values.squeeze() * (1 - dones)
        advantages = target_values.detach() - values.squeeze()

        actor_loss = ((pred_actions - actions) ** 2).mean()
        critic_loss = (advantages ** 2).mean()

        loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        logger.info(f"\U0001F3AF PPO step complete. Loss: {loss.item():.4f}")
        log_model_progress("PPO", loss.item(), buffer_size=len(self.buffer))


    def save_model(self, path="checkpoints/ppo.pt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.success(f"\u2705 Model saved to {path}")

    def load_model(self, path="checkpoints/ppo.pt"):
        self.model.load_state_dict(torch.load(path))
        logger.info(f"🔄 Loaded model from {path}")
