# rl_finetune.py
from rl.replay_buffer import ReplayBuffer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl.envs.trading_env import TradingEnv
import numpy as np

class OfflineEnv:
    def __init__(self, episodes):
        self.episodes = episodes
        self.idx = 0

    def reset(self):
        self.idx = 0
        s = np.array(self.episodes.iloc[self.idx]["state"])
        return s

    def step(self, action):
        row = self.episodes.iloc[self.idx]
        reward = row["reward"]
        next_state = np.array(row["next_state"])
        done = row["done"]
        self.idx += 1
        if self.idx >= len(self.episodes):
            done = True
        return next_state, reward, done, False, {}


def finetune_rl(model_path="models/rl_policy.zip", steps=5000):
    buffer = ReplayBuffer()
    if len(buffer) < 100:
        print("❌ Not enough data in buffer to fine-tune RL.")
        return

    episodes = buffer.sample(n=1000)
    for col in ["state", "next_state"]:
        episodes[col] = episodes[col].apply(np.array)

    env = DummyVecEnv([lambda: OfflineEnv(episodes)])

    model = PPO.load(model_path)
    model.set_env(env)
    print("🔁 Fine-tuning PPO...")
    model.learn(total_timesteps=steps)
    model.save(model_path)
    print("✅ PPO fine-tuned and saved.")


if __name__ == "__main__":
    finetune_rl()
