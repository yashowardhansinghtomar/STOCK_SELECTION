import numpy as np
import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl.envs.trading_env import TradingEnv
from db.replay_buffer_sql import load_replay_episodes
from core.model_io import save_model, load_model, load_latest_model
from core.logger.logger import logger
from core.config.config import settings

class OfflineEnv:
    def __init__(self, episodes):
        self.episodes = episodes.reset_index(drop=True)
        self.idx = 0

    def reset(self):
        self.idx = 0
        return np.array(self.episodes.iloc[self.idx]["state"])

    def step(self, action):
        row = self.episodes.iloc[self.idx]
        reward = row["reward"]
        next_state = np.array(row["next_state"])
        done = row.get("done", False)
        self.idx += 1
        if self.idx >= len(self.episodes):
            done = True
        return next_state, reward, done, False, {}

def finetune_rl(model_name=None, stock=None, interval=None, steps=5000):
    logger.start("🔁 Fine-tuning RL model from replay buffer...")

    episodes_df = load_replay_episodes(stock=stock, interval=interval)
    if episodes_df is None or len(episodes_df) < 100:
        logger.error("❌ Not enough episodes to fine-tune RL.")
        return

    episodes_df = episodes_df.dropna(subset=["features", "reward"]).copy()
    episodes_df["state"] = episodes_df["features"].apply(lambda f: np.array(list(f.values())))
    episodes_df["next_state"] = episodes_df["features"].apply(lambda f: np.array(list(f.values())))

    env = DummyVecEnv([lambda: OfflineEnv(episodes_df)])

    # Determine model name to load
    if not model_name:
        if not stock or not interval:
            logger.error("❌ Must provide stock and interval if model_name is not specified.")
            return
        model_name = f"{settings.model_names['ppo']}_{stock}_{interval}_latest"

    try:
        model = load_model(model_name)
        model = model if isinstance(model, PPO) else model["model"]
    except Exception:
        logger.warnings(f"⚠️ Fallback to loading base model {model_name.replace('_latest', '')}")
        model = PPO.load(model_name.replace("_latest", ""))

    model.set_env(env)
    model.learn(total_timesteps=steps)

    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    final_model_name = f"{settings.model_names['ppo']}_{stock}_{interval}_{date_str}"
    save_model(final_model_name, model, meta={
        "trained_at": str(datetime.now()),
        "stock": stock,
        "interval": interval,
        "steps": steps,
        "replay_rows": len(episodes_df),
        "origin_model": model_name
    })

    logger.success(f"✅ Fine-tuned PPO model saved as: {final_model_name}")

if __name__ == "__main__":
    finetune_rl(stock="RELIANCE", interval="15minute")
