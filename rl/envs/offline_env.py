import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import pandas as pd
import hashlib
import logging

logger = logging.getLogger("offline_env")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

class OfflineEnv(Env):
    """
    An offline environment that replays stored transitions from a replay buffer.
    Each row contains: state, action, reward, next_state, and done.
    Adds inferred next_state, done flag, episode_id, step_count, and metadata.
    Applies reward shaping based on metadata and logs shaped vs raw rewards.
    """

    def __init__(self, episodes_df):
        super().__init__()
        self.episodes = self._prepare_episodes(episodes_df.reset_index(drop=True))
        self.idx = 0

        # Set observation and action space based on first row
        sample_state = np.array(self.episodes.iloc[0]["state"])
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=sample_state.shape, dtype=np.float32)
        self.action_space = Discrete(3)

    def _prepare_episodes(self, df):
        df = df.copy()

        # Add unique episode_id per (stock, interval, date)
        def generate_episode_id(row):
            key = f"{row['stock']}_{row['interval']}_{row['date']}"
            return hashlib.md5(key.encode()).hexdigest()

        df["episode_id"] = df.apply(generate_episode_id, axis=1)

        # Ensure 'done' flag is set for last in each group
        if "done" not in df.columns:
            df["rank"] = df.groupby("episode_id").cumcount()
            last_rank = df.groupby("episode_id")["rank"].transform("max")
            df["done"] = df["rank"] == last_rank
            df.drop(columns=["rank"], inplace=True)

        # Add step_count within each episode
        df["step_count"] = df.groupby("episode_id").cumcount()

        # Rebuild state and next_state from features
        df["state"] = df["features"].apply(lambda f: np.array(list(f.values())))
        next_states = df["state"].shift(-1)

        # Replace with same state if at end of group
        is_terminal = df["done"]
        df["next_state"] = np.where(is_terminal, df["state"], next_states)

        return df

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        state = np.array(self.episodes.iloc[self.idx]["state"])
        return state, {}

    def _calculate_shaped_reward(self, row):
        reward = row["reward"]
        confidence = row.get("confidence", 0.5)
        step_count = row.get("step_count", 0)

        # Penalize failed high-confidence trades
        if reward < 0 and confidence > 0.7:
            reward *= 1.5

        # Reward boost for early profitable exits
        if reward > 0 and step_count <= 3:
            reward += 0.01

        return reward

    def step(self, action):
        row = self.episodes.iloc[self.idx]
        next_state = np.array(row["next_state"])
        done = bool(row.get("done", self.idx >= len(self.episodes) - 1))

        raw_reward = row["reward"]
        shaped_reward = self._calculate_shaped_reward(row)

        # Log raw vs shaped reward
        logger.info(f"Episode {row.get('episode_id')} Step {row.get('step_count')}: Raw={raw_reward:.4f}, Shaped={shaped_reward:.4f}")

        self.idx += 1
        return next_state, shaped_reward, done, False, {
            "done": done,
            "raw_reward": raw_reward,
            "shaped_reward": shaped_reward,
            "episode_id": row.get("episode_id"),
            "step_count": row.get("step_count", self.idx),
            "entry_price": row.get("entry_price"),
            "strategy_config": row.get("strategy_config"),
            "confidence": row.get("confidence"),
            "interval": row.get("interval"),
            "stock": row.get("stock")
        }
