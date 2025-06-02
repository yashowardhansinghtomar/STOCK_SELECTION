# core/rl/gym_env.py
import gym
import numpy as np
import pandas as pd
from gym import spaces
from core.logger.logger import logger
from core.feature_store.feature_store import get_or_compute
from core.time_context.time_context import get_simulation_date
from db.postgres_manager import run_query

class ODINTradingEnv(gym.Env):
    """
    A custom Gym environment for RL agent in O.D.I.N.
    Reads replay events from SQL replay_buffer and yields observations and rewards.
    """

    def __init__(self, limit=5000):
        super().__init__()
        self.cursor = 0
        self.limit = limit
        self.events = self._load_events()

        # Extended for 13 base features + 3 regime vectors
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

    def _load_events(self):
        today = str(get_simulation_date().date())
        query = f"""
            SELECT id, stock, date, reward, done, reason,
                   missed_pnl, holding_cost, slippage_penalty, capital_efficiency
            FROM replay_buffer
            WHERE date <= '{today}'
            ORDER BY id
            LIMIT {self.limit};
        """
        df = run_query(query)
        if df is None or df.empty:
            logger.warning("[ENV] No replay events found.")
            return []
        return df.to_dict(orient="records")

    def _parse_event(self, row):
        try:
            symbol = row["stock"]
            date = row["date"]

            # Reward attribution
            base_reward = float(row.get("reward", 0.0))
            missed_pnl = float(row.get("missed_pnl", 0.0))
            holding_cost = float(row.get("holding_cost", 0.0))
            slippage_penalty = float(row.get("slippage_penalty", 0.0))
            capital_efficiency = float(row.get("capital_efficiency", 0.0))
            reward = base_reward + capital_efficiency - holding_cost - slippage_penalty

            # Load features including regime
            df = get_or_compute(symbol, interval="15minute", date=date)
            if df is None or df.empty:
                obs = np.zeros(self.observation_space.shape)
            else:
                base = df.drop(columns=["stock", "date", "interval", "regime_tag"], errors="ignore").values[0]

                regime = df["regime_tag"].values[0] if "regime_tag" in df.columns else "unknown"
                regime_vector = {
                    "trending": [1, 0, 0],
                    "volatile": [0, 1, 0],
                    "sideways": [0, 0, 1]
                }.get(regime, [0, 0, 0])

                obs = np.concatenate([base, regime_vector])

            info = {
                "symbol": symbol,
                "raw_event": row,
                "reward_components": {
                    "base": base_reward,
                    "missed_pnl": missed_pnl,
                    "holding_cost": holding_cost,
                    "slippage_penalty": slippage_penalty,
                    "capital_efficiency": capital_efficiency
                },
                "regime_tag": regime
            }
            return obs, reward, row.get("done", False), info

        except Exception as e:
            logger.warning(f"[ENV] Failed to parse SQL replay row: {e}")
            return np.zeros(self.observation_space.shape), 0.0, True, {}

    def reset(self, *, seed=None, options=None):
        self.cursor = 0
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def step(self, action):
        if self.cursor >= len(self.events):
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, True, False, {}

        event = self.events[self.cursor]
        self.cursor += 1
        obs, reward, done, info = self._parse_event(event)
        return obs.astype(np.float32), float(reward), done, False, info
