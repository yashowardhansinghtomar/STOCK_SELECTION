# core/rl/sql_env.py

import gym
import numpy as np
import pandas as pd
from gym import spaces
from core.logger.logger import logger
from core.feature_store.feature_store import get_or_compute
from core.time_context.time_context import get_simulation_date
from db.postgres_manager import run_query

class ODINSQLTradingEnv(gym.Env):
    """
    Fallback Gym environment for PPO trainer that pulls replay events from SQL
    instead of Redis (used when Redis is not running).
    """

    def __init__(self):
        super().__init__()
        self.events = self._load_events()
        self.index = 0

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

    def _load_events(self):
        sim_date = get_simulation_date().date()
        query = f"""
            SELECT id, stock, date, reward, reason, done,
                   missed_pnl, holding_cost, slippage_penalty, capital_efficiency
            FROM replay_buffer
            WHERE date <= '{sim_date}'
            ORDER BY id
            LIMIT 1000
        """
        df = run_query(query)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.to_dict(orient="records")
        else:
            logger.warning("[SQL ENV] No replay events available in SQL.")
            return []

    def _parse_event(self, event):
        try:
            symbol = event["stock"]
            date = event["date"]

            # Full reward components
            base_reward = float(event.get("reward", 0.0))
            missed_pnl = float(event.get("missed_pnl", 0.0))
            holding_cost = float(event.get("holding_cost", 0.0))
            slippage_penalty = float(event.get("slippage_penalty", 0.0))
            capital_efficiency = float(event.get("capital_efficiency", 0.0))

            total_reward = base_reward + capital_efficiency - holding_cost - slippage_penalty
            done = bool(event.get("done", True))

            df = get_or_compute(symbol, interval="15minute", date=date)
            if df is None or df.empty:
                obs = np.zeros(self.observation_space.shape)
            else:
                obs = df.drop(columns=["stock", "date", "interval"], errors="ignore").values[0]

            info = {
                "symbol": symbol,
                "raw_event": event,
                "reward_breakdown": {
                    "base": base_reward,
                    "missed_pnl": missed_pnl,
                    "holding_cost": holding_cost,
                    "slippage_penalty": slippage_penalty,
                    "capital_efficiency": capital_efficiency
                },
                "reason": event.get("reason", "unknown")
            }

            return obs, total_reward, done, info

        except Exception as e:
            logger.warning(f"[SQL ENV] Failed to parse event: {e}")
            return np.zeros(self.observation_space.shape), 0.0, True, {}

    def reset(self, *, seed=None, options=None):
        self.index = 0
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def step(self, action):
        if self.index >= len(self.events):
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, True, False, {}

        event = self.events[self.index]
        self.index += 1
        obs, reward, done, info = self._parse_event(event)
        return obs.astype(np.float32), float(reward), done, False, info

