import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TradingEnv(gym.Env):
    """
    A simple long-only trading environment for PPO.
    Supports different frequencies: e.g., 'day', '15m'
    Actions: 0 = hold, 1 = buy, 2 = sell
    """

    def __init__(self, df, window=30, fee_pct=0.001, cash=1.0, freq: str = "day"):
        super().__init__()
        self.df = df.astype(np.float32)
        self.window = window
        self.fee = fee_pct
        self.init_cash = cash
        self.freq = freq  # 🆕 frequency tag for logging/inference

        self.start_idx = window
        self.end_idx = len(df) - 1

        self.action_space = spaces.Discrete(3)

        obs_size = window * df.shape[1] + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.reset()

    def _get_obs(self):
        window_data = self.df.iloc[self.idx - self.window:self.idx].values.flatten()
        obs = np.append(window_data, [self.cash, self.position]).astype(np.float32)

        if not np.all(np.isfinite(obs)):
            raise ValueError(f"🚨 NaN or inf in observation at idx={self.idx} → {obs}")

        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = self.start_idx
        self.cash = self.init_cash
        self.position = 0
        self.cost_basis = 0

        obs = self._get_obs()
        assert np.all(np.isfinite(obs)), f"🧨 Bad reset obs: {obs}"
        return obs, {}

    def step(self, action):
        price = self.df.iloc[self.idx]["close"]
        reward = 0

        if action == 1 and self.position == 0:
            self.position = 1
            self.cost_basis = price
            self.cash -= price * self.fee

        elif action == 2 and self.position == 1:
            pnl = (price - self.cost_basis) / self.cost_basis
            reward = pnl - self.fee
            self.cash *= (1 + reward)
            self.position = 0

        self.idx += 1
        done = self.idx >= self.end_idx
        truncated = False

        if done and self.position == 1:
            pnl = (price - self.cost_basis) / self.cost_basis
            reward += pnl - self.fee
            self.cash *= (1 + pnl - self.fee)
            self.position = 0

        obs = self._get_obs()
        return obs, reward, done, truncated, {}
