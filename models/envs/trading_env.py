import gym
import numpy as np
import pandas as pd
from gym import spaces
from core.data_provider.data_provider import fetch_stock_data
from core.config.config import settings

class TradingEnv(gym.Env):
    def __init__(self, stock: str, max_steps=200):
        super(TradingEnv, self).__init__()
        self.stock = stock
        self.df = fetch_stock_data(stock, days=settings.exit_lookback_days)
        self.df = self.df.reset_index(drop=True)
        self.max_steps = min(max_steps, len(self.df) - 1)
        self.current_step = 0

        # Action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)

        # Observation space: OHLCV + price indicators (can be extended)
        self.features = ["open", "high", "low", "close", "volume"]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.features),), dtype=np.float32
        )

        self.position = None   # None, 'long'
        self.entry_price = None
        self.profit = 0.0

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        return row[self.features].astype(np.float32).values

    def reset(self):
        self.current_step = 0
        self.position = None
        self.entry_price = None
        self.profit = 0.0
        return self._get_obs()

    def step(self, action):
        done = False
        reward = 0.0

        current_price = self.df.iloc[self.current_step]["close"]

        if action == 1 and self.position is None:  # buy
            self.position = 'long'
            self.entry_price = current_price
        elif action == 2 and self.position == 'long':  # sell
            pnl = current_price - self.entry_price
            reward = pnl
            self.profit += pnl
            self.position = None
            self.entry_price = None

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
            # force close position at end
            if self.position == 'long':
                reward += self.df.iloc[self.current_step - 1]["close"] - self.entry_price
                self.position = None

        obs = self._get_obs() if not done else np.zeros_like(self._get_obs())
        return obs, reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Position: {self.position}, Profit: {self.profit:.2f}")
