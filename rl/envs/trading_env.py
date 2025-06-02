import gymnasium as gym
import numpy as np
from gymnasium import spaces
from datetime import datetime
from core.model_io import save_model  # centralized model saving
from core.model_io import load_model, load_latest_model
from stable_baselines3 import PPO
from core.logger.logger import logger
from stable_baselines3.common.vec_env import DummyVecEnv


class TradingEnv(gym.Env):
    """
    A trading environment supporting long and short positions with reward shaping,
    drawdown penalties, max holding logic, and episodic control.
    """

    def __init__(
        self,
        df,
        window=30,
        fee_pct=0.001,
        cash=1.0,
        freq: str = "day",
        max_steps: int = 50,
        max_holding: int = 10,
        allow_short: bool = True,
        reward_mode: str = "raw",
        penalty_weight: float = 0.1,
    ):
        super().__init__()
        self.df = df.astype(np.float32)
        self.window = window
        self.fee = fee_pct
        self.init_cash = cash
        self.freq = freq
        self.allow_short = allow_short
        self.max_steps = max_steps
        self.max_holding = max_holding
        self.reward_mode = reward_mode
        self.penalty_weight = penalty_weight

        self.start_idx = window
        self.end_idx = len(df) - 1

        self.action_space = spaces.Discrete(3)
        obs_size = window * df.shape[1] + 3  # cash, position, holding_days
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.reset()

    def _get_obs(self):
        window_data = self.df.iloc[self.idx - self.window:self.idx].values.flatten()
        obs = np.append(window_data, [self.cash, self.position, self.holding_days]).astype(np.float32)

        if not np.all(np.isfinite(obs)):
            raise ValueError(f"🧨 NaN or inf in observation at idx={self.idx}: {obs}")

        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = self.start_idx
        self.cash = self.init_cash
        self.position = 0
        self.cost_basis = 0
        self.step_count = 0
        self.holding_days = 0
        self.returns = []
        self.peak_price = 0
        self.max_drawdown = 0
        return self._get_obs(), {}

    def _calculate_reward(self, raw_reward):
        penalty = self.penalty_weight * self.max_drawdown
        net_reward = raw_reward - penalty

        if self.reward_mode == "raw":
            return net_reward
        elif self.reward_mode == "sharpe":
            self.returns.append(net_reward)
            if len(self.returns) > 1:
                mean_r = np.mean(self.returns)
                std_r = np.std(self.returns) + 1e-8
                return mean_r / std_r
            return net_reward

        return net_reward

    def step(self, action):
        price = self.df.iloc[self.idx]["close"]
        reward = 0

        # ========== LONG & SHORT LOGIC ==========
        if action == 1:
            if self.position == 0:
                self.position = 1
                self.cost_basis = price
                self.cash -= price * self.fee
                self.peak_price = price
                self.holding_days = 0
                self.max_drawdown = 0
            elif self.position == -1 and self.allow_short:
                pnl = (self.cost_basis - price) / self.cost_basis
                reward = pnl - self.fee
                self.cash *= (1 + reward)
                self.position = 0
                self.max_drawdown = 0

        elif action == 2:
            if self.position == 0 and self.allow_short:
                self.position = -1
                self.cost_basis = price
                self.cash -= price * self.fee
                self.peak_price = price
                self.holding_days = 0
                self.max_drawdown = 0
            elif self.position == 1:
                pnl = (price - self.cost_basis) / self.cost_basis
                reward = pnl - self.fee
                self.cash *= (1 + reward)
                self.position = 0
                self.max_drawdown = 0

        # ========== DRAWNDOWN TRACKING ==========
        if self.position != 0:
            self.holding_days += 1
            self.peak_price = max(self.peak_price, price) if self.position == 1 else min(self.peak_price, price)
            drawdown = (
                (self.peak_price - price) / self.peak_price if self.position == 1
                else (price - self.peak_price) / self.peak_price
            )
            self.max_drawdown = max(self.max_drawdown, drawdown)

        # ========== TERMINATION CONDITIONS ==========
        self.idx += 1
        self.step_count += 1
        done = self.idx >= self.end_idx or self.step_count >= self.max_steps
        truncated = False

        # Auto-close for max holding days
        if self.holding_days >= self.max_holding and self.position != 0:
            pnl = (
                (price - self.cost_basis) / self.cost_basis if self.position == 1
                else (self.cost_basis - price) / self.cost_basis
            )
            reward += pnl - self.fee
            self.cash *= (1 + pnl - self.fee)
            self.position = 0
            self.max_drawdown = 0
            truncated = True

        if done and self.position != 0:
            pnl = (
                (price - self.cost_basis) / self.cost_basis if self.position == 1
                else (self.cost_basis - price) / self.cost_basis
            )
            reward += pnl - self.fee
            self.cash *= (1 + pnl - self.fee)
            self.position = 0
            self.max_drawdown = 0

        shaped_reward = self._calculate_reward(reward)
        return self._get_obs(), shaped_reward, done, truncated, {"done": done}


def save_rl_model(model, stock: str, interval: str, steps: int, replay_rows: int):
    model_name = f"ppo_{stock}_{interval}"
    meta = {
        "trained_at": str(datetime.now()),
        "stock": stock,
        "interval": interval,
        "steps": steps,
        "replay_rows": replay_rows,
    }
    save_model(model_name, model, meta=meta)
    return model_name


def predict_with_fallback(symbol: str, model_name: str = "ppo_intraday") -> str:
    from core.data_provider.data_provider import fetch_stock_data
    from core.feature_engineering.feature_enricher_multi import enrich_multi_interval_features

    from core.predict.rl_predictor import TradingEnv, load_policy

    import numpy as np
    from stable_baselines3.common.vec_env import DummyVecEnv
    from core.time_context.time_context import get_simulation_date

    for interval in ["day", "60minute", "15minute"]:
        try:
            from core.predict.rl_predictor import load_rl_frame
            df = load_rl_frame(symbol, days=1500, interval=interval)
            if df.empty or len(df) < 30:
                continue

            env = DummyVecEnv([lambda: TradingEnv(df, freq=interval)])
            obs = env.reset()
            if not np.all(np.isfinite(obs[0])):
                continue

            policy = load_policy(f"ppo_{symbol}_{interval}_latest")
            action, _ = policy.predict(obs, deterministic=True)
            return ["hold", "buy", "sell"][int(action)]
        except Exception:
            continue

    return "hold"
