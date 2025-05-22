from core.rl_predictor import load_policy, load_rl_frame
from core.time_context import get_simulation_date
from rl.envs.trading_env import TradingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import numpy as np
from core.logger import logger


class RLStrategyAgent:
    def __init__(self, model_name="rl_policy", intervals=None):
        self.model_name = model_name
        self.today = pd.to_datetime(get_simulation_date()).date()
        self.intervals = intervals or ["day", "60m", "15m"]

    def _evaluate_reward(self, df: pd.DataFrame) -> float:
        try:
            env = TradingEnv(df)
            obs = env.reset()[0]
            total_reward = 0

            policy = load_policy(self.model_name)
            for _ in range(env.start_idx, env.end_idx):
                action, _ = policy.predict(obs, deterministic=True)
                obs, reward, done, _, _ = env.step(action)
                total_reward += reward
                if done:
                    break

            return total_reward
        except Exception as e:
            logger.warning(f"⚠️ Reward simulation failed: {e}")
            return -np.inf

    def evaluate(self, stock: str) -> dict | None:
        best_signal = None

        for interval in self.intervals:
            logger.info(f"🧠 [RL] Evaluating {stock} @ {interval}...")
            df = load_rl_frame(stock, interval=interval)
            if df.empty or len(df) < 30:
                continue

            reward = self._evaluate_reward(df)
            if reward <= 0:
                continue  # Skip weak signals

            env = TradingEnv(df)
            obs = env.reset()[0]

            policy = load_policy(self.model_name)
            action, _ = policy.predict(obs, deterministic=True)
            if action == 0:
                continue

            signal = {
                "stock": stock,
                "date": self.today,
                "strategy": "RL_PPO",
                "trade_triggered": 1,
                "source": "rl_agent",
                "interval": interval,
                "imported_at": pd.Timestamp.now(),
                "confidence": float(reward),
            }
            if action == 2:
                signal["action"] = "sell"

            if best_signal is None or reward > best_signal["confidence"]:
                best_signal = signal

        if best_signal:
            logger.success(f"✅ RL best signal: {best_signal}")
        else:
            logger.info(f"🤷 RL: No valid trade for {stock}")

        return best_signal
