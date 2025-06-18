# agents/strategy/rl_strategy_agent.py

from core.predict.rl_predictor import load_policy, load_rl_frame
from core.time_context.time_context import get_simulation_date
from rl.envs.trading_env import TradingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import numpy as np
from core.logger.logger import logger


class RLStrategyAgent:
    def __init__(self, model_name="ppo_intraday", intervals=None, today=None):
        self.model_name = model_name
        # allow overriding the date for live vs. backtest
        sim_date = today or get_simulation_date()
        self.today = pd.to_datetime(sim_date).date()
        self.intervals = intervals or ["day", "60minute", "15minute"]

    def _evaluate_reward(self, df: pd.DataFrame, model_name: str) -> float:
        try:
            env = TradingEnv(df)
            obs = env.reset()[0]
            total_reward = 0

            policy = load_policy(model_name)
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

            model_name = f"ppo_{stock}_{interval}_latest"
            reward = self._evaluate_reward(df, model_name=model_name)
            if reward <= 0:
                continue

            env = TradingEnv(df)
            obs = env.reset()[0]

            policy = load_policy(model_name)
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

    def generate_trades(self, stocks: list[str], today=None) -> list[dict]:

        """
        Wrapper to produce a list of RL signals for bootstrap_trader.
        Calls evaluate() for each stock and returns only the non-None signals.
        """
        # if someone passed today in, override our self.today
        if today is not None:
            self.today = pd.to_datetime(today).date()
        signals = []
        for s in stocks:
            sig = self.evaluate(s)
            if sig is not None:
                signals.append(sig)
        return signals