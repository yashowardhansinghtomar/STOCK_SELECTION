import os
import argparse
from datetime import datetime
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl.envs.trading_env import TradingEnv
from core.predict.rl_predictor import load_rl_frame
from core.model_io import save_model
from core.logger.logger import logger
from core.config.config import settings

def get_symbols(default=True):
    return ["RELIANCE", "TCS", "INFY"] if default else settings.stock_whitelist or []

def make_env(symbol, freq: str):
    def _env():
        df = load_rl_frame(symbol, days=1500, interval=freq)
        if df is None or df.empty or len(df) < 60:
            raise ValueError(f"❌ Not enough data to train RL for {symbol}")
        return TradingEnv(df, freq=freq)
    return _env

def main(args):
    logger.start("🏋️ PPO training for RL agent...")

    symbols = args.symbols or get_symbols()
    freq = args.freq.lower()

    valid_envs, skipped = [], []
    for sym in symbols:
        try:
            valid_envs.append(make_env(sym, freq))
        except Exception as e:
            logger.warnings(f"⚠️ Skipping {sym}: {e}")
            skipped.append(sym)

    if not valid_envs:
        logger.error("❌ No valid symbols to train on. Aborting.")
        return

    vec_env = DummyVecEnv(valid_envs)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=8192,
        ent_coef=0.01,
        tensorboard_log="logs/rl_ppo/"
    )

    logger.info(f"⏱️ Training PPO for {args.steps:,} timesteps across {len(valid_envs)} stocks...")
    model.learn(total_timesteps=args.steps)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.name or f"{settings.model_names['ppo']}_{freq}_{now}"

    save_model(model_name, model, meta={
        "algo": "PPO",
        "steps": args.steps,
        "trained_at": str(pd.Timestamp.now()),
        "symbols_trained": symbols,
        "symbols_skipped": skipped,
        "interval": freq,
        "reward_mode": "raw",
        "model_params": {
            "n_steps": 2048,
            "batch_size": 8192,
            "learning_rate": 3e-4,
            "ent_coef": 0.01
        }
    })

    logger.success(f"✅ PPO model saved as {model_name}")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs/rl_ppo", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2_000_000, help="Total PPO timesteps")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to train on")
    parser.add_argument("--name", type=str, default=None, help="Model save name")
    parser.add_argument("--freq", type=str, default="day", help="Data frequency: 'day', '15m', etc.")
    args = parser.parse_args()

    main(args)
