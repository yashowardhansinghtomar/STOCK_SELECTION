# core/rl_predictor.py

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from core.logger import logger
from core.config import settings
from core.time_context import get_simulation_date
from core.model_io import load_model
from core.data_provider import fetch_stock_data, load_data
from rl.envs.trading_env import TradingEnv
from core.feature_enricher_multi import enrich_multi_interval_features
_rl_policies = {}

def load_policy(model_name: str = "rl_policy") -> PPO:
    if model_name not in _rl_policies:
        try:
            model_bin = load_model(model_name)
            _rl_policies[model_name] = PPO.load(model_bin)
            logger.success(f"🤖 Loaded RL policy: {model_name}")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load RL policy '{model_name}': {e}")
    return _rl_policies[model_name]

def load_rl_frame(symbol: str, days: int = 1500, interval: str = "day") -> pd.DataFrame:
    end = get_simulation_date()
    price = fetch_stock_data(symbol, end=end, days=days, interval=interval)
    if price is None or price.empty:
        logger.warning(f"⚠️ No price data for {symbol} @ {interval}")
        return pd.DataFrame()

    
    feats = enrich_multi_interval_features(symbol, end)

    if feats.empty:
        logger.warning(f"⚠️ No feature data for {symbol} @ {interval}")
        return pd.DataFrame()

    df = price.merge(feats, on="date", how="inner").sort_values("date")

    df_date = df[["date"]]
    df_feats = df.select_dtypes(include=[np.number])
    df_feats = df_feats.infer_objects(copy=False).interpolate(limit_direction="both")
    df_feats = df_feats.fillna(0).replace([np.inf, -np.inf], 0)
    df = pd.concat([df_date.reset_index(drop=True), df_feats.reset_index(drop=True)], axis=1)

    essential = ["sma_short", "sma_long", "rsi_thresh", "volume_spike", "vwap_dev"]
    missing = [col for col in essential if col not in df.columns]
    if missing:
        logger.warning(f"⚠️ {symbol} is missing: {missing} @ {interval}")
    available = [col for col in essential if col in df.columns]
    if not available:
        return pd.DataFrame()

    df = df.dropna(subset=available)
    if len(df) < 60:
        logger.warning(f"⚠️ {symbol}: {len(df)} usable rows after filtering @ {interval}")
        return pd.DataFrame()

    return df.set_index("date")

def predict_action(symbol: str, model_name: str = "rl_policy", interval: str = "day") -> str:
    df = load_rl_frame(symbol, days=1500, interval=interval)
    if df.empty or len(df) < 30:
        return "hold"

    env = DummyVecEnv([lambda: TradingEnv(df, freq=interval)])
    obs = env.reset()

    if not np.all(np.isfinite(obs[0])):
        logger.warning(f"🧨 RL obs for {symbol} has NaN or inf: {obs}")
        return "hold"

    policy = load_policy(model_name)
    action, _ = policy.predict(obs, deterministic=True)
    return ["hold", "buy", "sell"][int(action)]
