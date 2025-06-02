import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from core.logger.logger import logger
from core.config.config import settings
from core.time_context.time_context import get_simulation_date
from core.model_io import load_model, load_latest_model
from core.data_provider.data_provider import fetch_stock_data, load_data
from rl.envs.trading_env import TradingEnv
from core.feature_engineering.feature_enricher_multi import enrich_multi_interval_features

_rl_policies = {}

def load_policy(model_name: str = "ppo_intraday") -> PPO:
    if model_name.endswith("_latest"):
        return load_latest_model(model_name.replace("_latest", ""))

    if model_name not in _rl_policies:
        try:
            model_obj = load_model(model_name)
            model = model_obj["model"]
            _rl_policies[model_name] = model
            logger.success(f"🤖 Loaded RL policy from DB: {model_name}")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load RL policy '{model_name}': {e}")
    return _rl_policies[model_name]

def load_rl_frame(symbol: str, days: int = 1500, interval: str = "day") -> pd.DataFrame:
    end = get_simulation_date()
    price = fetch_stock_data(symbol, end=end, days=days, interval=interval)
    if price is None or price.empty:
        logger.warnings(f"⚠️ No price data for {symbol} @ {interval}")
        return pd.DataFrame()

    feats = enrich_multi_interval_features(symbol, end, intervals=[interval])
    if feats.empty:
        logger.warnings(f"⚠️ No feature data for {symbol} @ {interval}")
        return pd.DataFrame()

    df = price.merge(feats, on="date", how="inner").sort_values("date")

    df_feats = df.select_dtypes(include=[np.number]).infer_objects(copy=False)
    df_feats = df_feats.interpolate(limit_direction="both").fillna(0).replace([np.inf, -np.inf], 0)
    df = pd.concat([df[["date"]].reset_index(drop=True), df_feats.reset_index(drop=True)], axis=1)

    essential = ["sma_short", "sma_long", "rsi_thresh", "volume_spike", "vwap_dev"]
    missing = [col for col in essential if col not in df.columns]
    if missing:
        logger.warnings(f"⚠️ {symbol} is missing: {missing} @ {interval}")
    available = [col for col in essential if col in df.columns]
    df = df.dropna(subset=available)

    if len(df) < 60:
        logger.warnings(f"⚠️ {symbol}: only {len(df)} usable rows @ {interval}")
        return pd.DataFrame()

    return df.set_index("date")

def predict_action(symbol: str, model_name: str = "rl_policy", interval: str = "day") -> str:
    df = load_rl_frame(symbol, days=1500, interval=interval)
    if df.empty or len(df) < 30:
        logger.warnings(f"📭 Not enough data to predict RL action for {symbol} @ {interval}")
        return "hold"

    env = DummyVecEnv([lambda: TradingEnv(df, freq=interval)])
    obs = env.reset()

    if not np.all(np.isfinite(obs[0])):
        logger.warnings(f"🧨 RL obs for {symbol} has NaN or inf: {obs}")
        return "hold"

    try:
        specific_model = f"{settings.model_names['ppo']}_{symbol}_{interval}_latest"
        policy = load_policy(specific_model)
        logger.info(f"✅ Using specific RL policy: {specific_model}")
    except Exception:
        try:
            policy = load_policy(model_name)
            logger.warnings(f"⚠️ Fallback to default policy: {model_name}")
        except Exception as e:
            logger.error(f"❌ No valid RL policy found for {symbol}: {e}")
            return "hold"

    action, _ = policy.predict(obs, deterministic=True)
    return ["hold", "buy", "sell"][int(action)]

def predict_with_fallback(symbol: str, model_name: str = "ppo_intraday") -> str:
    for interval in ["day", "60minute", "15minute"]:
        df = load_rl_frame(symbol, days=1500, interval=interval)
        if df.empty or len(df) < 30:
            continue

        env = DummyVecEnv([lambda: TradingEnv(df, freq=interval)])
        obs = env.reset()
        if not np.all(np.isfinite(obs[0])):
            continue

        try:
            model_id = f"{settings.model_names['ppo']}_{symbol}_{interval}_latest"
            policy = load_policy(model_id)
            action, _ = policy.predict(obs, deterministic=True)
            logger.info(f"🔁 RL fallback used model: {model_id}")
            return ["hold", "buy", "sell"][int(action)]
        except Exception:
            continue

    logger.warnings(f"🪫 RL fallback failed for {symbol}. Returning 'hold'")
    return "hold"
