# models/run_stock_filter.py

import os
import pandas as pd
import lightgbm as lgb
import joblib
from datetime import datetime, timedelta
from core.data_provider.data_provider import load_data, save_data
from db.postgres_manager import run_query
from core.logger.logger import logger
from core.config.config import settings

FEATURE_TABLE = "stock_features_day"
PREDICTION_TABLE = settings.tables.predictions["filter"]
RECS_TABLE = settings.tables.recommendations
MODEL_PATH = os.path.join("models", "filter_model.lgb")

FEATURE_COLS = [
    "sma_short", "sma_long", "rsi_thresh", "macd", "vwap", "atr_14",
    "bb_width", "macd_histogram", "price_compression", "volatility_10",
    "volume_spike", "vwap_dev", "stock_encoded"
]

def run_stock_filter(as_of: datetime = None, lookback_only: bool = False):
    as_of = as_of or datetime.today()
    date_str = as_of.strftime("%Y-%m-%d")
    logger.info(f"🧪 Running filter model for {date_str} | lookback_only={lookback_only}")

    try:
        features = load_data(FEATURE_TABLE)
        features["date"] = pd.to_datetime(features["date"])

        # Lookback-only: use only past dates, not today
        if lookback_only:
            features = features[features["date"] < as_of]
        else:
            features = features[features["date"] == as_of]

        features = features.dropna(subset=FEATURE_COLS)
        logger.info(f"📊 Loaded {len(features)} rows of features for filter input")
        if features.empty:
            raise ValueError(f"No valid features found for filtering.")
    except Exception as e:
        logger.warning(f"⚠️ Feature load failed: {e}")
        logger.warning(f"🔙 Using fallback stocks: {settings.fallback_stocks}")
        return settings.fallback_stocks or []

    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"📦 Loaded filter model from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"❌ Model load failed: {e}")
        logger.warning(f"🔙 Using fallback stocks: {settings.fallback_stocks}")
        return settings.fallback_stocks or []

    # Predict scores
    X = features[FEATURE_COLS]
    probs = model.predict_proba(X)[:, 1]

    results = features[["stock", "date"]].copy()
    results["score"] = probs
    results["rank"] = results["score"].rank(pct=True)
    results["confidence"] = results["score"]
    results["decision"] = (results["score"] > 0.5).map({True: "buy", False: "reject"})

    try:
        save_data(results, PREDICTION_TABLE, if_exists="replace")
        logger.success(f"✅ Stock filter predictions saved for {date_str} ({len(results)} total)")
    except Exception as e:
        logger.warning(f"❌ Could not save filter predictions: {e}")

    selected = results[results["decision"] == "buy"]
    if selected.empty:
        logger.warning("⚠️ No stocks selected by filter model — using fallback.")
        logger.info(f"🔙 Fallback stocks: {settings.fallback_stocks}")

        # 🔎 DEBUG: Show top-scoring stocks even if none selected
        top_debug = results.sort_values("score", ascending=False).head(10)
        logger.info("📉 Top 10 stocks by score:")
        logger.info(top_debug[["stock", "score", "rank"]].to_string(index=False))

        return settings.fallback_stocks or []

    logger.info(f"✅ Selected {len(selected)} stocks | Top: {selected[:5]} ...")


    # 🔍 Optional: Log stats
    above_60 = (results["score"] > 0.6).sum()
    above_50 = (results["score"] > 0.5).sum()
    logger.info(f"📈 {above_60} stocks have score > 0.6")
    logger.info(f"📈 {above_50} stocks have score > 0.5")

    return selected["stock"].tolist()


if __name__ == "__main__":
    print(run_stock_filter())
