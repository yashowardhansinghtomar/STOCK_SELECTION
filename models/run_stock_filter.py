# models/run_stock_filter.py

import os
import pandas as pd
import lightgbm as lgb
import joblib
from datetime import datetime
from core.data_provider.data_provider import load_data, save_data
from db.postgres_manager import run_query
from core.logger.logger import logger
from core.config.config import settings

FEATURE_TABLE = "stock_features_day"
PREDICTION_TABLE = "filter_model_predictions"
MODEL_PATH = os.path.join("models", "filter_model.lgb")

FEATURE_COLS = [
    "sma_short", "sma_long", "rsi_thresh", "macd", "vwap", "atr_14",
    "bb_width", "macd_histogram", "price_compression", "volatility_10",
    "volume_spike", "vwap_dev", "stock_encoded"
]

def run_stock_filter(as_of: datetime = None):
    as_of = as_of or datetime.today()
    date_str = as_of.strftime("%Y-%m-%d")

    try:
        features = load_data(FEATURE_TABLE)
        features = features[features["date"] == date_str].dropna(subset=FEATURE_COLS)
        if features.empty:
            raise ValueError(f"No valid features found for {date_str}")
    except Exception as e:
        logger.warning(f"Feature load failed: {e}")
        return settings.fallback_stocks or []

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        logger.error(f"⚠️ Model load failed: {e} — falling back to fallback stocks.")
        logger.warning(f"Using fallback stock list: {settings.fallback_stocks}")

        return settings.fallback_stocks or []

    X = features[FEATURE_COLS]
    probs = model.predict_proba(X)[:, 1]

    results = features[["stock", "date"]].copy()
    results["score"] = probs
    results["rank"] = results["score"].rank(pct=True)
    results["confidence"] = results["score"]
    results["decision"] = (results["score"] > 0.5).map({True: "buy", False: "reject"})

    try:
        save_data(results, PREDICTION_TABLE, if_exists="replace")
        logger.success(f"✅ Stock filter run for {date_str}. Predictions saved.")
    except Exception as e:
        logger.warning(f"❌ Could not save filter predictions: {e}")

    selected = results[results["decision"] == "buy"]["stock"].tolist()
    if not selected:
        logger.warning("⚠️ No stocks selected by filter model — falling back.")
        return settings.fallback_stocks or []

    return selected

if __name__ == "__main__":
    print(run_stock_filter())
