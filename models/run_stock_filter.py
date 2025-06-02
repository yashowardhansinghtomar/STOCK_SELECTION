# models/run_stock_filter.py

import pandas as pd
import lightgbm as lgb
import joblib
from datetime import datetime
from core.data_provider.data_provider import load_data, save_data
from db.postgres_manager import run_query
from core.logger.logger import logger

FEATURE_TABLE = "stock_features_day"
PREDICTION_TABLE = "filter_model_predictions"
MODEL_PATH = "models/filter_model.lgb"

FEATURE_COLS = [
    "sma_short", "sma_long", "rsi_thresh", "macd", "vwap", "atr_14",
    "bb_width", "macd_histogram", "price_compression", "volatility_10",
    "volume_spike", "vwap_dev", "stock_encoded"
]


def run_stock_filter(as_of: datetime = None):
    as_of = as_of or datetime.today()
    date_str = as_of.strftime("%Y-%m-%d")
    
    features = load_data(FEATURE_TABLE)
    if features.empty:
        logger.warning("No features available for prediction.")
        return

    features = features[features["date"] == date_str].dropna(subset=FEATURE_COLS)
    if features.empty:
        logger.warning(f"No features found for {date_str}.")
        return

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        return

    X = features[FEATURE_COLS]
    probs = model.predict_proba(X)[:, 1]

    results = features[["stock", "date"]].copy()
    results["score"] = probs
    results["rank"] = results["score"].rank(pct=True)
    results["confidence"] = results["score"]
    results["decision"] = (results["score"] > 0.5).map({True: "buy", False: "reject"})

    save_data(results, PREDICTION_TABLE, if_exists="replace")
    logger.success(f"Stock filter run for {date_str}. Predictions saved.")


if __name__ == "__main__":
    run_stock_filter()
