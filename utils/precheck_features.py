# utils/precheck_features.py

import pandas as pd
from core.feature_generator import generate_features
from core.model_io import load_model
from config.paths import PATHS
from core.logger import logger
from datetime import datetime
import json
import os
from core.data_provider import load_data

PRECHECK_CACHE = "cache/precheck_valid_stocks.json"

def get_model_features():
    try:
        _, features = load_model("filter_model")
        return features
    except Exception as e:
        logger.warning(f"⚠️ Failed to load filter model: {e}")
        return []

from core.data_provider import load_data

def is_feature_usable(stock: str, required_features: list) -> bool:
    try:
        df = load_data(stock)
        if df is None or df.empty:
            logger.warning(f"⚠️ {stock}: No data loaded.")
            return False

        df.columns = [col.lower() for col in df.columns]

        if "close" not in df.columns or "volume" not in df.columns:
            logger.warning(f"⚠️ {stock}: Missing essential columns.")
            return False

        # ✅ Must have at least 100 non-null close prices
        if df["close"].dropna().shape[0] < 100:
            logger.warning(f"⚠️ {stock}: Not enough close history (<100 days).")
            return False

        # ✅ Must have at least 100 nonzero volume days
        if (df["volume"] <= 0).sum() > 10:  # allow minor zero days, not many
            logger.warning(f"⚠️ {stock}: Too many days with zero volume.")
            return False

        # ✅ Recent trading check (last 30 days)
        if df["close"].tail(30).isna().any():
            logger.warning(f"⚠️ {stock}: Missing recent close prices.")
            return False
        if (df["volume"].tail(30) <= 0).any():
            logger.warning(f"⚠️ {stock}: Missing recent trading volume.")
            return False

        # ✅ Optional: Check indicators if you already have SMA/RSI
        if "sma20" in df.columns and df["sma20"].tail(30).isna().any():
            logger.warning(f"⚠️ {stock}: Recent SMA missing.")
            return False
        if "rsi" in df.columns and df["rsi"].tail(30).isna().any():
            logger.warning(f"⚠️ {stock}: Recent RSI missing.")
            return False

        return True

    except Exception as e:
        logger.error(f"❌ Precheck error for {stock}: {e}")
        return False



def prefilter_valid_stocks(stocks: list[str]) -> list[str]:
    valid = []
    failed_today = {}

    for stock in stocks:
        try:
            df = load_data(stock)
            if df is None or df.empty:
                failed_today[stock] = "Missing price data"
                continue

            df.columns = [col.lower() for col in df.columns]

            if "close" not in df.columns:
                failed_today[stock] = "Missing close column"
                continue

            if len(df) < 50:
                failed_today[stock] = f"Insufficient history ({len(df)} days)"
                continue

            valid.append(stock)

        except Exception as e:
            failed_today[stock] = f"Error: {str(e)}"

    logger.info(f"✅ Precheck complete: {len(valid)} valid, {len(failed_today)} failed.")

    # Save valid list for reuse
    with open(PRECHECK_CACHE, "w") as f:
        json.dump(valid, f)

    return valid