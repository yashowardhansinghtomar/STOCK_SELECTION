import pandas as pd
from datetime import datetime
from core.logger.logger import logger
from core.config.config import settings
from core.skiplist.skiplist import is_in_skiplist
from core.feature_store.feature_store import get_or_compute
from core.feature_engineering.regime_features import compute_regime_features  # NEW

def fetch_features(stock: str, interval: str, refresh_if_missing: bool = True, start: str = None, end: str = None) -> pd.DataFrame:
    """
    Unified interface to fetch features using DuckDB-backed cache.

    Args:
        stock (str): stock symbol
        interval (str): time interval (e.g. 'day', '15minute')
        refresh_if_missing (bool): whether to compute if missing
        start (str or datetime): optional start date
        end (str or datetime): optional end date

    Returns:
        pd.DataFrame with feature rows (can be empty)
    """
    if is_in_skiplist(stock):
        logger.warning(f"⏩ Skipping {stock} — already in skiplist.")
        return pd.DataFrame()

    try:
        date = pd.to_datetime(start).date() if start else pd.to_datetime(datetime.now()).date()
        features = get_or_compute(stock, interval, str(date)) if refresh_if_missing else pd.DataFrame()
        if features.empty:
            return features

        # ✅ Add regime tag
        raw_df = get_or_compute(stock, interval, str(date), window=20)
        regime_df = compute_regime_features(raw_df)
        merged = pd.merge(features, regime_df, on="date", how="left")

        return merged

    except Exception as e:
        logger.warning(f"⚠️ Failed to fetch/compute features for {stock} @ {interval}: {e}")
        return pd.DataFrame()

def fetch_features_with_backfill(stock: str, interval: str, sim_date=None) -> pd.DataFrame:
    """
    Backward-compatible fetch with fallback, used in simulation mode.
    """
    try:
        date = sim_date if sim_date else datetime.today().date()
        return fetch_features(stock, interval, start=str(date))
    except Exception as e:
        logger.warning(f"⚠️ Backfill fallback failed for {stock} @ {interval}: {e}")
        return pd.DataFrame()
