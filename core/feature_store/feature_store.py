import duckdb
import os
import pandas as pd
from core.feature_engineering.feature_computer import compute_and_prepare_features
from core.logger.logger import logger
from core.time_context.time_context import get_simulation_date

FEATURE_DB_PATH = "data/feature_store.duckdb"
os.makedirs(os.path.dirname(FEATURE_DB_PATH), exist_ok=True)

# One-time setup for schema
with duckdb.connect(FEATURE_DB_PATH) as con:
    con.execute("""
    CREATE TABLE IF NOT EXISTS stock_features (
        stock TEXT,
        date DATE,
        interval TEXT,
        features STRUCT(
            sma_short DOUBLE,
            sma_long DOUBLE,
            rsi_thresh DOUBLE,
            macd DOUBLE,
            vwap DOUBLE,
            atr_14 DOUBLE,
            bb_width DOUBLE,
            macd_histogram DOUBLE,
            price_compression DOUBLE,
            stock_encoded INT,
            volatility_10 DOUBLE,
            volume_spike BOOLEAN,
            vwap_dev DOUBLE
        ),
        PRIMARY KEY(stock, date, interval)
    )
    """)
    con.commit()

def get_cached_features(stock: str, date: str, interval: str) -> pd.DataFrame:
    query = f"""
        SELECT * FROM stock_features
        WHERE stock = '{stock}'
        AND date = '{date}'
        AND interval = '{interval}'
    """
    try:
        with duckdb.connect(FEATURE_DB_PATH, read_only=True) as con:
            result = con.execute(query).df()
        if not result.empty:
            f = pd.json_normalize(result["features"]).add_prefix("")
            f["stock"] = stock
            f["date"] = date
            f["interval"] = interval
            return f
    except Exception as e:
        logger.warning(f" DuckDB query failed for {stock} @ {interval}: {e}")
    return pd.DataFrame()

def insert_features(stock: str, date: str, interval: str, features: dict):
    try:
        with duckdb.connect(FEATURE_DB_PATH) as con:
            con.execute(
                "INSERT OR REPLACE INTO stock_features VALUES (?, ?, ?, ?)",
                (stock, date, interval, features)
            )
            con.commit()
    except Exception as e:
        logger.warning(f" Insert failed for {stock} @ {interval}: {e}")

def get_or_compute(stock: str, interval: str, date: str = None) -> pd.DataFrame:
    date = date or str(get_simulation_date().date())
    cached = get_cached_features(stock, date, interval)
    if not cached.empty:
        return cached

    try:
        computed_df = compute_and_prepare_features(stock, interval=interval, date=date)
        if not computed_df.empty:
            last_row = computed_df.iloc[-1].to_dict()
            insert_features(stock, date, interval, last_row)
            return pd.DataFrame([last_row])
    except Exception as e:
        logger.warning(f"⚠️ Could not compute features for {stock} @ {interval}: {e}")

    return pd.DataFrame()
