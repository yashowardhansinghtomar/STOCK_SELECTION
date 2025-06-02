# core/precompute_features.py
import pandas as pd
from datetime import datetime, timedelta
from core.config.config import settings
from core.logger.logger import logger
from core.data_provider.data_provider import fetch_stock_data
from db.db import SessionLocal
from sqlalchemy.sql import text
from utils.time_utils import to_naive_utc
import ta
import argparse

REQUIRED_FEATURES = [
    "sma_short", "sma_long", "rsi_thresh", "macd", "vwap", "atr_14",
    "bb_width", "macd_histogram", "price_compression", "stock_encoded",
    "volatility_10", "volume_spike", "vwap_dev"
]
INTERVALS = ["day", "60minute", "15minute"]
DAYS_LOOKBACK = 90


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sma_short"] = df["close"].rolling(window=5).mean()
    df["sma_long"] = df["close"].rolling(window=20).mean()
    df["rsi_thresh"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_histogram"] = macd.macd_diff()

    df["vwap"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()
    df["vwap_dev"] = (df["close"] - df["vwap"]) / df["vwap"]

    df["atr_14"] = ta.volatility.AverageTrueRange(high=df["high"], low=df["low"], close=df["close"]).average_true_range()

    bb = ta.volatility.BollingerBands(close=df["close"])
    df["bb_width"] = bb.bollinger_wband()

    df["price_compression"] = (df["high"] - df["low"]) / df["close"]
    df["volatility_10"] = df["close"].pct_change().rolling(10).std()
    df["volume_spike"] = df["volume"] > (df["volume"].rolling(20).mean() * 2)

    return df.dropna(subset=REQUIRED_FEATURES)


def insert_feature_row(session, table_name, row, refresh=False):
    if refresh:
        delete_sql = text(f"DELETE FROM {table_name} WHERE stock = :stock AND date = :date")
        session.execute(delete_sql, {"stock": row["stock"], "date": row["date"]})

    sql = text(f"""
        INSERT INTO {table_name} (
            stock, date, sma_short, sma_long, rsi_thresh, macd, vwap, atr_14,
            bb_width, macd_histogram, price_compression, stock_encoded,
            volatility_10, volume_spike, vwap_dev
        ) VALUES (
            :stock, :date, :sma_short, :sma_long, :rsi_thresh, :macd, :vwap, :atr_14,
            :bb_width, :macd_histogram, :price_compression, :stock_encoded,
            :volatility_10, :volume_spike, :vwap_dev
        )
        ON CONFLICT (stock, date) DO NOTHING;
    """)
    session.execute(sql, row)


def enrich_and_store(stock: str, interval: str, refresh: bool = False):
    table = settings.interval_feature_table_map.get(interval)
    if not table:
        logger.error(f"Unknown interval: {interval}")
        return

    logger.info(f"📊 Computing features for {stock} @ {interval}...")
    df = fetch_stock_data(stock, interval=interval, days=DAYS_LOOKBACK)
    if df is None or df.empty:
        logger.warnings(f"⚠️ No price data for {stock} @ {interval}")
        return

    df = compute_features(df)
    if df.empty:
        logger.warnings(f"⚠️ No features computed for {stock} @ {interval}")
        return

    df["stock"] = stock
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["stock_encoded"] = hash(stock) % 10000

    session = SessionLocal()
    inserted = 0
    try:
        for _, row in df.iterrows():
            insert_feature_row(session, table, row.to_dict(), refresh=refresh)
            inserted += 1
        session.commit()
        logger.info(f"✅ {inserted} features inserted for {stock} @ {interval}")
    except Exception as e:
        session.rollback()
        logger.error(f"❌ Failed to insert features for {stock} @ {interval}: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Force overwrite existing features")
    args = parser.parse_args()

    from core.time_context.time_context import get_stock_universe
    stocks = get_stock_universe()
    for stock in stocks:
        for interval in INTERVALS:
            enrich_and_store(stock, interval, refresh=args.refresh)
