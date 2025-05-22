# data_pipeline/zerodha_to_postgres.py

import os
import time
import pandas as pd
from datetime import datetime, timedelta

from core.logger import logger
from core.data_provider import save_data, load_data
from db.postgres_manager import run_query
from integrations.zerodha_fetcher import fetch_historical_data

DEFAULT_DAYS = 500
INSTRUMENTS_TABLE = "instruments"
PRICE_HISTORY_TABLE = "stock_price_history"
SKIPLIST_TABLE = "skiplist_stocks"

WHITELIST = []

def load_stock_list():
    try:
        df = load_data(INSTRUMENTS_TABLE)
        if df is None or df.empty:
            raise ValueError("❌ No instruments found via load_data.")
    except Exception as e:
        logger.warning(f"⚠️ load_data fallback for instruments: {e}")
        result = run_query(f"SELECT * FROM {INSTRUMENTS_TABLE}", fetchall=True)
        if not result:
            raise ValueError(f"❌ No instruments found even via fallback.")
        df = pd.DataFrame(result, columns=[
            'instrument_token', 'exchange_token', 'tradingtrading_symbol', 'name',
            'last_price', 'expiry', 'strike', 'tick_size', 'lot_size',
            'instrument_type', 'segment', 'exchange'
        ])

    df = df[
        (df["exchange"] == "NSE") &
        (df["instrument_type"] == "EQ")
    ]

    stocks = df['tradingtrading_symbol'].dropna().unique().tolist()

    result = run_query(f"SELECT symbol FROM {SKIPLIST_TABLE}", fetchall=True)
    skipped_trading_symbols = {row[0] for row in result}

    stocks = [s for s in stocks if s not in skipped_trading_symbols]
    stocks = [s for s in stocks if s in WHITELIST]

    return sorted(stocks)

def fetch_and_save_stock(symbol, days=DEFAULT_DAYS):
    try:
        logger.info(f"📥 Fetching {symbol}...")
        df = fetch_historical_data(symbol, days=days, interval="day")

        if df is None or df.empty:
            logger.warning(f"⚠️ No data fetched for {symbol}. Skipping.")
            return

        df["symbol"] = symbol
        base_cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
        df = df[base_cols]

        save_data(df, PRICE_HISTORY_TABLE, if_exists="append")
        logger.success(f"✅ {symbol}: {len(df)} rows saved.")

    except Exception as e:
        logger.error(f"❌ Error fetching {symbol}: {e}")

def main():
    logger.start("\n🚀 Starting Zerodha ➔ PostgreSQL fetch...")

    symbols = load_stock_list()
    logger.info(f"🔎 Total {len(symbols)} stocks to fetch...")

    for idx, symbol in enumerate(symbols, 1):
        logger.info(f"[{idx}/{len(symbols)}] {symbol}")
        fetch_and_save_stock(symbol, days=DEFAULT_DAYS)
        time.sleep(0.5)  # avoid API rate limits

    logger.success("\n🏁 Stock price history fetch complete!")

if __name__ == "__main__":
    main()
