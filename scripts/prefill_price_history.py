# scripts/prefill_price_history.py

import pandas as pd
from datetime import datetime
from core.logger import logger
from core.data_provider import fetch_stock_data, save_data
from integrations.zerodha_fetcher import INTERVAL_LIMIT_DAYS
from pathlib import Path

INTERVALS = ["day", "15minute", "60minute"]
MAX_DAYS = {
    "day": 2000,
    "15minute": 60,
    "60minute": 60,
}

def load_symbols():
    path = Path("data/instruments.csv")
    if not path.exists():
        raise FileNotFoundError("Missing instruments.csv in /data")
    df = pd.read_csv(path)
    return df["tradingsymbol"].dropna().unique().tolist()

def prefill_all():
    symbols = load_symbols()
    logger.start(f"📦 Prefilling price history for {len(symbols)} stocks...")

    for interval in INTERVALS:
        for sym in symbols:
            try:
                df = fetch_stock_data(sym, interval=interval, days=MAX_DAYS[interval])
                if df is not None and not df.empty:
                    logger.success(f"✅ {sym} [{interval}] → {len(df)} rows")
                else:
                    logger.warning(f"⚠️ {sym} [{interval}] → No data")
            except Exception as e:
                logger.warning(f"❌ {sym} [{interval}] failed: {e}")

    logger.success("🎯 Done pre-filling price history.")

if __name__ == "__main__":
    prefill_all()
