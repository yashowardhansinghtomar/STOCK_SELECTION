# scripts/validate_duckdb_feature_coverage.py

import pandas as pd
from datetime import datetime, timedelta
from core.feature_store.feature_store import get_cached_features
from core.logger.logger import logger

# Customize this list based on your active universe
STOCKS = ["RELIANCE", "TCS", "HDFCBANK", "INFY"]
INTERVALS = ["day", "15minute", "60minute"]
DAYS_BACK = 5  # recent trading days to check

def get_recent_dates(n=5):
    today = datetime.today().date()
    return [(today - timedelta(days=i)).isoformat() for i in range(n)]

def validate_duckdb_cache():
    missing = []
    for stock in STOCKS:
        for interval in INTERVALS:
            for date in get_recent_dates(DAYS_BACK):
                df = get_cached_features(stock, date, interval)
                if df.empty:
                    logger.warning(f"🚨 Missing: {stock} @ {interval} on {date}")
                    missing.append((stock, interval, date))
                else:
                    logger.info(f"✅ OK: {stock} @ {interval} on {date}")
    logger.info(f"\n🎯 Checked {len(STOCKS) * len(INTERVALS) * DAYS_BACK} combinations.")
    logger.info(f"❌ Missing {len(missing)} entries.")
    return missing

if __name__ == "__main__":
    validate_duckdb_cache()
