# run_feature_backfill_test.py
from core.feature_engineering.feature_provider import fetch_features_with_backfill
from core.logger.logger import logger
from datetime import datetime

stocks = ["RELIANCE", "SBIN", "INFY", "LT", "ICICIBANK"]
interval = "day"
start = datetime(2023, 1, 1)
end = datetime(2023, 1, 10)

for stock in stocks:
    logger.info(f"🔁 Backfilling features for {stock} @ {interval} (start={start}, end={end})")
    try:
        df = fetch_features_with_backfill(stock, interval, start=start, end=end)
        rows = len(df) if df is not None else 0
        logger.info(f"✅ ✅ Done: {stock} — {rows} rows saved.")
    except Exception as e:
        logger.error(f"❌ Failed: {stock} → {e}")
