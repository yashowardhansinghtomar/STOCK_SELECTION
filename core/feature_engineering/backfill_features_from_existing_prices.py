# core/feature_engineering/backfill_features_from_existing_prices.py

from concurrent.futures import ThreadPoolExecutor, as_completed
from core.feature_engineering.feature_provider import fetch_features_with_backfill
from core.config.config import settings
from core.logger.logger import logger
from db.postgres_manager import get_all_symbols as get_stock_universe
import logging
import os
from tqdm import tqdm


INTERVALS = ["day", "60minute", "15minute", "minute"]
MAX_WORKERS = 8

LOG_FILE = "backfill_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def backfill(stock: str, interval: str):
    try:
        table = settings.interval_feature_table_map.get(interval)
        if not table:
            logging.error(f"[ERROR] Unknown interval: {interval}")
            return "invalid", stock, interval

        df = fetch_features_with_backfill(stock, interval)
        if df is None or df.empty:
            logging.warning(f"[WARN] No usable features for {stock} @ {interval}")
            return "empty", stock, interval

        logging.info(f"[OK] Features OK: {stock} @ {interval}")
        return "success", stock, interval
    except Exception as e:
        logging.error(f"[ERROR] Failed {stock} @ {interval}: {e}")
        return "error", stock, interval

if __name__ == "__main__":
    stocks = get_stock_universe()
    tasks = [(stock, interval) for stock in stocks for interval in INTERVALS]

    logging.info(f"Starting threaded backfill for {len(tasks)} tasks using {MAX_WORKERS} workers...")

    summary = {"success": 0, "empty": 0, "error": 0, "invalid": 0}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(backfill, stock, interval): (stock, interval) for stock, interval in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Backfill Progress", ncols=100):
            status, stock, interval = future.result()
            summary[status] += 1
            logging.info(f"{status.upper()}: {stock} @ {interval}")

    logging.info(f"Summary: {summary}")
