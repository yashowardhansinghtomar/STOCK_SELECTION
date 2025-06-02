# jobs/recheck_expired_skips.py

from db.postgres_manager import run_query
from core.skiplist.skiplist import remove_from_skiplist
from core.logger.logger import logger
from datetime import datetime

def recheck_stock(stock: str) -> bool:
    """
    Placeholder for actual recheck logic.
    Currently assumes recheck always succeeds.
    Replace with fetch or retry logic.
    """
    logger.info(f"🔄 Rechecking expired skip: {stock}")
    # TODO: Add real check, like trying to fetch features or price data
    return True  # assume eligible again

def run_expired_skip_recheck():
    query = """
    SELECT stock FROM skiplist_stocks
    WHERE expires_at IS NOT NULL AND expires_at <= NOW()
    """
    df = run_query(query)
    if df is None or df.empty:
        logger.info("✅ No expired skiplist entries found.")
        return

    for stock in df["stock"]:
        try:
            if recheck_stock(stock):
                remove_from_skiplist(stock)
                logger.info(f"✅ Removed {stock} from skiplist after recheck.")
        except Exception as e:
            logger.warning(f"⚠️ Failed recheck for {stock}: {e}")

if __name__ == "__main__":
    run_expired_skip_recheck()
