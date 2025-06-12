# core/skiplist/skiplist.py

from db.postgres_manager import run_query
from core.logger.logger import logger
from core.config.config import settings
from datetime import datetime, timedelta

def is_in_skiplist(stock: str, silent: bool = False) -> bool:
    try:
        result = run_query(
            """
            SELECT 1 FROM skiplist_stocks
            WHERE stock = :stock AND (expires_at IS NULL OR expires_at > NOW())
            LIMIT 1
            """,
            params={"stock": stock}
        )
        if result and not silent and getattr(settings, "log_skiplist_verbose", False):
            logger.info(f"⏭️ Skipping {stock} (in skiplist)")
        return bool(result)
    except Exception as e:
        logger.warning(f"⚠️ Failed to check skiplist for {stock}: {e}")
        return False

def add_to_skiplist(stock: str, reason: str = "unknown", ttl_days: int = None):
    try:
        FALLBACK_STOCKS = settings.fallback_stocks
        if stock.upper() in FALLBACK_STOCKS:
            logger.warning(f"🛑 Refused to add fallback stock '{stock}' to skiplist.")
            return

        query = """
        INSERT INTO skiplist_stocks (stock, reason, imported_at, expires_at)
        VALUES (:stock, :reason, now(), 
                CASE WHEN :ttl IS NOT NULL THEN now() + interval ':ttl days' ELSE NULL END)
        ON CONFLICT (stock) DO UPDATE
        SET reason = EXCLUDED.reason, imported_at = now(), 
            expires_at = CASE WHEN :ttl IS NOT NULL THEN now() + interval ':ttl days' ELSE NULL END;
        """
        run_query(query, params={"stock": stock, "reason": reason, "ttl": ttl_days}, fetchall=False)

        if getattr(settings, "log_skiplist_verbose", False):
            expiry_msg = f"(expires in {ttl_days}d)" if ttl_days else "(permanent)"
            logger.warning(f"⛔️ {stock} → skiplist: {reason} {expiry_msg}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to add {stock} to skiplist: {e}")


def remove_from_skiplist(stock: str):
    try:
        run_query(
            "DELETE FROM skiplist_stocks WHERE stock = :stock;",
            params={"stock": stock},
            fetchall=False
        )
    except Exception as e:
        logger.warning(f"⚠️ Failed to remove {stock} from skiplist: {e}")

def get_skiplist() -> list:
    try:
        result = run_query("SELECT stock FROM skiplist_stocks")
        return [row[0] for row in result] if result else []
    except Exception as e:
        logger.warning(f"⚠️ Failed to fetch skiplist: {e}")
        return []
