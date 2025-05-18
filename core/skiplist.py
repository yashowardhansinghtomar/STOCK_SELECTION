from db.postgres_manager import run_query
from core.logger import logger
from core.config import settings

def is_in_skiplist(stock: str, silent: bool = False) -> bool:
    try:
        result = run_query(
            "SELECT 1 FROM skiplist_stocks WHERE stock = :stock LIMIT 1",
            params={"stock": stock}
        )
        if result and not silent and getattr(settings, "log_skiplist_verbose", False):
            logger.info(f"⏭️ Skipping {stock} (in skiplist)")
        return bool(result)
    except Exception as e:
        logger.warning(f"⚠️ Failed to check skiplist for {stock}: {e}")
        return False

def add_to_skiplist(stock: str, reason: str = "unknown"):
    try:
        run_query(
            """
            INSERT INTO skiplist_stocks (stock, reason, imported_at)
            VALUES (:stock, :reason, now())
            ON CONFLICT (stock) DO UPDATE
            SET reason = EXCLUDED.reason, imported_at = now();
            """,
            params={"stock": stock, "reason": reason},
            fetchall=False
        )
        if getattr(settings, "log_skiplist_verbose", False):
            logger.warning(f"⛔️ {stock} → skiplist: {reason}")
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
