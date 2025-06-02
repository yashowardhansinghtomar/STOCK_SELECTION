# flows/clean_skiplist_flow.py

from prefect import flow
from core.logger.logger import logger
from db.postgres_manager import run_query
from core.time_context.time_context import get_simulation_date

@flow(name="Clean Expired Skiplist Stocks", log_prints=True)
def clean_skiplist_flow():
    today = get_simulation_date().date()
    logger.info(f"[SKIPLIST CLEANER] Running for {today}")

    query = """
    DELETE FROM skiplist_stocks
    WHERE expires_at IS NOT NULL AND expires_at < CURRENT_DATE;
    """
    try:
        run_query(query, fetchall=False)
        logger.success("[SKIPLIST CLEANER] Expired entries removed.")
    except Exception as e:
        logger.warning(f"[SKIPLIST CLEANER] Failed to clean skiplist: {e}")

if __name__ == "__main__":
    clean_skiplist_flow()
