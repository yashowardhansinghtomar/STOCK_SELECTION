# diagnosis/clear_all_sql_tables.py

from db.db_router import execute_raw_sql
from core.logger.logger import logger

TABLES = [
    "stock_fundamentals",
    "ml_selected_stocks",
    "recommendations",
    "open_positions",
    "training_data",
    "paper_trades"
]

def clear_all_tables():
    for table in TABLES:
        try:
            logger.info(f"🧹 Clearing table: {table}")
            execute_raw_sql(f"DELETE FROM {table}")
            logger.success(f"✅ Cleared: {table}")
        except Exception as e:
            logger.warnings(f"⚠️ Could not clear {table}: {e}")

if __name__ == "__main__":
    clear_all_tables()
