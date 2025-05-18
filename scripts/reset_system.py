# scripts/reset_system.py

from core.logger import logger
from db.db_router import run_query
import os
import shutil
from pathlib import Path
import argparse
import sys

# Tables to wipe
TABLES = [
    "stock_fundamentals",
    "stock_features",
    "recommendations",
    "trades",
    "open_positions",
    "paper_trades",
    "training_data",
    "model_store",
    "model_metadata",
    "skiplist_stocks"
]

def drop_partitioned_feature_tables():
    logger.info("🧹 Dropping stock_features partitioned tables...")
    try:
        tables = run_query("SELECT tablename FROM pg_tables WHERE tablename LIKE 'stock_features_%'", fetchall=True)
        for (t,) in tables:
            run_query(f'DROP TABLE IF EXISTS \"{t}\"', fetchall=False)
            logger.info(f"🗑️ Dropped: {t}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to drop partitioned tables: {e}")

def delete_model_files():
    model_dir = Path("project_data/models")
    if model_dir.exists():
        shutil.rmtree(model_dir)
        logger.info(f"🧨 Deleted all model files in {model_dir}")
        model_dir.mkdir(parents=True, exist_ok=True)

def clear_cache_dirs():
    cache_dirs = [
        Path("cache/fundamentals"),
        Path("cache/price_history"),
        Path("cache")
    ]
    for path in cache_dirs:
        if path.exists():
            shutil.rmtree(path)
            logger.info(f"🧹 Cleared: {path}")
            path.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes-i-know", action="store_true", help="Confirm you want to reset the entire system.")
    args = parser.parse_args()

    if not args.yes_i_know:
        logger.error("❌ This will wipe all models, trades, and data. Re-run with --yes-i-know to proceed.")
        sys.exit(1)

    logger.start("🚨 RESETTING SYSTEM — CLEAN SLATE 🚨")

    for table in TABLES:
        try:
            run_query(f'TRUNCATE TABLE \"{table}\" RESTART IDENTITY', fetchall=False)
            logger.info(f"✅ Cleared: {table}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to truncate {table}: {e}")

    drop_partitioned_feature_tables()
    delete_model_files()
    clear_cache_dirs()

    logger.success("✅ System reset complete.")

if __name__ == "__main__":
    main()
