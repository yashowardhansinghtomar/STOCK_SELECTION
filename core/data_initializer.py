import os
import pandas as pd
from core.logger import logger
from db.postgres_manager import read_table
from db.conflict_utils import insert_with_conflict_handling
from core.data_provider import fetch_stock_data
from integrations.zerodha_fetcher import fetch_historical_data

def ensure_price_history_prefilled(current_date=None, silent=False, stock_list=None):
    """
    Ensures price history is populated in SQL for selected stocks in stock_fundamentals.
    Uses whitelist if provided. Skips if enough data is already present.
    """
    try:
        fundamentals = read_table("stock_fundamentals")
        symbols = fundamentals["stock"].unique()

        # 🔒 Apply whitelist from param or env
        if stock_list is None:
            env = os.getenv("STOCK_WHITELIST")
            if env:
                stock_list = [s.strip() for s in env.split(",") if s.strip()]
        if stock_list:
            symbols = [s for s in symbols if s in stock_list]
    except Exception as e:
        logger.error(f"❌ Failed to load fundamentals for price prefill: {e}")
        return

    for stock in symbols:
        try:
            existing = fetch_stock_data(stock, end=current_date, days=200)
            if existing is not None and len(existing) >= 50:
                continue  # ✅ Already have enough history
        except Exception:
            pass

        try:
            df = fetch_historical_data(stock, end=current_date, days=200, interval="day")
            if df is not None and not df.empty:
                insert_with_conflict_handling(df, "stock_price_history")
                if not silent:
                    logger.info(f"💾 Prefilled price history for {stock}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to prefill price for {stock}: {e}")
