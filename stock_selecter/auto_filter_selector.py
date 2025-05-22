# stock_selecter/auto_filter_selector.py

import os
from datetime import datetime
import pandas as pd

from core.logger import logger
from core.config import settings
from core.data_provider import load_data
from core.time_context import get_simulation_date
from stock_selecter.stock_screener import run_stock_filter
from stock_selecter.fallback_technical_filter import run_technical_filter

FILTERS = ["growth", "momentum", "value", "high_volatility", "small_cap_gems", "defensive"]
MIN_REQUIRED_STOCKS = 5
FILTER_LOG = os.path.join("stock_selecter", "filter_usage_log.csv")


def auto_select_filter() -> str:
    today = pd.to_datetime(get_simulation_date()).normalize()

    if settings.use_fundamentals:
        df = load_data(settings.fundamentals_table)
        if df is None or df.empty:
            raise ValueError("⚠️ No fundamental data found in SQL.")

        for f in FILTERS:
            logger.info(f"🔍 Trying filter: '{f}'")
            filtered_df = run_stock_filter(f)

            # Only include stocks matching the current SIMULATED_DATE
            if "imported_at" in filtered_df.columns:
                filtered_df["imported_at"] = pd.to_datetime(filtered_df["imported_at"]).dt.normalize()
                filtered_df = filtered_df[filtered_df["imported_at"] == today]

            count = len(filtered_df) if filtered_df is not None else 0
            if count >= MIN_REQUIRED_STOCKS:
                logger.success(f"✅ Filter '{f}' selected with {count} stocks.")
                chosen_filter = f
                break
            else:
                logger.warning(f"❌ Filter '{f}' returned only {count} stocks (<{MIN_REQUIRED_STOCKS}).")
        else:
            raise RuntimeError(f"🚫 No filter produced ≥{MIN_REQUIRED_STOCKS} stocks.")
    else:
        logger.info("⚙️ Skipping fundamentals — using no-fundamental mode.")
        filtered_df, rejected_reasons = run_technical_filter(return_reasons=True)
        chosen_filter = "fallback"

        if len(filtered_df) < MIN_REQUIRED_STOCKS:
            for symbol, reasons in rejected_reasons.items():
                reason_str = " | ".join(r if isinstance(r, str) else str(r) for r in reasons) if isinstance(reasons, (list, tuple)) else str(reasons)
                logger.warning(f"{symbol}: ❌ {reason_str}")
            raise RuntimeError(f"🚫 Fallback technical filter produced <{MIN_REQUIRED_STOCKS} stocks.")

        logger.success(f"✅ Fallback technical filter selected {len(filtered_df)} stocks.")

    usage = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filter_used": chosen_filter,
        "num_stocks": len(filtered_df)
    }
    pd.DataFrame([usage]).to_csv(
        FILTER_LOG,
        mode="a" if os.path.exists(FILTER_LOG) else "w",
        header=not os.path.exists(FILTER_LOG),
        index=False
    )

    return chosen_filter
