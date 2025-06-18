# reports/weekly_snapshot.py → now daily snapshot

import pandas as pd
from datetime import datetime, timedelta
from core.data_provider.data_provider import load_data
from core.logger.logger import logger
from core.config.config import settings

TABLE_NAME = settings.trades_table


def compute_snapshot(start_date=None, end_date=None):
    df = load_data(TABLE_NAME)
    if df is None or df.empty:
        logger.warning("No trade data found for snapshot.")
        return None

    df["imported_at"] = pd.to_datetime(df["imported_at"])
    df = df.sort_values("imported_at")

    today = datetime.now().date()
    if end_date is None:
        end_date = today
    if start_date is None:
        start_date = today - timedelta(days=1)

    mask = (df["imported_at"].dt.date >= start_date) & (df["imported_at"].dt.date <= end_date)
    df_window = df[mask].copy()

    if df_window.empty:
        logger.warning("No trades in selected window.")
        return None

    trade_count = len(df_window)
    buy_count = len(df_window[df_window["action"] == "buy"])
    sell_count = len(df_window[df_window["action"] == "sell"])
    unique_stocks = df_window["stock"].nunique()
    avg_price = df_window["price"].mean()

    summary = {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "trades": trade_count,
        "buys": buy_count,
        "sells": sell_count,
        "stocks_traded": unique_stocks,
        "avg_price": round(avg_price, 2),
    }

    logger.success(f"Snapshot ready → {summary}")
    return summary


if __name__ == "__main__":
    print(compute_snapshot())
