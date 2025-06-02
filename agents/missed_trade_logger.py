# agents/missed_trade_logger.py

from core.data_provider.data_provider import fetch_stock_data
from core.logger.logger import logger
from core.event_bus import publish_event
from db.db import SessionLocal
from core.time_context.time_context import get_simulation_date
import pandas as pd

def simple_backtest_profit(symbol: str, date: str) -> float:
    """
    A naive counterfactual: Buy at open, sell at close of the same day.
    """
    df = fetch_stock_data(symbol, start=date, end=date)
    if df is None or df.empty:
        return 0.0
    row = df.iloc[-1]
    return row["close"] - row["open"]

def get_all_candidates(date: str) -> list:
    # Load from recommendations
    from core.config.config import settings
    from core.data_provider.data_provider import load_data

    df = load_data(settings.recommendations_table)
    if df is None or df.empty:
        return []

    df = df[df["date"] == date]
    return df["stock"].unique().tolist()

def get_traded_today(date: str) -> list:
    from core.config.config import settings
    from core.data_provider.data_provider import load_data

    df = load_data(settings.trades_table)
    if df is None or df.empty:
        return []

    df = df[df["timestamp"].str.startswith(date)]
    return df["stock"].unique().tolist()

def run_missed_trade_logger():
    date = pd.to_datetime(get_simulation_date()).strftime("%Y-%m-%d")
    candidates = get_all_candidates(date)
    traded = get_traded_today(date)
    missed = [s for s in candidates if s not in traded]

    logger.info(f"📉 Found {len(missed)} missed trade candidates on {date}.")

    for symbol in missed:
        try:
            reward = simple_backtest_profit(symbol, date)
            publish_event("TRADE_CLOSE", {
                "symbol": symbol,
                "exit_price": None,
                "reward": reward,
                "timestamp": date + " 23:59",
                "strategy_config": {},
                "virtual": True
            })
        except Exception as e:
            logger.warning(f"⚠️ Could not simulate missed trade for {symbol}: {e}")

if __name__ == "__main__":
    run_missed_trade_logger()
