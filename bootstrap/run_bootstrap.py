# bootstrap/run_bootstrap.py

import argparse
from datetime import datetime
from bootstrap.historical_bootstrap_runner import run_historical_bootstrap
from core.market_calendar import is_market_holiday
from core.data_provider.data_provider import fetch_stock_data
from db.models import FilterModelPrediction
from db.db import SessionLocal
from core.logger.logger import logger


def prefetch_minute_bars(sim_date: str, top_n: int = 300):
    date = datetime.strptime(sim_date, "%Y-%m-%d").date()
    session = SessionLocal()
    try:
        preds = (
            session.query(FilterModelPrediction)
            .filter(FilterModelPrediction.date == date)
            .order_by(FilterModelPrediction.score.desc())
            .limit(top_n)
            .all()
        )
        symbols = sorted({p.stock for p in preds})
        logger.info(f"📡 Prefetching 1m data for top {len(symbols)} stocks on {date}...")
        for symbol in symbols:
            try:
                fetch_stock_data(symbol, start=sim_date, end=sim_date, interval="minute")
            except Exception as e:
                logger.warning(f"⚠ Failed to prefetch 1m for {symbol}: {e}")
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description="Run Historical Bootstrap Simulation")
    parser.add_argument("--start", type=str, required=True, help="Start date in YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="End date in YYYY-MM-DD")

    args = parser.parse_args()
    start_date = datetime.strptime(args.start, "%Y-%m-%d")

    if is_market_holiday(start_date):
        logger.warning(f"⚠️ Skipping {start_date.date()} — market holiday")
        return

    prefetch_minute_bars(args.start)  # 🧊 Ensure 1m bars are cached before simulation

    run_historical_bootstrap(args.start, args.end)


if __name__ == "__main__":
    main()
