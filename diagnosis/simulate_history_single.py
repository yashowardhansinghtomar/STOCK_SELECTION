# simulate_history_single.py

import argparse
import pandas as pd
from datetime import datetime, timedelta
from agents.planner_agent_sql import PlannerAgentSQL
from core.time_context import set_simulation_date
from core.logger import logger


def simulate_stock_over_range(symbol: str, start_date: str, end_date: str):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    logger.start(f"\n🕰️ Simulating history for {symbol} from {start_date} to {end_date}...")

    current = start
    while current <= end:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        sim_date = current.strftime("%Y-%m-%d")
        set_simulation_date(sim_date)

        logger.info(f"\n📆 Simulating {symbol} on {sim_date}...")
        agent = PlannerAgentSQL(
            force_fetch=False,
            force_enrich=False,
            force_filter=False,
            force_eval=True,
            dry_run=True,
            stock_whitelist=[symbol]
        )
        agent.run_weekly_routine()

        current += timedelta(days=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", required=True, help="Stock symbol")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    simulate_stock_over_range(args.stock, args.start, args.end)
