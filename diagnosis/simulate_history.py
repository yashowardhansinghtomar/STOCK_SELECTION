# diagnosis/simulate_history.py

import os
import argparse
from datetime import datetime, timedelta
from agents.planner_agent_sql import PlannerAgentSQL
from core.time_context import set_simulation_date
from core.logger import logger  # existing logger
import logging

# Setup file logger specifically for simulate_history
log_dir = "logs/history_simulation"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"simulate_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

logger.addHandler(file_handler)
logger.info(f"📂 Writing simulate_history logs to {log_file}")

def daterange(start_date, end_date):
    """Yield one date per week between start and end."""
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=7)

def parse_args():
    parser = argparse.ArgumentParser(description="Run historical simulation.")
    parser.add_argument("--start", required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end", required=True, help="End date in YYYY-MM-DD format")
    parser.add_argument("--stock", required=False, help="(Optional) Only simulate this stock")
    return parser.parse_args()

def main():
    args = parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    stock_list = [args.stock.upper()] if args.stock else None

    print(f"\n🕰️ Starting historical simulation from {start_date} to {end_date}")
    if stock_list:
        print(f"🎯 Focusing on single stock: {stock_list[0]}")
    else:
        print("🎯 Evaluating full ML-selected stock universe.")

    for current_start in daterange(start_date, end_date):
        sim_date = current_start
        os.environ["SIMULATED_DATE"] = sim_date.strftime("%Y-%m-%d")

        print(f"\n📆 Simulating week starting {sim_date}...")

        try:
            agent = PlannerAgentSQL(
                force_fetch=False,
                force_enrich=False,
                force_filter=False,
                force_eval=False,
                dry_run=False,
                stock_whitelist=stock_list,
            )
            agent.run_weekly_routine()

        except Exception as e:
            print(f"❌ Simulation failed for {sim_date}: {e}")

    print("\n🏁 Historical simulation complete.")

if __name__ == "__main__":
    main()
