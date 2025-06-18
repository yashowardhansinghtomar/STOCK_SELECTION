# diagnosis/simulate_history.py

import pandas as pd
from datetime import datetime
from core.time_context.time_context import set_simulation_date
from core.logger.logger import logger
from agents.planner.planner_agent_sql import PlannerAgentSQL
from agents.planner.intraday_planner_agent import IntradayPlannerAgent
from agents.memory.memory_agent import MemoryAgent
from models.train_exit_model import train_exit_model
from models.train_param_model import train_param_model
from models.train_stock_filter_model import train_stock_filter_model
from models.train_dual_model_sql import train_dual_model
from models.meta_strategy_selector import train_meta_model
import subprocess
import argparse


def daterange(start, end, step_days=1):
    cur = pd.to_datetime(start)
    end = pd.to_datetime(end)
    while cur <= end:
        yield cur
        cur += pd.Timedelta(days=step_days)


def simulate_and_bootstrap(start="2024-01-01", end="2025-04-01", weekly_only=False):
    logger.start("Starting full backfilled simulation...")

    for date in daterange(start, end):
        logger.info(f"🧭 Simulating date: {date.date()}...")
        set_simulation_date(date)

        if not weekly_only:
            IntradayPlannerAgent(dry_run=False).run()

        if date.weekday() == 0 or weekly_only:  # Monday
            PlannerAgentSQL(dry_run=False).run()

    logger.start("Simulation complete. Starting model training...")

    # === Train all ML models ===
    train_exit_model()
    train_param_model()
    train_stock_filter_model()
    train_dual_model()
    train_meta_model()

    logger.success("ML model training complete. Starting RL training...")

    try:
        subprocess.run(["python", "rl/train_rl_agent.py", "--freq", "day", "--steps", "1000000"])
        subprocess.run(["python", "rl/train_rl_intraday.py"])
    except Exception as e:
        logger.warning(f"RL training failed: {e}")

    logger.success("All models trained. O.D.I.N. is now live-ready.")
    logger.start("Updating Memory Agent (final cleanup and summary)...")

    MemoryAgent().update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2025-04-01")
    parser.add_argument("--weekly-only", action="store_true")
    args = parser.parse_args()

    simulate_and_bootstrap(start=args.start, end=args.end, weekly_only=args.weekly_only)
