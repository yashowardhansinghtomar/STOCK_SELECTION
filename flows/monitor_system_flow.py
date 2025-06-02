# flows/monitor_system_flow.py

import json
import pandas as pd
from prefect import flow
from core.logger.logger import logger
from core.time_context.time_context import get_simulation_date
from db.postgres_manager import run_query, insert_row
from core.system_state import get_system_config
from core.predict.policy_chooser import get_sharpe

@flow(name="Monitor System Flow", log_prints=True)
def monitor_system_flow():
    today = get_simulation_date().date()
    logger.info(f"[Monitor] Running system monitoring for {today}")

    # 1. RL Allocation & Sharpe Comparison
    config = get_system_config()
    rl_alloc = int(config.get("rl_allocation", 10))
    rl_sharpe = get_sharpe("RL")
    rf_sharpe = get_sharpe("RF")

    # 2. Reward Attribution Summary
    reward_counts = _get_reward_summary(today)

    # 3. Training Summary (stub for now)
    training_info = _get_training_summary(today)

    # 4. Persist
    insert_row("daily_system_summary", {
        "date": today,
        "rl_sharpe": rl_sharpe,
        "rf_sharpe": rf_sharpe,
        "rl_alloc": rl_alloc,
        "reward_counts_json": json.dumps(reward_counts),
        "training_info_json": json.dumps(training_info),
    })
    logger.success("[Monitor] Summary inserted.")

def _get_reward_summary(date: pd.Timestamp) -> dict:
    query = f"""
    SELECT reward_type, COUNT(*) as count
    FROM replay_log
    WHERE DATE(created_at) = '{date}'
    GROUP BY reward_type;
    """
    df = run_query(query)
    return dict(zip(df["reward_type"], df["count"])) if not df.empty else {}

def _get_training_summary(date: pd.Timestamp) -> dict:
    # Placeholder: Replace with real training metrics lookup if available
    return {
        "model": "PPO",
        "steps": 3000,
        "status": "ok"
    }

if __name__ == "__main__":
    monitor_system_flow()
