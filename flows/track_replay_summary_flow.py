# flows/track_replay_summary_flow.py

import pandas as pd
from core.logger.logger import logger
from db.postgres_manager import run_query, insert_df
from core.time_context.time_context import get_simulation_date
from prefect import flow

@flow(name="Track Daily PPO Replay Transitions", log_prints=True)
def track_replay_summary_flow():
    today = get_simulation_date().date()
    logger.info(f"[PPO SUMMARY] Running for {today}")

    query = f"""
    SELECT * FROM ppo_buffer
    WHERE DATE(timestamp) = '{today}'
    """
    df = run_query(query)

    if df.empty:
        logger.warning("[PPO SUMMARY] No transitions found.")
        return

    # Basic stats
    num_transitions = len(df)
    avg_reward = df['reward'].mean()
    pct_enter = (df['action'] == 0).mean() * 100
    pct_hold = (df['action'] == 1).mean() * 100
    pct_exit = (df['action'] == 2).mean() * 100

    # Sharpe over past 7 days including today
    sharpe_query = f"""
    SELECT date, SUM(pnl) AS daily_pnl
    FROM paper_trades
    WHERE model_type = 'RL' AND date >= '{today - pd.Timedelta(days=6)}'
    GROUP BY date ORDER BY date;
    """
    df_sharpe = run_query(sharpe_query)
    if len(df_sharpe) >= 2:
        returns = df_sharpe['daily_pnl'].pct_change().dropna()
        rolling_sharpe = returns.mean() / returns.std()
    else:
        rolling_sharpe = 0.0

    # Reward by reason
    reason_summary = df.groupby("reason")["reward"].agg(["mean", "count"]).reset_index()
    reason_summary = reason_summary.rename(columns={"mean": "avg_reward", "count": "num_transitions"})
    reason_summary["date"] = today

    # Merge global stats to each row
    reason_summary["rolling_sharpe"] = rolling_sharpe
    reason_summary["pct_enter"] = pct_enter
    reason_summary["pct_hold"] = pct_hold
    reason_summary["pct_exit"] = pct_exit
    reason_summary["total_transitions"] = num_transitions
    reason_summary["avg_reward_overall"] = avg_reward

    insert_df(reason_summary, table="ppo_training_summary")
    logger.success("[PPO SUMMARY] Inserted reason-wise training summary.")

if __name__ == "__main__":
    track_replay_summary_flow()
