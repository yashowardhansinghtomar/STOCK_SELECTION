import pandas as pd
from core.logger.logger import logger
from core.time_context.time_context import get_simulation_date
from db.postgres_manager import run_query
from core.event_bus import emit_event
from prefect import flow

@flow(name="Reward Attribution", log_prints=True)
def reward_attribution_flow():
    today = get_simulation_date().date()
    logger.info(f"[ATTRIBUTION] Running for {today}")

    query = f"""
    SELECT * FROM paper_trades
    WHERE model_type = 'RL' AND date = '{today}' AND exit_price IS NOT NULL
    """
    df = run_query(query)
    if df.empty:
        logger.warning("[ATTRIBUTION] No RL trades closed today.")
        return

    for _, row in df.iterrows():
        duration = (pd.to_datetime(row['exit_time']) - pd.to_datetime(row['entry_time'])).total_seconds() / 60
        holding_cost = duration * 0.001  # per-minute cost
        slippage_penalty = max(0, (row['planned_exit'] - row['exit_price']) / row['exit_price']) if row.get('planned_exit') else 0
        missed_pnl = row.get('max_possible_pnl', row['pnl']) - row['pnl']
        capital_efficiency = row['pnl'] / (row['entry_price'] * duration) if duration > 0 else 0

        event = {
            "event_type": "TRADE_CLOSE",
            "symbol": row['stock'],
            "timestamp": row['exit_time'],
            "reward": row['pnl'],
            "interval": row['interval'],
            "strategy_config": row.get("strategy_config", {}),
            "missed_pnl": missed_pnl,
            "holding_cost": holding_cost,
            "slippage_penalty": slippage_penalty,
            "capital_efficiency": capital_efficiency,
        }

        emit_event(event)
        logger.info(f"[ATTRIBUTION] Emitted for {row['stock']} | reward: {row['pnl']:.2f}, hold: {holding_cost:.2f}")

if __name__ == "__main__":
    reward_attribution_flow()
