# bootstrap_trader.py

from datetime import datetime
import pandas as pd
from core.filter_model.run_filter import run_filter_model
from core.utils.market_conditions import get_volatility_regime
from core.phase_controller import PhaseController
from core.logger.logger import logger
from agents.execution.execution_agent_sql import ExecutionAgentSQL


def run_bootstrap_trader(today=None):
    today = pd.to_datetime(today or datetime.now().date())
    logger.info(f"🚨 LIVE BOOTSTRAP TRADER: Running for {today.date()}...")

    phase_controller = PhaseController(initial_phase=0)
    session = None  # Add SQLAlchemy session if needed
    exec_agent = ExecutionAgentSQL(session=session, dry_run=False)

    # STEP 1: Select stocks
    filtered_stocks = run_filter_model(today)

    # STEP 2: Generate trades (ε-greedy)
    trades = phase_controller.generate_trades(filtered_stocks, today)

    # STEP 3: Convert to DataFrame format expected by enter_trades
    trade_df = pd.DataFrame([{
        "symbol": t.symbol,
        "interval": t.meta.get("interval", "day"),
        "strategy_config": t.meta.get("strategy_config", {}),
        "source": t.meta.get("exploration_type", "live_bootstrap"),
        "sma_short": t.meta.get("sma_short"),
        "sma_long": t.meta.get("sma_long"),
        "rsi_thresh": t.meta.get("rsi_thresh"),
        "confidence": t.meta.get("confidence"),
        "sharpe": t.meta.get("sharpe"),
        "rank": t.meta.get("rank"),
        "trade_triggered": 1
    } for t in trades])

    # STEP 4: Load open positions
    open_pos = exec_agent.load_open_positions()

    # STEP 5: Execute trades via central agent
    exec_agent.enter_trades(trade_df, open_pos)

    logger.info(f"✅ Live bootstrap trading complete for {today.date()}. {len(trade_df)} trades attempted.")


if __name__ == "__main__":
    run_bootstrap_trader()
