# bootstrap/bootstrap_trader.py

from datetime import datetime
import pandas as pd
from models.run_stock_filter import run_stock_filter
from core.market_conditions import get_volatility_regime
from bootstrap.phase_controller import PhaseController
from core.logger.logger import logger
from agents.execution.execution_agent_sql import ExecutionAgentSQL

from agents.strategy.rl_strategy_agent import RLStrategyAgent
from db.replay_buffer_sql import policy_converged, SQLReplayBuffer

def run_bootstrap_trader(today=None, phase_controller=None):
    # Normalize today as a pandas Timestamp (date-only)
    today = pd.to_datetime(today or datetime.now().date())
    logger.info(f"🚨 LIVE BOOTSTRAP TRADER: Running for {today.date()}...")

    # Initialize executor and replay buffer
    exec_agent    = ExecutionAgentSQL(session=None, dry_run=False)
    replay_buffer = SQLReplayBuffer()

    # STEP 1: Filter stocks
    filtered_stocks = run_stock_filter(today)

    # STEP 2: Advance phase (or init if first call)
    if not phase_controller:
        phase_controller = PhaseController(initial_phase=0)
    phase_controller.update_phase(replay_buffer)
    real_trades = replay_buffer.count_real_trades()
    logger.info(f"🔄 Phase {phase_controller.phase} | ε={phase_controller.epsilon:.2f} | Trades={real_trades}")

    # STEP 3: Generate trades
    trades = []
    if phase_controller.phase >= 2 and policy_converged():
        logger.info("🤖 Phase ≥2 & PPO converged — using RLStrategyAgent for trades.")
        # Pass today into the RL agent
        strategy_agent = RLStrategyAgent(today=today)
        trades = strategy_agent.generate_trades(filtered_stocks, today)
        # Log RL signals into the SQL replay buffer
        for t in trades:
            if t.get("source") in ("ppo", "rl_agent"):
                replay_buffer.add(
                    t,
                    tags={"phase": phase_controller.phase, "source": t.get("source")}
                )
    else:
        logger.info("🧪 Phase <2 — ε-greedy exploration via PhaseController.")
        trades = phase_controller.generate_trades(filtered_stocks, today)

    # STEP 4: Build the DataFrame of signals for entry_trades
    trade_df = pd.DataFrame([{
        "symbol":           t["symbol"],
        "interval":         t.get("interval", "day"),
        "strategy_config":  t.get("strategy_config", {}),
        "source":           t.get("source", "live_bootstrap"),
        "sma_short":        t.get("sma_short"),
        "sma_long":         t.get("sma_long"),
        "rsi_thresh":       t.get("rsi_thresh"),
        "confidence":       t.get("confidence"),
        "sharpe":           t.get("sharpe"),
        "rank":             t.get("rank"),
        "trade_triggered":  1
    } for t in trades])

    # STEP 5: Execute trades and persist state
    open_positions = exec_agent.load_open_positions()
    open_positions = exec_agent.enter_trades(trade_df, open_positions)
    # exec_agent.run() also handles exits & persistence if you prefer:
    exec_agent.run()

    logger.info(f"✅ Live bootstrap trading complete for {today.date()}.")

if __name__ == "__main__":
    run_bootstrap_trader()
