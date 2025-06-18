# bootstrap/historical_bootstrap_runner.py

from datetime import datetime
import random

import numpy as np
import pandas as pd

from db.replay_buffer_sql import SQLReplayBuffer as ReplayBuffer
from agents.execution.execution_agent_sql import ExecutionAgentSQL
from models.run_stock_filter import run_stock_filter as run_filter_model
from bootstrap.simulate_trade_execution import simulate_trade_execution
from core.model_trainer.trainer import train_models
from core.market_calendar import get_trading_days
from bootstrap.phase_controller import PhaseController
from core.logger.logger import logger
from core.config.config import settings


# ─── Seed RNGs for reproducibility ─────────────────────────────────────────────
random.seed(42)
np.random.seed(42)


def run_historical_bootstrap(start_date: str, end_date: str):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date,   "%Y-%m-%d")
    trading_days = get_trading_days(start, end)

    replay_buffer = ReplayBuffer()
    exec_agent    = ExecutionAgentSQL(session=None, dry_run=True)

    # start with no open positions
    open_positions = pd.DataFrame(columns=[
        "stock", "entry_price", "entry_date", "quantity",
        "sma_short", "sma_long", "rsi_thresh",
        "strategy_config", "interval"
    ])

    # persistent phase controller
    phase_controller = PhaseController(initial_phase=0)

    logger.info(f"🚀 Starting historical bootstrap from {start_date} to {end_date}...")

    for date in trading_days:
        logger.info(f"\n📅 Simulating for {date.date()} | Phase {phase_controller.phase}")

        # 1️⃣ Phase check & auto-transition
        phase_controller.update_phase(replay_buffer)

        # 2️⃣ Exit any existing positions
        try:
            open_positions, exits = exec_agent.exit_trades(open_positions)
        except Exception as e:
            logger.warning(f"⚠️ exit_trades failed on {date.date()}: {e}")
            exits = pd.DataFrame([])

        # 2.a Log exits into the RL replay buffer
        for ex in exits.to_dict(orient="records"):
            replay_buffer.add(ex, tags={
                "phase": phase_controller.phase,
                "source": "historical_exit"
            })

        # 3️⃣ Filter stocks
        try:
            filtered = run_filter_model(date, lookback_only=True)
        except Exception as e:
            logger.warning(f"⚠️ run_filter_model failed on {date.date()}: {e}")
            filtered = []
        logger.info(f"🔎 {len(filtered)} stocks selected for {date.date()}")

        # 4️⃣ Generate trades
        try:
            trades = phase_controller.generate_trades(filtered, date)
        except Exception as e:
            logger.warning(f"⚠️ generate_trades failed on {date.date()}: {e}")
            trades = []
        logger.info(f"📈 {len(trades)} trades generated")

        intraday = sum(1 for t in trades if t.meta.get("interval") == "15minute")
        swing    = len(trades) - intraday
        avg_hold = np.mean([t.holding_period.total_seconds() / 60 for t in trades]) if trades else 0
        logger.info(f"📊 {intraday} intraday, {swing} swing | Avg hold: {avg_hold:.1f} min")

        # 5️⃣ Simulate execution & log to RL buffer
        executed_trades = []
        for trade in trades:
            try:
                result = simulate_trade_execution(trade, date)
                if not result:
                    continue
                executed_trades.append(result)

                replay_buffer.add(result, tags={
                    "phase": phase_controller.phase,
                    "source": phase_controller.get_source_label(),
                    "exploration_type": trade.meta.get("exploration_type", "random")
                })
            except Exception as e:
                logger.warning(f"⚠️ simulate_trade_execution failed for {trade.symbol} on {date.date()}: {e}")

        logger.info(f"✅ Executed {len(executed_trades)} / {len(trades)} trades")

        # 6️⃣ Enter new positions & persist them
        sig_records = []
        for r in executed_trades:
            sig_records.append({
                "symbol":          r["symbol"],
                "sma_short":       r["meta"].get("sma_short"),
                "sma_long":        r["meta"].get("sma_long"),
                "rsi_thresh":      r["meta"].get("rsi_thresh"),
                "confidence":      r["meta"].get("confidence"),
                "rank":            r["meta"].get("rank"),
                "strategy_config": r["meta"].get("strategy_config", {}),
                "source":          phase_controller.get_source_label(),
                "interval":        r["meta"].get("interval", "day"),
                "direction":       r["meta"].get("direction", 1)
            })
        sig_df = pd.DataFrame(sig_records)

        try:
            open_positions = exec_agent.enter_trades(sig_df, open_positions)
        except Exception as e:
            logger.warning(f"⚠️ enter_trades failed on {date.date()}: {e}")

        # 7️⃣ Conditional retraining
        buffer_size = replay_buffer.size()
        if (date.weekday() == 4) or (buffer_size > settings.retrain.training_data_threshold):
            logger.info(f"📚 Retraining models (buffer={buffer_size}) at {date.date()}")
            try:
                train_models(replay_buffer, up_to_date=date)
            except Exception as e:
                logger.warning(f"⚠️ train_models failed on {date.date()}: {e}")

    logger.success("✅ Historical bootstrap completed.")
