# bootstrap/historical_bootstrap_runner.py

from datetime import datetime, timedelta
import random
from db.replay_buffer_sql import SQLReplayBuffer as ReplayBuffer
from models.run_stock_filter import run_stock_filter as run_filter_model
from bootstrap.simulate_trade_execution import simulate_trade_execution
from core.model_trainer.trainer import train_models
from core.market_calendar import get_trading_days
from bootstrap.phase_controller import PhaseController
from core.logger.logger import logger
import numpy as np


def run_historical_bootstrap(start_date: str, end_date: str):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    trading_days = get_trading_days(start, end)

    replay_buffer = ReplayBuffer()
    phase_controller = PhaseController(initial_phase=0)

    logger.info(f"🚀 Starting historical bootstrap from {start_date} to {end_date}...")

    for date in trading_days:
        logger.info(f"\n📅 Simulating for {date.date()} | Phase {phase_controller.phase}")

        # Phase check and auto-transition logic
        phase_controller.update_phase(replay_buffer)

        # Step 1: Get filtered stocks for that day using lookback-only features
        filtered_stocks = run_filter_model(date, lookback_only=True)
        logger.info(f"🔎 {len(filtered_stocks)} stocks selected by filter model on {date.date()}")

        # Step 2: Generate trades (random param config for early phases)
        trades = phase_controller.generate_trades(filtered_stocks, date)
        logger.info(f"📈 {len(trades)} trades generated")

        intraday = sum(1 for t in trades if t.meta.get("interval") == "15minute")
        swing = len(trades) - intraday
        if trades:
            avg_hold_min = np.mean([t.holding_period.total_seconds() / 60 for t in trades])
        else:
            avg_hold_min = 0
        logger.info(f"📊 Breakdown: {intraday} intraday, {swing} swing | Avg hold: {avg_hold_min:.1f} min")

        # Step 3: Simulate execution using minute bars and realism
        executed_trades = []
        for trade in trades:
            result = simulate_trade_execution(trade, date)
            if result:
                executed_trades.append(result)

        logger.info(f"✅ Executed {len(executed_trades)} / {len(trades)} trades successfully")

        # Step 4: Add to replay buffer with phase-aware tags
        for ex in executed_trades:
            replay_buffer.add(
                ex,
                tags={
                    "phase": phase_controller.phase,
                    "source": phase_controller.get_source_label(),
                    "exploration_type": ex["meta"].get("exploration_type", "random")
                }
            )

        logger.info(f"🧠 Replay buffer size: {replay_buffer.size()} after {date.date()}")

        # Step 5: Weekly model retraining
        if date.weekday() == 4:  # Friday
            logger.info(f"📚 Retraining models at week ending {date.date()}")
            train_models(replay_buffer, up_to_date=date)

    logger.info("✅ Historical bootstrap completed.")
