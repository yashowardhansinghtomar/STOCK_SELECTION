# bootstrap/historical_bootstrap_runner.py

from datetime import datetime, timedelta
import random
from bootstrap.replay_buffer import ReplayBuffer
from core.filtering.run_filter import run_filter_model
from bootstrap.simulate_trade_execution import simulate_trade_execution
from core.model_trainer.trainer import train_models
from core.market_calendar import get_trading_days
from bootstrap.phase_controller import PhaseController
from core.logger.logger import logger


def run_historical_bootstrap(start_date: str, end_date: str):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    trading_days = get_trading_days(start, end)

    replay_buffer = ReplayBuffer()
    phase_controller = PhaseController(initial_phase=0)

    logger.info(f"🚀 Starting historical bootstrap from {start_date} to {end_date}...")

    for date in trading_days:
        logger.info(f"📅 Simulating for {date.date()} | Phase {phase_controller.phase}")

        # Phase check and auto-transition logic
        phase_controller.update_phase(replay_buffer)

        # Step 1: Get filtered stocks for that day using lookback-only features
        filtered_stocks = run_filter_model(date, lookback_only=True)

        # Step 2: Generate trades (random param config for early phases)
        trades = phase_controller.generate_trades(filtered_stocks, date)

        # Step 3: Simulate execution using minute bars and realism
        executed_trades = []
        for trade in trades:
            result = simulate_trade_execution(trade, date)
            if result:
                executed_trades.append(result)

        # Step 4: Add to replay buffer with phase-aware tags
        for ex in executed_trades:
            replay_buffer.add(
                ex,
                tags={
                    "phase": phase_controller.phase,
                    "source": phase_controller.get_source_label(),
                    "exploration_type": trade.meta.get("exploration_type", "random")
                }
            )

        # Step 5: Weekly model retraining
        if date.weekday() == 4:  # Friday
            train_models(replay_buffer, up_to_date=date)

    logger.info("✅ Historical bootstrap completed.")
