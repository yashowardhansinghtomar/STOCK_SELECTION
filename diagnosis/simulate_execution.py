from datetime import datetime, timedelta
from agents.execution.execution_agent_sql import ExecutionAgentSQL
from predictive_trader.curve_predictor import generate_curves_for_list
from predictive_trader.curve_signal_generator import generate_signals_from_curves
from core.time_context.time_context import set_simulation_date, clear_simulation_date
from core.logger.logger import logger

STOCK_LIST = ["RELIANCE", "TCS", "INFY", "ICICIBANK", "HDFCBANK"]
SIMULATE_NUM_DAYS = 20
START_DATE_STR = "2025-04-01"

def generate_trading_days(start_date, num_days=20):
    trading_days = []
    date = start_date
    while len(trading_days) < num_days:
        if date.weekday() < 5:  # Monday-Friday
            trading_days.append(date)
        date += timedelta(days=1)
    return trading_days

def simulate_trading_days(start_date_str=START_DATE_STR, num_days=SIMULATE_NUM_DAYS):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    trading_days = generate_trading_days(start_date, num_days)

    for idx, date in enumerate(trading_days):
        sim_date_str = date.strftime("%Y-%m-%d")
        print(f"\n📆 [DAY {idx+1}/{len(trading_days)}] Simulating {sim_date_str}")
        set_simulation_date(sim_date_str)

        # Step 1: Predict future curve
        logger.info(f"📈 Predicting curve for {sim_date_str}...")
        generate_curves_for_list(STOCK_LIST)   # ✅ Corrected here!

        # Step 2: Generate signals based on curve
        logger.info(f"🧠 Generating signals from curve for {sim_date_str}...")
        generate_signals_from_curves()

        # Step 3: Execute trades
        logger.info(f"🚀 Executing trades for {sim_date_str}...")
        ExecutionAgentSQL(dry_run=False).run()

    clear_simulation_date()
    logger.success("✅ Full curve-based simulation completed!")

if __name__ == "__main__":
    simulate_trading_days()
