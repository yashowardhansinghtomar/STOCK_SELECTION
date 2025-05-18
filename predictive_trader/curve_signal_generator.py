# predictive_trader/curve_signal_generator.py

import pandas as pd
from core.data_provider import load_data
from db.db_router import insert_dataframe, run_query
from core.time_context import get_simulation_date
from core.logger import logger

# --- Configuration ---
CURVE_TABLE = "predicted_curves"
SIGNAL_TABLE = "predicted_signals"

BUY_THRESHOLD = 2.0    # +2% rise → Buy
SELL_THRESHOLD = -2.0  # -2% fall → Sell

def generate_signals_from_curves():
    sim_date = get_simulation_date()
    logger.info(f"🔎 Analyzing curves for {sim_date}...")

    df = load_data(CURVE_TABLE)
    if df is None or df.empty:
        logger.warning("⚠️ No curve predictions available.")
        return

    df_today = df[df['timestamp'] == sim_date]
    if df_today.empty:
        logger.warning(f"⚠️ No curve predictions for {sim_date}.")
        return

    signals = []

    for _, row in df_today.iterrows():
        symbol = row["symbol"]
        day1 = row.get("day_1")
        day5 = row.get("day_5")

        if pd.isna(day1) or pd.isna(day5):
            continue

        pct_change = (day5 - day1) / day1 * 100

        if pct_change > BUY_THRESHOLD:
            signal = "BUY"
        elif pct_change < SELL_THRESHOLD:
            signal = "SELL"
        else:
            signal = "HOLD"

        signals.append({
            "timestamp": sim_date,
            "symbol": symbol,
            "current_close": day1,      # 🔵 Today's price estimate
            "predicted_close": day5,    # 🔵 5th day predicted price
            "predicted_change_percent": pct_change,
            "signal": signal,
            "source": "curve_predictor",
            "imported_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        })

    if not signals:
        logger.warning("⚠️ No actionable signals generated.")
        return

    df_signals = pd.DataFrame(signals)

    # 🧹 Clear any old signals for today
    try:
        run_query(f"DELETE FROM {SIGNAL_TABLE} WHERE timestamp = '{sim_date}'")
    except Exception as e:
        logger.warning(f"⚠️ Could not delete old signals: {e}")

    insert_dataframe(df_signals, SIGNAL_TABLE, if_exists="append")
    logger.success(f"✅ {len(df_signals)} new signals saved to {SIGNAL_TABLE}.")

if __name__ == "__main__":
    generate_signals_from_curves()
