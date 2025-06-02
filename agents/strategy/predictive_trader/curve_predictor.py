# predictive_trader/curve_predictor.py

import os
import pandas as pd

from predictive_trader.price_predictor_lstm import train_lstm_model, predict_next_5days_lstm
from core.data_provider.data_provider import load_data
from db.postgres_manager import insert_dataframe, run_query
from core.logger.logger import logger
from core.time_context.time_context import get_simulation_date
from config.paths import PATHS

# 🛠 Setup
MODEL_DIR = os.path.join(PATHS.get("model_dir", "models"), "predictive_trader")
os.makedirs(MODEL_DIR, exist_ok=True)

# 🟰 Stocks to predict curves for
STOCK_LIST = ["RELIANCE", "TCS", "INFY", "ICICIBANK", "HDFCBANK"]

# 🟰 SQL table where curves are saved
CURVE_TABLE = "predicted_curves"

def generate_curves_for_list(ticker_list=STOCK_LIST):
    curves = []

    sim_date = get_simulation_date()
    logger.info(f"📈 Generating curves for {sim_date}...")

    for ticker in ticker_list:
        # 🛡️ Check if model exists for this date
        model_path = os.path.join(MODEL_DIR, f"{ticker}_{sim_date}_lstm.keras")
        scaler_path = os.path.join(MODEL_DIR, f"{ticker}_{sim_date}_scaler.pkl")

        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            logger.info(f"🛠 Training model for {ticker} (safe till {sim_date})...")
            train_lstm_model(ticker, epochs=10, simulation_date=sim_date)
        else:
            logger.info(f"✅ Model already exists for {ticker} (safe till {sim_date}), skipping retrain.")

        # 🔮 Predict using trained model
        curve = predict_next_5days_lstm(ticker, simulation_date=sim_date)
        if curve is None:
            logger.warnings(f"⚠️ Skipping {ticker} - no prediction available.")
            continue

        curves.append({
            "timestamp": sim_date,
            "symbol": ticker,
            "day1": curve.get("day1"),
            "day2": curve.get("day2"),
            "day3": curve.get("day3"),
            "day4": curve.get("day4"),
            "day5": curve.get("day5"),
            "imported_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        })

    if not curves:
        logger.warnings("⚠️ No curves generated.")
        return

    df = pd.DataFrame(curves)

    # 🛠 Rename columns to match database
    df = df.rename(columns={
        "day1": "day_1",
        "day2": "day_2",
        "day3": "day_3",
        "day4": "day_4",
        "day5": "day_5"
    })

    # 🧹 Remove old curves for this simulation date
    try:
        run_query(f"DELETE FROM {CURVE_TABLE} WHERE timestamp = '{sim_date}'")
    except Exception as e:
        logger.warnings(f"⚠️ Could not delete old curves: {e}")

    # 💾 Insert new curves
    insert_dataframe(df, CURVE_TABLE, if_exists="append")
    logger.success(f"✅ {len(curves)} curves saved to {CURVE_TABLE}.")

if __name__ == "__main__":
    generate_curves_for_list()
