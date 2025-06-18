# predictive_trader/trade_signal_generator.py

import pandas as pd
from datetime import datetime
import os

from predictive_trader.price_predictor_lstm import predict_next_close_lstm, predict_next_n_days_lstm
from predictive_trader.price_predictor_lgbm import predict_movement_lgbm
from core.data_provider.data_provider import fetch_stock_data
from db.db_router import insert_dataframe
from core.logger.logger import logger

# --- Signal thresholds
BUY_THRESHOLD = 1.02   # +2%
SELL_THRESHOLD = 0.98  # -2%

# --- Signal Generator per stock
def generate_trade_signal(ticker):
    try:
        # Current close price
        df = fetch_stock_data(ticker, return_last_n_days=5)
        if df is None or df.empty:
            logger.error(f"❌ Failed to fetch last close price for {ticker}")
            return None
        current_close = df['close'].iloc[-1]

        # Predict next 5 closes
        future_preds = predict_next_n_days_lstm(ticker, n_days=5)
        if future_preds is None or len(future_preds) < 5:
            logger.error(f"❌ LSTM prediction failed for {ticker}")
            return None

        fifth_day_price = future_preds[-1]
        
        # Movement prediction (optional second layer)
        movement = predict_movement_lgbm(ticker)

        # Decide signal
        if fifth_day_price >= current_close * BUY_THRESHOLD and movement == 1:
            signal = "BUY"
        elif fifth_day_price <= current_close * SELL_THRESHOLD and movement == -1:
            signal = "SELL"
        else:
            signal = "HOLD"

        logger.info(f"🧠 {ticker} Signal → {signal}")

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "symbol": ticker,
            "current_close": current_close,
            "predicted_day_5": fifth_day_price,
            "lgbm_movement": movement,
            "signal": signal,
            "future_curve": str(future_preds)  # Optional: save whole curve as string
        }

    except Exception as e:
        logger.error(f"❌ Error generating signal for {ticker}: {e}")
        return None

# --- Bulk Signal Generator
def generate_signals_for_list(ticker_list):
    signals = []
    for ticker in ticker_list:
        result = generate_trade_signal(ticker)
        if result:
            signals.append(result)

    if signals:
        df = pd.DataFrame(signals)
        insert_dataframe(df, "predicted_signals", if_exists="append")
        logger.success(f"✅ {len(signals)} signals saved to predicted_signals.")
    else:
        logger.warning("⚠️ No signals generated.")

if __name__ == "__main__":
    generate_signals_for_list(["RELIANCE", "TCS", "INFY"])
