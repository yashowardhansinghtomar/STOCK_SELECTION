import numpy as np
import pandas as pd
import os
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Input
from core.data_provider.data_provider import load_data
from core.logger.logger import logger
from core.time_context.time_context import get_simulation_date
from db.db_router import insert_dataframe
from config.paths import PATHS
from datetime import datetime
from predictive_trader.model_manager import load_model_for_date  # ✅ Missing import added!

# --- Configuration ---
print(tf.config.list_physical_devices('GPU'))  # Debugging

MODEL_DIR = os.path.join(PATHS.get("model_dir", "models"), "predictive_trader")
os.makedirs(MODEL_DIR, exist_ok=True)

FUTURE_DAYS = 5

# --- Build fresh LSTM model (only used internally for training) ---
def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(FUTURE_DAYS)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# --- Load historical price data ---
def load_price_data(ticker, num_days, simulation_date=None):
    df = load_data('stock_price_history')
    if df is None or df.empty:
        logger.error("❌ stock_price_history table is empty or missing.")
        return None
    df = df[df['symbol'] == ticker].sort_values('date')
    if simulation_date:
        df = df[df['date'] <= simulation_date]  # ✅ Only past data up to simulation
    if df.empty:
        logger.error(f"❌ No price history found for {ticker} (up to {simulation_date}).")
        return None
    return df.tail(num_days)

# --- Train and Save Model for today ---
def train_lstm_model(ticker, epochs=30, simulation_date=None):
    df = load_price_data(ticker, 365, simulation_date=simulation_date)
    if df is None or df.empty:
        logger.error(f"⚠️ No training data for {ticker}")
        return None

    df['close'] = df['close'].ffill()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df['close'].values.reshape(-1, 1))

    X, y = [], []
    for i in range(60, len(scaled) - FUTURE_DAYS):
        X.append(scaled[i-60:i, 0])
        y.append(scaled[i:i+FUTURE_DAYS, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = build_lstm_model((X.shape[1], 1))
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

    model_path = os.path.join(MODEL_DIR, f"{ticker}_lstm.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    logger.success(f"✅✅ 5-day LSTM model trained and saved for {ticker}")

# --- Predict next 5 days (simulation safe) ---
def predict_next_5days_lstm(ticker, simulation_date=None):
    sim_date = simulation_date if simulation_date else get_simulation_date()
    model, scaler = load_model_for_date(ticker, sim_date)

    df = load_price_data(ticker, 70, simulation_date=sim_date)
    if df is None or df.empty:
        return None

    df = df[df['date'] < sim_date]
    close_prices = df['close'].ffill().values[-60:]
    scaled = scaler.transform(close_prices.reshape(-1, 1))

    X = np.reshape(scaled, (1, 60, 1))
    pred_scaled = model.predict(X)
    pred_prices = scaler.inverse_transform(pred_scaled)[0]

    return {f"day{i+1}": float(price) for i, price in enumerate(pred_prices)}

# --- Save predictions into SQL ---
def save_5day_predictions(ticker, preds, simulation_date=None):
    timestamp = simulation_date if simulation_date else datetime.now().strftime("%Y-%m-%d %H:%M")
    df = pd.DataFrame([{
        "timestamp": timestamp,
        "symbol": ticker,
        "day_1": preds["day1"],
        "day_2": preds["day2"],
        "day_3": preds["day3"],
        "day_4": preds["day4"],
        "day_5": preds["day5"],
        "model_used": "lstm_5day_pred",
        "imported_at": datetime.now().strftime("%Y-%m-%d %H:%M")
    }])
    insert_dataframe(df, "predicted_curves", if_exists="append")
    logger.info(f"📈 Saved 5-day curve prediction for {ticker} to predicted_curves (simulation safe)")

# --- Manual Testing ---
if __name__ == "__main__":
    ticker = "RELIANCE"
    train_lstm_model(ticker)
    preds = predict_next_5days_lstm(ticker)
    if preds:
        print(preds)
        save_5day_predictions(ticker, preds)
