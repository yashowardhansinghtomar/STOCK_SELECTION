import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Input

from core.data_provider.data_provider import load_data
from core.logger.logger import logger
from config.paths import PATHS

MODEL_DIR = os.path.join(PATHS.get("model_dir", "models"), "predictive_trader")
os.makedirs(MODEL_DIR, exist_ok=True)

FUTURE_DAYS = 5  # Predict 5 days

# --- Build a fresh LSTM model (for retraining) ---
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
def load_price_data(ticker, num_days=500):
    df = load_data('stock_price_history')
    if df is None or df.empty:
        logger.error("❌ stock_price_history table is empty or missing.")
        return None
    df = df[df['symbol'] == ticker].sort_values('date')
    return df.tail(num_days)

# --- Train LSTM up to a given simulation date ---
def train_model_upto(ticker, sim_date, epochs=30):
    logger.info(f"🛠 Training model for {ticker} up to {sim_date}...")

    df = load_price_data(ticker, 500)
    if df is None or df.empty:
        logger.error(f"⚠️ No data to train for {ticker}")
        return None

    df = df[df['date'] < sim_date]  # ✅ Only past data
    if len(df) < 100:
        logger.warning(f"⚠️ Not enough past data for {ticker} (found {len(df)})")
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

    # Save model
    model_path = os.path.join(MODEL_DIR, f"{ticker}_{sim_date}_lstm.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_{sim_date}_scaler.pkl")

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    logger.success(f"✅ Model trained and saved: {model_path}")
    return model_path

# --- Load model corresponding to simulation date ---
def load_model_for_date(ticker, sim_date):
    model_path = os.path.join(MODEL_DIR, f"{ticker}_{sim_date}_lstm.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_{sim_date}_scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        logger.warning(f"❗ Model for {ticker} at {sim_date} not found, training now...")
        train_model_upto(ticker, sim_date)

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler
