# predictive_trader/price_predictor_lstm_intraday.py

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Input
from datetime import time

from core.data_provider.data_provider import fetch_stock_data
from core.logger.logger import logger
from config.paths import PATHS

# --- Config ---
TARGET_STOCK = "RELIANCE"
MODEL_SUFFIX = "intraday_5min"
MODEL_DIR = os.path.join(PATHS.get("model_dir", "models"), "predictive_trader")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_WINDOW = 24      # 2 hours of 5-min candles
FUTURE_OFFSET = 6        # predict 30-min ahead return

FEATURE_COLUMNS = [
    'sma_5', 'sma_20', 'rsi_14', 'volume_spike', 'atr_14',
    'macd_histogram', 'bb_width', 'vwap_dev', 'price_compression'
]

# --- Model Builder ---
def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(100, return_sequences=True),
        LSTM(100),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# --- Training ---
def train_intraday_model(ticker, epochs=20):
    df = fetch_stock_data(ticker, interval="5minute", days=10)
    if df is None or df.empty:
        logger.error("❌ No intraday data")
        return False

    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.time.between(time(9,15), time(15,30))]
    df = df.sort_values("date").reset_index(drop=True)

    # Assume intraday features already added externally
    if any(col not in df.columns for col in FEATURE_COLUMNS):
        logger.error("❌ Missing feature columns in intraday data")
        return False

    features = df[FEATURE_COLUMNS]
    target = df["Close"].shift(-FUTURE_OFFSET) / df["Close"] - 1

    features = features.iloc[:-FUTURE_OFFSET].reset_index(drop=True)
    target = target.dropna().reset_index(drop=True)
    min_len = min(len(features), len(target))
    features = features.iloc[:min_len]
    target = target.iloc[:min_len]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features)

    X_seq, y_seq = [], []
    for i in range(FEATURE_WINDOW, len(X_scaled)):
        X_seq.append(X_scaled[i - FEATURE_WINDOW:i])
        y_seq.append(target.iloc[i])

    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    print(f"✅ Final X: {X_seq.shape}, y: {y_seq.shape}")

    model = build_model((FEATURE_WINDOW, len(FEATURE_COLUMNS)))
    model.fit(X_seq, y_seq, epochs=epochs, batch_size=32, verbose=0)

    model.save(os.path.join(MODEL_DIR, f"{ticker}_{MODEL_SUFFIX}.keras"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"{ticker}_{MODEL_SUFFIX}_scaler.pkl"))
    logger.success(f"✅ Intraday model trained and saved for {ticker}")
    return True

# --- Prediction ---
def predict_intraday_return(ticker):
    from tensorflow.keras.models import load_model

    model_path = os.path.join(MODEL_DIR, f"{ticker}_{MODEL_SUFFIX}.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_{MODEL_SUFFIX}_scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        logger.warnings("❗ Model not found, training now...")
        trained = train_intraday_model(ticker)
        if not trained:
            return None

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    df = fetch_stock_data(ticker, interval="5minute", days=2)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.time.between(time(9,15), time(15,30))]
    df = df.sort_values("date")

    if len(df) < FEATURE_WINDOW:
        logger.error("⚠️ Not enough intraday data for prediction")
        return None

    latest_features = df[FEATURE_COLUMNS].iloc[-FEATURE_WINDOW:]

    if latest_features.isna().any().any():
        logger.error("❌ NaNs in latest features")
        return None

    X = scaler.transform(latest_features)
    X = np.expand_dims(X, axis=0)

    pred_return = model.predict(X)[0][0]
    return pred_return

# --- Entry Point ---
if __name__ == "__main__":
    trained = train_intraday_model(TARGET_STOCK)
    if trained:
        pred = predict_intraday_return(TARGET_STOCK)
        if pred is not None:
            print(f"📈 Predicted 30-min return for {TARGET_STOCK}: {pred:.4f}")
        else:
            print("❌ Prediction failed.")
    else:
        print("❌ Training failed.")
