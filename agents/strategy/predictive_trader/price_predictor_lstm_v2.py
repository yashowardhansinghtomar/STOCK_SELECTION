# predictive_trader/price_predictor_lstm_v2.py

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Input

from core.data_provider.data_provider import load_data
from core.logger.logger import logger
from config.paths import PATHS

# --- Config ---
TARGET_STOCK = "RELIANCE"
MODEL_DIR = os.path.join(PATHS.get("model_dir", "models"), "predictive_trader")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_WINDOW = 30
FUTURE_OFFSET = 5

# Define the features to use (must match enrich_stock_price_history + DB schema)
FEATURE_COLUMNS = [
    'sma_10', 'sma_30', 'rsi_14', 'volume_spike', 'volatility_10',
    'atr_14', 'macd_histogram', 'bb_width', 'vwap_dev', 'price_compression'
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
def train_lstm_model_v2(ticker, epochs=50, simulation_date=None):
    df = load_data("stock_features")
    if df is None or df.empty:
        logger.error("❌ stock_features missing!")
        return False

    df = df[df["stock"] == ticker]
    if simulation_date:
        df = df[df["date"] < simulation_date]
    df = df.sort_values("date")

    print(f"📊 Loaded {len(df)} feature rows for {ticker}")
    if len(df) < FEATURE_WINDOW + FUTURE_OFFSET:
        logger.error(f"⚠️ Not enough data for {ticker}")
        return False

    # Input and target
    features = df[FEATURE_COLUMNS]
    target = df["vwap_dev"].shift(-FUTURE_OFFSET)

    # Trim the last few rows of features and target to be the same length
    features = features.iloc[:-FUTURE_OFFSET].reset_index(drop=True)
    target = target.dropna().reset_index(drop=True)

    # Ensure alignment
    min_len = min(len(features), len(target))
    features = features.iloc[:min_len]
    target = target.iloc[:min_len]

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features)

    # Create windowed sequences
    X_seq = []
    y_seq = []
    for i in range(FEATURE_WINDOW, len(X_scaled)):
        X_seq.append(X_scaled[i - FEATURE_WINDOW:i])
        y_seq.append(target.iloc[i])

    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    print(f"✅ Final X: {X_seq.shape}, y: {y_seq.shape}")

    model = build_model((FEATURE_WINDOW, len(FEATURE_COLUMNS)))
    model.fit(X_seq, y_seq, epochs=epochs, batch_size=32, verbose=0)

    model.save(os.path.join(MODEL_DIR, f"{ticker}_v2_lstm.keras"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"{ticker}_v2_scaler.pkl"))
    logger.success(f"✅ Model v2 trained and saved for {ticker}")
    return True

# --- Prediction ---
def predict_5day_return_v2(ticker, simulation_date=None):
    from tensorflow.keras.models import load_model

    model_path = os.path.join(MODEL_DIR, f"{ticker}_v2_lstm.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_v2_scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        logger.warnings(f"❗ Model not found for {ticker}, training now...")
        trained = train_lstm_model_v2(ticker, simulation_date)
        if not trained:
            return None

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    df = load_data("stock_features")
    df = df[df["stock"] == ticker]
    if simulation_date:
        df = df[df["date"] <= simulation_date]
    df = df.sort_values("date")

    if len(df) < FEATURE_WINDOW:
        logger.error(f"⚠️ Not enough data to predict {ticker}")
        return None

    latest_features = df[FEATURE_COLUMNS].iloc[-FEATURE_WINDOW:]

    if latest_features.isna().any().any():
        logger.error(f"❌ Cannot predict: NaNs found in latest features for {ticker}")
        return None

    X = scaler.transform(latest_features)
    X = np.expand_dims(X, axis=0)

    pred_return = model.predict(X)[0][0]
    return pred_return

# --- Entry point ---
if __name__ == "__main__":
    trained = train_lstm_model_v2(TARGET_STOCK)
    if trained:
        ret = predict_5day_return_v2(TARGET_STOCK)
        if ret is not None:
            print(f"Predicted 5-day return for {TARGET_STOCK}: {ret:.2f}%")
        else:
            print(f"❌ Prediction failed for {TARGET_STOCK}")
    else:
        print(f"❌ Model not trained. Prediction skipped.")
