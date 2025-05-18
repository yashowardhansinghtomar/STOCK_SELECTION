# predictive_trader/price_predictor_lgbm.py

import numpy as np
import pandas as pd
import lightgbm as lgb
import os
import joblib

from core.data_provider import load_data
from core.logger import logger
from config.paths import PATHS

MODEL_DIR = os.path.join(PATHS.get("model_dir", "models"), "predictive_trader")
os.makedirs(MODEL_DIR, exist_ok=True)

def generate_features(df):
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_15'] = df['close'].rolling(window=15).mean()
    df['rsi_14'] = compute_rsi(df['close'], 14)
    df['volatility'] = df['close'].pct_change().rolling(window=14).std()
    df = df.dropna()
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def load_price_data(ticker, num_days):
    df = load_data('stock_price_history')
    if df is None or df.empty:
        logger.error("❌ stock_price_history table is empty or missing.")
        return None
    df = df[df['symbol'] == ticker].sort_values('date')
    if df.empty:
        logger.error(f"❌ No price history found for {ticker}.")
        return None
    return df.tail(num_days)

def train_lgbm_model(ticker, num_days=400):
    df = load_price_data(ticker, num_days)
    if df is None or df.empty:
        logger.error(f"⚠️ No training data for {ticker}")
        return None

    df = generate_features(df)

    X = df[['sma_5', 'sma_15', 'rsi_14', 'volatility']]
    y = np.sign(df['close'].shift(-1) - df['close'])  # 1 for up, -1 for down
    y = y.dropna()
    X = X.iloc[:-1]

    model = lgb.LGBMClassifier()
    model.fit(X, y)

    model_path = os.path.join(MODEL_DIR, f"{ticker}_lgbm.pkl")
    joblib.dump(model, model_path)
    logger.success(f"✅ LightGBM model trained and saved for {ticker}")

def predict_movement_lgbm(ticker):
    try:
        model_path = os.path.join(MODEL_DIR, f"{ticker}_lgbm.pkl")
        model = joblib.load(model_path)
    except Exception as e:
        logger.error(f"❌ LightGBM model not found for {ticker}: {e}")
        return None

    df = load_price_data(ticker, 30)
    if df is None or df.empty:
        return None

    df = generate_features(df)
    if df.empty:
        return None

    X = df[['sma_5', 'sma_15', 'rsi_14', 'volatility']].iloc[-1:]

    pred = model.predict(X)[0]
    return pred  # 1 = UP, -1 = DOWN

if __name__ == "__main__":
    ticker = "RELIANCE"
    train_lgbm_model(ticker)
    print("LGBM Movement Prediction:", predict_movement_lgbm(ticker))
