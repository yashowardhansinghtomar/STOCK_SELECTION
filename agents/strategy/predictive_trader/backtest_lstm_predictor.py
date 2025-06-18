import os
import pandas as pd
import numpy as np
from datetime import timedelta

from core.data_provider.data_provider import load_data
from core.logger.logger import logger
from config.paths import PATHS
from predictive_trader.price_predictor_lstm_v2 import (
    train_lstm_model_v2,
    predict_5day_return_v2,
    FEATURE_WINDOW,
    FUTURE_OFFSET
)

TICKER = "RELIANCE"

def backtest_lstm_predictor(ticker=TICKER, start_date="2024-01-01", end_date="2024-04-01"):
    logger.info("\n🚀 Starting LSTM backtest...")
    df = load_data("stock_features")
    df = df[df["stock"] == ticker].sort_values("date").reset_index(drop=True)

    if df.empty:
        logger.error(f"❌ No data in stock_features for {ticker}")
        return

    backtest_dates = pd.date_range(start=start_date, end=end_date, freq="10D")
    results = []

    for test_date in backtest_dates:
        logger.info(f"\n📅 Backtesting for {ticker} on {test_date.date()}...")

        success = train_lstm_model_v2(ticker, simulation_date=test_date)
        if not success:
            continue

        pred = predict_5day_return_v2(ticker, simulation_date=test_date)
        if pred is None:
            continue

        df_future = df[df["date"] > test_date]
        if len(df_future) < FUTURE_OFFSET:
            logger.warning(f"⚠️ Not enough future data to calculate return on {test_date.date()}")
            continue

        price_now = df[df["date"] == test_date]["vwap_dev"].values
        price_later = df_future.iloc[FUTURE_OFFSET - 1]["vwap_dev"]

        if len(price_now) == 0:
            continue

        actual_return = price_later - price_now[0]
        error = pred - actual_return

        results.append({
            "date": test_date.date(),
            "predicted_return": round(pred, 4),
            "actual_return": round(actual_return, 4),
            "error": round(error, 4)
        })

    result_df = pd.DataFrame(results)
    logger.success("✅ Backtest Complete!")
    print(result_df)
    return result_df

if __name__ == "__main__":
    backtest_lstm_predictor()
