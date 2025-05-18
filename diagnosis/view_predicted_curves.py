# diagnosis/evaluate_model_curves.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from core.data_provider import load_data
from core.logger import logger

# --- Config ---
CURVE_TABLE = "predicted_curves"
PRICE_TABLE = "stock_price_history"
STOCK = "RELIANCE"  # 🔥 change here if you want to evaluate another

# --- Load predicted curves ---
def load_predictions(stock):
    df = load_data(CURVE_TABLE)
    if df is None or df.empty:
        logger.error("❌ No predicted curves found.")
        return None
    df = df[df['symbol'] == stock]
    return df

# --- Load actual prices ---
def load_actual_prices(stock):
    df = load_data(PRICE_TABLE)
    if df is None or df.empty:
        logger.error("❌ No stock price history found.")
        return None
    df = df[df['symbol'] == stock]
    return df

# --- Evaluation Logic ---
def evaluate_model(stock=STOCK):
    pred_df = load_predictions(stock)
    actual_df = load_actual_prices(stock)

    if pred_df is None or actual_df is None:
        return

    results = []

    # Ensure date types match
    if actual_df['date'].dt.tz is not None:
        pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp']).dt.tz_localize('UTC')
    else:
        pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])

    for _, row in pred_df.iterrows():
        date = row['timestamp']
        pred_day5_price = row['day_5']

        # find actual close price after 5 trading days
        future_prices = actual_df[actual_df['date'] > date].sort_values('date')
        if len(future_prices) < 5:
            continue

        actual_day5_price = future_prices.iloc[4]['close']

        pct_error = (pred_day5_price - actual_day5_price) / actual_day5_price * 100
        correct_direction = np.sign(pred_day5_price - row['day_1']) == np.sign(actual_day5_price - row['day_1'])

        results.append({
            'pred_date': date.strftime('%Y-%m-%d'),
            'pred_day5': pred_day5_price,
            'actual_day5': actual_day5_price,
            'pct_error': pct_error,
            'correct_direction': correct_direction
        })

    if not results:
        logger.warning("⚠️ No evaluatable predictions.")
        return

    result_df = pd.DataFrame(results)

    # --- Metrics ---
    mae = mean_absolute_error(result_df['actual_day5'], result_df['pred_day5'])

    try:
        rmse = mean_squared_error(result_df['actual_day5'], result_df['pred_day5'], squared=False)
    except TypeError:
        mse = mean_squared_error(result_df['actual_day5'], result_df['pred_day5'])
        rmse = np.sqrt(mse)

    direction_acc = result_df['correct_direction'].mean() * 100
    avg_predicted_return = ((result_df['pred_day5'] - result_df['pred_day5'].shift(1)) / result_df['pred_day5'].shift(1)).mean() * 100
    avg_actual_return = ((result_df['actual_day5'] - result_df['actual_day5'].shift(1)) / result_df['actual_day5'].shift(1)).mean() * 100
    bias = avg_predicted_return - avg_actual_return

    logger.info(f"\n📊 Evaluation Results for {stock}:")
    logger.info(f"MAE: {mae:.2f}")
    logger.info(f"RMSE: {rmse:.2f}")
    logger.info(f"Directional Accuracy: {direction_acc:.2f}%")
    logger.info(f"Avg Predicted 5-day Return: {avg_predicted_return:.2f}%")
    logger.info(f"Avg Actual 5-day Return: {avg_actual_return:.2f}%")
    logger.info(f"Bias (Predicted - Actual): {bias:.2f}%")

    # --- Plots ---
    plt.figure(figsize=(12, 5))
    plt.plot(result_df['pred_date'], result_df['actual_day5'], label='Actual Day5 Price', marker='o')
    plt.plot(result_df['pred_date'], result_df['pred_day5'], label='Predicted Day5 Price', marker='x')
    plt.xticks(rotation=45)
    plt.title(f"{stock} - Predicted vs Actual Day5 Prices")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(result_df['pred_date'], result_df['pct_error'], label='Prediction Error (%)', color='red')
    plt.axhline(0, color='black', linestyle='--')
    plt.xticks(rotation=45)
    plt.title(f"{stock} - Prediction Error Over Time")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_model()
