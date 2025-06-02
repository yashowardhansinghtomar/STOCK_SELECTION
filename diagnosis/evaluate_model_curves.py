# diagnosis/evaluate_model_curves.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from core.data_provider.data_provider import load_data
from core.logger.logger import logger

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
        pred_return5 = (row['day_5'] - row['day_1']) / row['day_1'] * 100

        # find actual close price after 5 trading days
        future_prices = actual_df[actual_df['date'] > date].sort_values('date')
        if len(future_prices) < 5:
            continue

        price_now = actual_df[actual_df['date'] <= date].sort_values('date').iloc[-1]['close']
        price_future = future_prices.iloc[4]['close']

        actual_return5 = (price_future - price_now) / price_now * 100

        pct_error = (pred_return5 - actual_return5)
        correct_direction = np.sign(pred_return5) == np.sign(actual_return5)

        results.append({
            'pred_date': date.strftime('%Y-%m-%d'),
            'pred_return5': pred_return5,
            'actual_return5': actual_return5,
            'pct_error': pct_error,
            'correct_direction': correct_direction
        })

    if not results:
        logger.warnings("⚠️ No evaluatable predictions.")
        return

    result_df = pd.DataFrame(results)

    # --- Metrics ---
    mae = mean_absolute_error(result_df['actual_return5'], result_df['pred_return5'])
    mse = mean_squared_error(result_df['actual_return5'], result_df['pred_return5'])
    rmse = np.sqrt(mse)
    direction_acc = result_df['correct_direction'].mean() * 100
    avg_pred_return = result_df['pred_return5'].mean()
    avg_actual_return = result_df['actual_return5'].mean()
    bias = avg_pred_return - avg_actual_return

    logger.info(f"\n📊 Evaluation Results for {stock}:")
    logger.info(f"MAE: {mae:.2f}%")
    logger.info(f"RMSE: {rmse:.2f}%")
    logger.info(f"Directional Accuracy: {direction_acc:.2f}%")
    logger.info(f"Avg Predicted 5-day Return: {avg_pred_return:.2f}%")
    logger.info(f"Avg Actual 5-day Return: {avg_actual_return:.2f}%")
    logger.info(f"Bias (Predicted - Actual): {bias:.2f}%")

    # --- Plots ---
    plt.figure(figsize=(12, 5))
    plt.plot(result_df['pred_date'], result_df['actual_return5'], label='Actual 5d Return', marker='o')
    plt.plot(result_df['pred_date'], result_df['pred_return5'], label='Predicted 5d Return', marker='x')
    plt.xticks(rotation=45)
    plt.title(f"{stock} - Predicted vs Actual 5-Day Returns")
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
