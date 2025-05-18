# scripts/seed_training_data.py
import pandas as pd
from core.data_provider import load_data, save_data
from core.logger import logger

def seed_training_data():
    features = load_data("stock_features")
    backtests = load_data("paper_trades")  # or "backtest_results" if exists

    if features.empty or backtests.empty:
        logger.error("Missing data to generate training set.")
        return

    df = features.merge(backtests, on=["stock", "date"], how="inner")

    df["target"] = (df["total_return"] > 5).astype(int)
    keep_cols = [*settings.indicator_columns, "stock", "date", "target"]
    df = df[keep_cols].dropna()

    save_data(df, "training_data", if_exists="replace")
    logger.success(f"✅ Seeded training_data with {len(df)} rows")

if __name__ == "__main__":
    seed_training_data()
