# inspect_feature_dates.py

import pandas as pd
from core.data_provider.data_provider import load_data
from core.logger.logger import logger

df = load_data("stock_features_day")
df = df[df["stock"].isin(["RELIANCE", "SBIN", "INFY", "LT", "ICICIBANK"])]
df["date"] = pd.to_datetime(df["date"]).dt.date

for stock in df["stock"].unique():
    sub = df[df["stock"] == stock].sort_values("date")
    logger.info(f"📅 {stock} earliest: {sub['date'].min()} | latest: {sub['date'].max()}")
    logger.info(f"🔍 {stock} sample dates: {sorted(sub['date'].unique()[:5])}")
