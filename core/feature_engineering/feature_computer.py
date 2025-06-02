# core/feature_engineering/feature_computer.py

import pandas as pd
from core.feature_engineering.precompute_features import compute_features
from core.data_provider.data_provider import fetch_stock_data
from core.logger.logger import logger

def compute_and_prepare_features(stock: str, interval: str, date: str = None) -> pd.DataFrame:
    try:
        df_price = fetch_stock_data(stock, interval=interval)
        if df_price is None or df_price.empty:
            return pd.DataFrame()

        df_price["stock"] = stock
        df_price["stock_encoded"] = hash(stock) % 10000
        df_feat = compute_features(df_price)
        if df_feat.empty:
            return pd.DataFrame()
        if "date" not in df_feat.columns:
            df_feat = df_feat.reset_index()
        df_feat["date"] = pd.to_datetime(df_feat["date"]).dt.date
        return df_feat
    except Exception as e:
        logger.warning(f"Failed to compute features for {stock}: {e}")
        return pd.DataFrame()
