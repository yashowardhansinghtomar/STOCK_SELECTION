# core/data_cleaner.py
import pandas as pd
from core.logger import logger

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Date": "date",
        "Stock": "stock",
    }
    df = df.rename(columns=rename_map)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df

def sanity_check_features(df: pd.DataFrame) -> pd.DataFrame:
    critical_columns = [
        "date", "stock", "sma_short", "sma_long", "rsi_thresh", "stock_encoded",
        "proxy_pe", "proxy_roe", "proxy_market_cap", "proxy_de_ratio", "proxy_growth"
    ]
    missing = set(critical_columns) - set(df.columns)
    if missing:
        raise ValueError(f"❌ Missing critical columns: {missing}")

    df_before = len(df)
    df = df.dropna(subset=critical_columns)
    df_after_nan_drop = len(df)
    if df_before != df_after_nan_drop:
        logger.warning(f"⚠️ Dropped {df_before - df_after_nan_drop} rows due to NaN in critical features.")

    for col in critical_columns:
        if col not in ["date", "stock"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["stock"] = df["stock"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors='coerce')
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    df = df.drop_duplicates(subset=["stock", "date"])
    logger.success(f"✅ Sanity Check Complete: {len(df)} clean rows ready to insert.")
    return df
