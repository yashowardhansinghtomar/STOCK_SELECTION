# core/data_provider/downsample.py

import pandas as pd
from typing import Dict

def downsample_ohlcv(df_1m: pd.DataFrame, interval: str) -> pd.DataFrame:
    df_1m = df_1m.copy()
    df_1m.index = pd.to_datetime(df_1m.index)
    df_1m = df_1m.sort_index()

    rule_map = {
        "15minute": "15min",
        "60minute": "60min",
        "day": "1D"
    }

    rule = rule_map.get(interval)
    if not rule:
        raise ValueError(f"Unsupported interval: {interval}")

    df_resampled = df_1m.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()

    df_resampled = df_resampled.reset_index().rename(columns={"index": "date"})
    return df_resampled