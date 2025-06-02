# core/feature_engineering/regime_features.py

import pandas as pd
import numpy as np

def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volatility and trend signals to classify market regime.
    Returns a DataFrame with added regime tag columns.
    """
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["atr"] = abs(df["high"] - df["low"])
    df["atr_pct"] = df["atr"] / df["close"]

    df["volatility"] = df["returns"].rolling(10).std()
    df["trend_strength"] = abs(df["close"].rolling(10).mean() - df["close"]) / df["close"]

    def classify_regime(row):
        if row["volatility"] > 0.02:
            return "volatile"
        elif row["trend_strength"] > 0.03:
            return "trending"
        else:
            return "sideways"

    df["regime_tag"] = df.apply(classify_regime, axis=1)
    return df[["date", "regime_tag", "volatility", "trend_strength", "atr_pct"]]
