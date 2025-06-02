import pandas as pd
from datetime import datetime
from core.feature_engineering.feature_enricher import enrich_features

def enrich_multi_interval_features(stock: str, sim_date: datetime, intervals: list = ["day", "60minute", "15minute"]) -> pd.DataFrame:
    dfs = []
    for interval in intervals:
        df = enrich_features(stock, sim_date, interval)
        if not df.empty:
            df = df.add_suffix(f"_{interval}")
            df = df.rename(columns={f"stock_{interval}": "stock", f"date_{interval}": "date"})
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    # Merge all interval features on stock + date
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on=["stock", "date"], how="outer")

    return merged
