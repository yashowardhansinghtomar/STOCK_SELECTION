# Diagnostic alignment script
from core.data_provider.data_provider import load_data
import pandas as pd

recs = load_data("recommendations")
feats = load_data("stock_features_day")

# Normalize
recs = recs.dropna(subset=["date"])
feats = feats.dropna(subset=["date"])
recs["stock"] = recs["stock"].str.strip().str.upper()
feats["stock"] = feats["stock"].str.strip().str.upper()
recs["date"] = pd.to_datetime(recs["date"]).dt.normalize()
feats["date"] = pd.to_datetime(feats["date"]).dt.normalize()

# Find mismatch
merged = recs.merge(feats, on=["stock", "date"], how="left", indicator=True)
missing = merged[merged["_merge"] == "left_only"]
print(f"🔍 Mismatch rows: {len(missing)} / {len(recs)} recommendations unmatched.")
print(missing[["stock", "date"]].head())
