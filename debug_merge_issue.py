# debug_merge_issue.py

import pandas as pd
from core.data_provider.data_provider import load_data
from core.logger.logger import logger

RECS_TABLE = "recommendations"
FEATURE_TABLE = "stock_features_day"

# Load and normalize
recs = load_data(RECS_TABLE)
feats = load_data(FEATURE_TABLE)

recs = recs.dropna(subset=["date"])
feats = feats.dropna(subset=["date"])

recs["stock"] = recs["stock"].astype(str).str.strip().str.upper()
feats["stock"] = feats["stock"].astype(str).str.strip().str.upper()

recs["date"] = pd.to_datetime(recs["date"]).dt.date
feats["date"] = pd.to_datetime(feats["date"]).dt.date

logger.info(f"📊 RECS date range: {recs['date'].min()} to {recs['date'].max()}")
logger.info(f"📊 FEATS date range: {feats['date'].min()} to {feats['date'].max()}")

common_stocks = set(recs["stock"]) & set(feats["stock"])
logger.info(f"🧩 Common stocks: {sorted(list(common_stocks))}")

for stock in sorted(common_stocks):
    rec_dates = set(recs[recs["stock"] == stock]["date"])
    feat_dates = set(feats[feats["stock"] == stock]["date"])
    common_dates = rec_dates & feat_dates

    logger.info(f"🔎 {stock}: {len(common_dates)} matching dates")

# Actual merge preview
merged = recs.merge(feats, on=["stock", "date"], how="inner")
logger.info(f"✅ FINAL MERGED ROWS: {len(merged)}")
