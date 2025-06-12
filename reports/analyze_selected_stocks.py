import pandas as pd
from sqlalchemy import create_engine

from core.config.config import settings
DB_URL = settings.database_url

engine = create_engine(DB_URL)

# Load latest predictions
recs = pd.read_sql("""
    SELECT *
    FROM filter_model_predictions
    WHERE date = (SELECT MAX(date) FROM filter_model_predictions)
""", engine)

# Load matching features
features = pd.read_sql("""
    SELECT *
    FROM stock_features_day
    WHERE date = (SELECT MAX(date) FROM stock_features_day)
""", engine)

# Load fundamentals if available
fundamentals = pd.read_sql("""
    SELECT stock, sector, industry, pe_ratio, market_cap
    FROM stock_fundamentals
    WHERE imported_at = (SELECT MAX(imported_at) FROM stock_fundamentals)
""", engine)

# Merge all
df = recs.merge(features, on="stock").merge(fundamentals, on="stock", how="left")

# Summary
print("\n📊 Confidence Scores:")
print(df["confidence"].describe())

print("\n📈 RSI Thresholds:")
print(df["rsi_thresh"].describe())

print("\n📉 SMA Short:")
print(df["sma_short"].describe())

print("\n💡 Volume Spike Frequency:")
print(f"{df['volume_spike'].mean() * 100:.2f}% of stocks show volume spike")

if "sector" in df.columns:
    print("\n🏢 Sector Distribution:")
    print(df["sector"].value_counts().head(10))

# Optional: Save CSV
df.to_csv("selected_stock_summary.csv", index=False)
