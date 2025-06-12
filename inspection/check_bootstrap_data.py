from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm

DATABASE_URL = "postgresql+psycopg2://postgres:0809@localhost:5432/trading_db"
PHASE0_START = "2023-01-01"
PHASE0_END = "2023-06-30"

engine = create_engine(DATABASE_URL)
dates = pd.date_range(PHASE0_START, PHASE0_END, freq="B")  # business days

def fetch_table(table, filters=""):
    query = f"SELECT * FROM {table} {filters}"
    return pd.read_sql(query, engine)

# Step 1: Bulk load data
print("📥 Loading required tables in bulk...")
recs_df   = fetch_table("recommendations", "")
feats_df  = fetch_table("stock_features_day", "")
price_df  = fetch_table("stock_price_history", "WHERE interval = 'minute'")
preds_df  = fetch_table("filter_model_predictions", "")

# Step 2: Normalize dates
recs_df["date"]  = pd.to_datetime(recs_df["date"])
feats_df["date"] = pd.to_datetime(feats_df["date"])
price_df["date"] = pd.to_datetime(price_df["date"])
preds_df["date"] = pd.to_datetime(preds_df["date"])

# Step 3: Build report
records = []
for date in tqdm(dates, desc="🔍 Checking dates"):
    prev_date = date - timedelta(days=1)
    records.append({
        "date": date.date(),
        "recommendations": not recs_df[recs_df["date"] <= prev_date].empty,
        "features_day": not feats_df[feats_df["date"] <= prev_date].empty,
        "minute_price": not price_df[price_df["date"] <= prev_date].empty,
        "filter_predictions": not preds_df[preds_df["date"] == date].empty,
    })

df = pd.DataFrame(records)
df.to_csv("fast_bootstrap_data_report.csv", index=False)
print("✅ Report saved to 'fast_bootstrap_data_report.csv'")
