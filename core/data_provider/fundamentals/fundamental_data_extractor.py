# fundamental_data_extractor.py

import pandas as pd
from core.data_provider.data_provider import save_data, load_data
from core.logger.logger import logger

CSV_PATH = "fundamental_data.csv"
TABLE_NAME = "stock_fundamentals"

def load_backup_and_save():
    
    try:
        df = pd.read_csv(CSV_PATH)

        df.columns = [
            "stock", "name", "pe_ratio", "debt_to_equity", "roe",
            "earnings_growth", "market_cap", "sector", "industry"
        ]

        df["pe_ratio"] = pd.to_numeric(df["pe_ratio"], errors="coerce")
        df["debt_to_equity"] = pd.to_numeric(df["debt_to_equity"], errors="coerce")
        df["roe"] = pd.to_numeric(df["roe"], errors="coerce")
        df["earnings_growth"] = pd.to_numeric(df["earnings_growth"], errors="coerce")
        df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")

        df = df.dropna(subset=["stock", "pe_ratio", "market_cap"])

        save_data(df, TABLE_NAME, if_exists="replace")
        logger.success(f"✅ Loaded {len(df)} rows into '{TABLE_NAME}'")
    except Exception as e:
        logger.error(f"❌ Failed to load fundamentals from backup: {e}")

def fetch_all():
    return load_data(TABLE_NAME)

if __name__ == "__main__":
    load_backup_and_save()
