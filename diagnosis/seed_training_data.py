# diagnosis/seed_training_data.py

import pandas as pd
from db.db_router import insert_dataframe
from core.logger import logger

# ✅ Updated seed list with required fields: de_ratio and stock_encoded
seed_examples = [
    {"stock": "INFY", "pe_ratio": 25.0, "roe": 18.5, "sma_short": 50, "sma_long": 200, "rsi_thresh": 60, "market_cap": 250000, "de_ratio": 0.5, "stock_encoded": 101, "label": 1},
    {"stock": "TCS", "pe_ratio": 30.0, "roe": 22.0, "sma_short": 40, "sma_long": 100, "rsi_thresh": 55, "market_cap": 200000, "de_ratio": 0.4, "stock_encoded": 102, "label": 1},
    {"stock": "RELIANCE", "pe_ratio": 45.0, "roe": 10.0, "sma_short": 20, "sma_long": 50, "rsi_thresh": 70, "market_cap": 800000, "de_ratio": 1.0, "stock_encoded": 103, "label": 0},
    {"stock": "ADANIPORTS", "pe_ratio": 60.0, "roe": 8.0, "sma_short": 10, "sma_long": 40, "rsi_thresh": 75, "market_cap": 90000, "de_ratio": 1.2, "stock_encoded": 104, "label": 0}
]

def seed():
    df = pd.DataFrame(seed_examples)
    insert_dataframe(df, "training_data", if_exists="replace")
    logger.success(f"✅ Seeded {len(df)} rows into training_data table.")

if __name__ == "__main__":
    seed()
