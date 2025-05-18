# scripts/load_backup_fundamentals.py

import pandas as pd
from core.data_provider import save_data

CSV_PATH = "fundamental_data.csv"  # your actual path
TABLE_NAME = "stock_fundamentals"

df = pd.read_csv(CSV_PATH)
df = df.rename(columns={
    "SYMBOL": "stock",
    "NAME_OF_COMPANY": "name",
    "P/E Ratio": "pe_ratio",
    "Debt-to-Equity": "debt_to_equity",
    "ROE (%)": "roe",
    "Earnings Growth": "earnings_growth",
    "Market Cap": "market_cap",
    "Sector": "sector",
    "Industry": "industry"
})
save_data(df, TABLE_NAME, if_exists="replace" )
