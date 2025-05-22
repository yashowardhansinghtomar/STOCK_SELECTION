# stock_selecter/stock_screener.py

from core.data_provider import load_data, save_data
from core.logger       import logger
from core.config       import settings
import pandas as pd
from datetime import datetime
from core.time_context import get_simulation_date

# ─── your existing filter functions ──────────────────────────────────────────
def filter_growth_stocks(df):
    return df[
        (df['Earnings Growth'] > 0.15) &
        (df['ROE (%)']         > 15)   &
        (df['Market Cap']     > 1e11)
    ]

def filter_value_stocks(df):
    return df[
        (df['P/E Ratio']       < 15)  &
        (df['Debt-to-Equity']  < 0.5) &
        (df['ROE (%)']        > 12)
    ]

def filter_momentum_stocks(df):
    return df[
        (df['Earnings Growth'] > 0.2) &
        (df['ROE (%)']         > 20)  &
        (df['Market Cap']     > 5e10)
    ]

def filter_defensive_stocks(df):
    return df[
        (df['Debt-to-Equity']  < 0.4) &
        (df['ROE (%)']         > 10)  &
        (df['Market Cap']     > 5e11)
    ]

def filter_small_cap_gems(df):
    return df[
        (df['Market Cap']       < 5e10) &
        (df['Earnings Growth'] > 0.2)  &
        (df['ROE (%)']         > 15)
    ]

def filter_high_volatility_stocks(df):
    return df[
        (df['Market Cap']       > 2e10) &
        (df['Earnings Growth'] > 0.1)  &
        (df['P/E Ratio']       > 20)
    ]

# ─── the runner ──────────────────────────────────────────────────────────────
def run_stock_filter(
    filter_name: str,
    output_table: str = settings.ml_selected_stocks_table
) -> pd.DataFrame:
    """
    1) Load fundamentals
    2) Apply filter_name
    3) Persist only 'stock' (+ timestamp) back to SQL
    """
    if not settings.use_fundamentals:
        logger.warning("⚠️ use_fundamentals=False — skipping filter.")
        return pd.DataFrame()
    sim_date = pd.to_datetime(get_simulation_date()).normalize()
    df = load_data(settings.fundamentals_table)
    if "imported_at" in df.columns:
        df["imported_at"] = pd.to_datetime(df["imported_at"]).dt.normalize()
        df = df[df["imported_at"] <= sim_date]
    if df is None or df.empty:
        logger.warning("⚠️ No fundamental data found in SQL.")
        return pd.DataFrame()

    logger.info(f"Loaded {len(df)} rows from '{settings.fundamentals_table}'")
    # rename to match your filter functions…
    df = df.rename(columns={
        "pe_ratio":        "P/E Ratio",
        "debt_to_equity":  "Debt-to-Equity",
        "roe":             "ROE (%)",
        "earnings_growth": "Earnings Growth",
        "market_cap":      "Market Cap",
    })

    filters = {
        "growth":          filter_growth_stocks,
        "value":           filter_value_stocks,
        "momentum":        filter_momentum_stocks,
        "defensive":       filter_defensive_stocks,
        "small_cap_gems":  filter_small_cap_gems,
        "high_volatility": filter_high_volatility_stocks
    }

    if filter_name not in filters:
        logger.error(f"Invalid filter '{filter_name}'. Choose from: {list(filters)}")
        return pd.DataFrame()

    filtered = filters[filter_name](df)
    if filtered.empty:
        logger.warning(f"⚠️ Filter '{filter_name}' returned no rows.")
        return filtered

    # Build a slim output DF for SQL
    out = pd.DataFrame({
        "stock":       filtered["stock"],
        "source":      filter_name,
        "imported_at": datetime.now()
    })

    out = out.drop_duplicates(subset=["stock"])

    save_data(out, output_table, if_exists="replace")
    logger.success(f"✅ Filter '{filter_name}' saved {len(out)} rows to '{output_table}'")
    return filtered


def get_stock_list() -> list[str]:
    """
    Read back the ML-selected table for downstream use.
    """
    df = load_data(settings.ml_selected_stocks_table)
    if "stock" not in df.columns:
        logger.error(f"⚠️ '{settings.ml_selected_stocks_table}' has no 'stock' column.")
        return []
    return df["stock"].dropna().unique().tolist()


if __name__ == "__main__":
    choice = input("Filter type: ")
    run_stock_filter(choice)
