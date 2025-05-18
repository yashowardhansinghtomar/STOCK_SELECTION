from datetime import datetime
from pandas.tseries.offsets import BDay
import pandas as pd
import os

from core.data_provider import load_data

def get_simulation_date():
    """Return SIMULATED_DATE from env or fallback to latest available date in price history."""
    date = os.environ.get("SIMULATED_DATE")
    if date:
        try:
            return datetime.strptime(date.strip(), "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            print(f"⚠️ Malformed SIMULATED_DATE: {date}. Falling back.")

    # Load max available date from stock_price_history
    df = load_data("stock_price_history")
    if df is not None and "date" in df.columns and not df.empty:
        max_date = pd.to_datetime(df["date"]).max()
        return max_date.strftime("%Y-%m-%d")

    # fallback to last weekday if nothing found
    return (pd.Timestamp.today().normalize() - BDay(1)).strftime("%Y-%m-%d")

def set_simulation_date(sim_date: str):
    os.environ["SIMULATED_DATE"] = sim_date

def clear_simulation_date():
    os.environ.pop("SIMULATED_DATE", None)
