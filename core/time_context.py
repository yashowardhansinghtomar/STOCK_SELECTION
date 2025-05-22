import os
import pandas as pd
from datetime import datetime, time
from pandas.tseries.offsets import BDay
import pytz

IST = pytz.timezone("Asia/Kolkata")

def get_simulation_date() -> pd.Timestamp:
    """Return simulation date as timezone-aware pd.Timestamp (IST)"""
    env_date = os.environ.get("SIMULATED_DATE")
    try:
        date = pd.to_datetime(env_date).date() if env_date else datetime.now(IST).date()
    except Exception:
        print(f"⚠️ Invalid SIMULATED_DATE: {env_date}. Falling back to today.")
        date = datetime.now(IST).date()

    # Shift to last business day if weekend
    if pd.Timestamp(date).dayofweek >= 5:
        date = (pd.Timestamp(date) - BDay(1)).date()

    # Return tz-aware timestamp for that date at 00:00
    return IST.localize(datetime.combine(date, time(0, 0)))

def set_simulation_date(sim_date: str):
    os.environ["SIMULATED_DATE"] = sim_date

def clear_simulation_date():
    os.environ.pop("SIMULATED_DATE", None)
