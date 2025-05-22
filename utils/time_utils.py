# utils/time_utils.py

import pandas as pd
from pytz import timezone

IST = timezone("Asia/Kolkata")

def to_naive_utc(df: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    df[column] = pd.to_datetime(df[column], errors="coerce")
    try:
        if hasattr(df[column].dt, "tz") and df[column].dt.tz is not None:
            df[column] = df[column].dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            df[column] = df[column].dt.tz_localize("UTC").dt.tz_localize(None)
    except Exception:
        df[column] = df[column].dt.tz_localize(None)
    return df


def to_ist(df: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    """
    Converts datetime column to IST (timezone-aware).
    """
    df[column] = pd.to_datetime(df[column], errors="coerce")
    df[column] = df[column].dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
    df[column] = df[column].dt.tz_convert(IST)
    return df

def localize_if_needed(df: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    df[column] = pd.to_datetime(df[column], errors="coerce")
    if df[column].dt.tz is not None:
        df[column] = df[column].dt.tz_convert(None)
    return df
