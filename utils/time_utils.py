# utils/time_utils.py

import pandas as pd
from pytz import timezone
from datetime import datetime

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


def make_naive(dt):
    """Converts a single datetime to naive UTC."""
    if dt.tzinfo:
        return dt.astimezone(timezone("UTC")).replace(tzinfo=None)
    return dt


def ensure_df_naive_utc(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        df = to_naive_utc(df, col)
    return df


def assert_naive(dt, context="datetime"):
    assert dt.tzinfo is None, f"{context} must be naive, got: {dt}"


def make_naive_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Forces a datetime index to be offset-naive.
    """
    if index.tz is not None:
        return index.tz_convert(None)
    return index

def to_naive_utc_timestamp(dt):
    dt = pd.to_datetime(dt)
    if dt.tzinfo is not None:
        dt = dt.tz_convert("UTC").tz_localize(None)
    return dt.normalize()

def to_naive_datetime(dt: datetime) -> datetime:
    dt = pd.to_datetime(dt)
    if dt.tzinfo is not None:
        return dt.tz_convert(None).to_pydatetime()
    return dt.to_pydatetime()
