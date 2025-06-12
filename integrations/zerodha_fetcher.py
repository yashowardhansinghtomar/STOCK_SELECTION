# integrations/zerodha_fetcher.py

import os
import pandas as pd
from datetime import datetime, timedelta
from integrations.zerodha_client import get_kite
from db.conflict_utils import insert_with_conflict_handling
from core.logger.logger import logger
from dateutil.parser import parse
from utils.time_utils import to_naive_utc
from db.postgres_manager import run_query

_instruments_cache = None

INTERVAL_LIMIT_DAYS = {
    "minute": 60,
    "3minute": 60,
    "5minute": 60,
    "10minute": 60,
    "15minute": 60,
    "30minute": 60,
    "60minute": 60,
    "day": 4000,
}

MINIMUM_START_DATE = datetime(2018, 1, 1)  # Clamp to realistic IPO boundary


def get_last_trading_day():
    today = datetime.now()
    while today.weekday() >= 5:
        today -= timedelta(days=1)
    return today


def is_valid_price_df(df):
    required_cols = {"date", "open", "high", "low", "close", "volume"}
    return df is not None and not df.empty and required_cols.issubset(df.columns)


def fetch_historical_data(symbol, interval="day", start=None, end=None, days=365, allow_fallback=True):
    kite = get_kite()
    instrument_token = get_instrument_token(symbol)
    if not instrument_token:
        logger.warning(f"⚠️ Instrument token not found for {symbol}. Skipping...")
        return

    if isinstance(end, str): end = parse(end)
    if isinstance(start, str): start = parse(start)
    if end is None: end = get_last_trading_day()
    if start is None: start = end - timedelta(days=days)

    start = pd.to_datetime(start).tz_localize(None)
    end = pd.to_datetime(end).tz_localize(None)
    
    logger.debug(f"📅 Fetching Zerodha data from {start} to {end} — tzinfo: {start.tzinfo}, {end.tzinfo}")


    start = max(pd.to_datetime(start).normalize(), MINIMUM_START_DATE)
    end = pd.to_datetime(end).normalize()

    interval_days = INTERVAL_LIMIT_DAYS.get(interval, 60)
    date_ranges = []
    temp_end = end

    while temp_end >= start:
        temp_start = max(start, temp_end - timedelta(days=interval_days - 1))
        date_ranges.append((temp_start, temp_end))
        temp_end = temp_start - timedelta(days=1)

    all_df = []

    for idx, (s, e) in enumerate(reversed(date_ranges)):
        try:
            data = kite.historical_data(instrument_token, s, e, interval)
            if not data:
                logger.warning(f"⚠️ No data for {symbol} {interval} ({s.date()} to {e.date()})")
                continue

            df = pd.DataFrame(data)
            df["symbol"] = symbol
            df["interval"] = interval
            df = df[['date', 'symbol', 'interval', 'open', 'high', 'low', 'close', 'volume']]
            df = to_naive_utc(df, "date")

            if is_valid_price_df(df):
                all_df.append(df)
                logger.debug(f"📦 Chunk {idx+1}: {len(df)} rows for {symbol} [{interval}]")
            else:
                logger.warning(f"⛔️ Skipping chunk {idx+1} – incomplete data")

        except Exception as e:
            if "invalid from date" in str(e).lower():
                logger.warning(f"🛑 Aborting chunks for {symbol} due to invalid start date: {e}")
                break  # Stop further chunks
            logger.error(f"❌ Failed chunk {idx+1} for {symbol}: {e}")

    if all_df:
        final_df = pd.concat(all_df).drop_duplicates().sort_values("date")
        if "date" in final_df.columns:
            final_df["date"] = pd.to_datetime(final_df["date"]).dt.tz_localize(None)
        logger.success(f"✅ Successfully fetched {len(final_df)} rows for {symbol} [{interval}]")
        return final_df


    logger.warning(f"⚠️ No usable data for {symbol} [{interval}] after batching")

    # Optional fallback to daily bars
    if allow_fallback and interval != "day":
        logger.warning(f"🧯 Falling back to daily bars for {symbol}")
        return fetch_historical_data(symbol, interval="day", start=start, end=end, days=days, allow_fallback=False)

    return None


def get_instrument_token(symbol):
    query = f"""
        SELECT instrument_token FROM instruments
        WHERE tradingsymbol = '{symbol}'
        ORDER BY last_price DESC
        LIMIT 1;
    """
    result = run_query(query)

    if isinstance(result, list) and result:
        return int(result[0][0])
    elif hasattr(result, 'empty') and not result.empty:
        return int(result['instrument_token'].iloc[0])
    return None


def main():
    symbols = ["RELIANCE"]
    for symbol in symbols:
        fetch_historical_data(symbol, interval="15minute", days=90)

if __name__ == "__main__":
    main()
