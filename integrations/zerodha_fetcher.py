import os
import pandas as pd
from datetime import datetime, timedelta
from integrations.zerodha_client import get_kite
from db.conflict_utils import insert_with_conflict_handling
from core.logger import logger
from dateutil.parser import parse
from utils.time_utils import to_naive_utc

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

def get_last_trading_day():
    today = datetime.now()
    while today.weekday() >= 5:
        today -= timedelta(days=1)
    return today

def is_valid_price_df(df):
    required_cols = {"date", "open", "high", "low", "close", "volume"}
    return df is not None and not df.empty and required_cols.issubset(df.columns)

def fetch_historical_data(symbol, interval="day", start=None, end=None, days=365):
    kite = get_kite()
    instrument_token = get_instrument_token(symbol)
    if not instrument_token:
        logger.warning(f"⚠️ Instrument token not found for {symbol}. Skipping...")
        return

    if isinstance(end, str): end = parse(end)
    if isinstance(start, str): start = parse(start)
    if end is None: end = get_last_trading_day()
    if start is None: start = end - timedelta(days=days)

    interval_days = INTERVAL_LIMIT_DAYS.get(interval, 60)
    date_ranges = []
    temp_end = end

    while temp_end > start:
        temp_start = max(start, temp_end - timedelta(days=interval_days))
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
            logger.error(f"❌ Failed chunk {idx+1} for {symbol}: {e}")

    if all_df:
        final_df = pd.concat(all_df).drop_duplicates().sort_values("date")
        logger.success(f"✅ Successfully fetched {len(final_df)} rows for {symbol} [{interval}]")
        return final_df


    logger.warning(f"⚠️ No usable data for {symbol} [{interval}] after batching")
    return None

def get_instrument_token(symbol):
    instruments_df = pd.read_csv("data/instruments.csv")
    instrument = instruments_df[instruments_df['tradingsymbol'] == symbol]
    if not instrument.empty:
        return int(instrument['instrument_token'].values[0])
    return None

def main():
    symbols = ["RELIANCE"]
    for symbol in symbols:
        fetch_historical_data(symbol, interval="15minute", days=90)

if __name__ == "__main__":
    main()
