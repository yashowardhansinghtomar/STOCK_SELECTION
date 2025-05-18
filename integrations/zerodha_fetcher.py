import os
import pandas as pd
from datetime import datetime, timedelta
from integrations.zerodha_client import get_kite
from db.conflict_utils import insert_with_conflict_handling
from core.logger import logger
from dateutil.parser import parse

_instruments_cache = None

def get_last_trading_day():
    today = datetime.now()
    while today.weekday() >= 5:  # Saturday = 5, Sunday = 6
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

    if isinstance(end, str):
        end = parse(end)
    if isinstance(start, str):
        start = parse(start)

    if end is None:
        end = get_last_trading_day()
    if start is None:
        start = end - timedelta(days=days)

    try:
        data = kite.historical_data(
            instrument_token=instrument_token,
            from_date=start,
            to_date=end,
            interval=interval
        )
        if not data:
            logger.warning(f"⚠️ No data returned for {symbol}. Skipping...")
            return

        df = pd.DataFrame(data)
        df['symbol'] = symbol
        df.rename(columns={'date': 'date', 'open': 'open', 'high': 'high',
                           'low': 'low', 'close': 'close', 'volume': 'volume'}, inplace=True)
        df = df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
        df['date'] = pd.to_datetime(df['date'])

        if is_valid_price_df(df):
            insert_with_conflict_handling(df, "stock_price_history")
            logger.success(f"✅ Successfully fetched and stored {len(df)} rows for {symbol}.")
            return df
        else:
            logger.warning(f"⛔️ Skipping {symbol} – invalid or incomplete data")
            return

    except Exception as e:
        logger.error(f"❌ Failed to fetch data for {symbol}: {e}")
        return None

def get_instrument_token(symbol):
    instruments_df = pd.read_csv("data/instruments.csv")
    instrument = instruments_df[instruments_df['tradingsymbol'] == symbol]
    if not instrument.empty:
        return int(instrument['instrument_token'].values[0])
    return None

def main():
    symbols = ['RELIANCE', 'TCS', 'INFY']
    for symbol in symbols:
        fetch_historical_data(symbol)

if __name__ == "__main__":
    main()
