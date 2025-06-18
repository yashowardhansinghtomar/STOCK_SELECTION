# core/data_provider/data_provider.py

from utils.time_utils import to_naive_utc, ensure_df_naive_utc, make_naive_index
from core.data_provider.downsample import downsample_ohlcv
from typing import Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
from db.models import Base
from sqlalchemy import inspect
from db.db import engine
from utils.time_utils import to_naive_datetime

from core.logger.logger import logger
from core.config.config import settings
from db.db import SessionLocal
from db.conflict_utils import insert_with_conflict_handling
from integrations.zerodha_fetcher import fetch_historical_data
from db.models import (
    Instrument,
    SkiplistStock,
    StockFundamental,
    Recommendation,
    OpenPosition,
    PaperTrade,
    StockEncoding,
    StockPriceHistory,
    StockFeatureDay,
    StockFeature15m,
    StockFeature60m,
    StockFeature1m,
    MLSelectedStock,
    FilterModelPrediction,
    ParamModelPrediction,
    PriceModelPrediction,
)

ORM_MODEL_MAP = {
    settings.tables.instruments:       Instrument,
    settings.tables.skiplist:          SkiplistStock,
    settings.tables.fundamentals:      StockFundamental,
    settings.tables.recommendations:   Recommendation,
    settings.tables.open_positions:    OpenPosition,
    settings.tables.trades:            PaperTrade,
    settings.tables.encoding:          StockEncoding,
    settings.tables.price_history:     StockPriceHistory,
    settings.tables.features["day"]:       StockFeatureDay,
    settings.tables.features["15minute"]:       StockFeature15m,
    settings.tables.features["60minute"]:       StockFeature60m,
    settings.tables.features["minute"]:        StockFeature1m,
    settings.tables.ml_selected:           MLSelectedStock,
    settings.tables.predictions["filter"]: FilterModelPrediction,
    settings.tables.predictions["param"]: ParamModelPrediction,
    settings.tables.predictions["price"]: PriceModelPrediction,
}

_log_cache = set()
def log_once(key: str, level: str = "info", message: str = ""):
    if key in _log_cache:
        return
    _log_cache.add(key)
    getattr(logger, level)(message)

def fetch_stock_data(symbol: str, start: str = None, end: str = None, interval: str = None, days: int = None) -> pd.DataFrame:
    from pytz import timezone
    from integrations.zerodha_fetcher import MINIMUM_START_DATE
    from utils.time_utils import make_naive  # ✅ fix tz mismatch

    ist = timezone("Asia/Kolkata")
    interval = interval or settings.price_fetch_interval

    INTERVAL_ALIASES = {
        "1m": "minute", "3m": "3minute", "5m": "5minute",
        "10m": "10minute", "15m": "15minute", "30m": "30minute",
        "60m": "60minute", "day": "day", "week": "week", "month": "month",
    }
    normalized_interval = INTERVAL_ALIASES.get(interval, interval)

    if isinstance(start, str): start = pd.to_datetime(start)
    if isinstance(end, str): end = pd.to_datetime(end)

    now_ist = datetime.now(ist)
    if not end:
        end = now_ist
    if not start:
        # Only fallback to settings.price_fetch_days if interval is NOT minute-based
        if interval in ("minute", "1m", "15minute", "60minute"):
            days = days or 2  # 1–2 days max
        else:
            days = days or settings.price_fetch_days
        start = end - timedelta(days=days)


    if end.tzinfo is None: end = ist.localize(end)
    else: end = end.astimezone(ist)

    if start.tzinfo is None: start = ist.localize(start)
    else: start = start.astimezone(ist)

    # 🛠 Normalize both to tz-naive before comparison
    start = make_naive(start)
    MINIMUM_START_DATE = make_naive(MINIMUM_START_DATE)
    start = max(start, MINIMUM_START_DATE)

    session = SessionLocal()
    if session.query(SkiplistStock).filter(SkiplistStock.stock == symbol).first():
        if log_once(f"skiplist:{symbol}"):
            logger.warning(f"⏩ Skipping {symbol} — already in skiplist.")
        session.close()
        return pd.DataFrame()


    try:
        Model = ORM_MODEL_MAP[settings.tables.price_history]
        recs = (
            session.query(Model)
            .filter(Model.symbol == symbol)
            .filter(Model.interval == normalized_interval)
            .filter(Model.date >= start.date())
            .filter(Model.date <= end.date())
            .order_by(Model.date)
            .all()
        )
        if recs:
            df = pd.DataFrame([r.__dict__ for r in recs])
            df = df.drop(columns=["_sa_instance_state"], errors="ignore")
            df = to_naive_utc(df, "date")
            df = df.set_index("date").sort_index()
            df.index = make_naive_index(df.index)
            df = df.loc[start.date():end.date()]
            df.attrs["start"] = start.date()
            df.attrs["end"] = end.date()
            return df
    except Exception as e:
        logger.warning(f"⚠ Could not load cached {interval} data for {symbol}: {e}")
    finally:
        session.close()

    logger.info(f"📡 Fetching historical data for {symbol} ({start.date()} → {end.date()})")
    df = fetch_historical_data(symbol, start=start, end=end, interval="minute")
    if df is not None and not df.empty:
        is_fake_minute = (
            df is not None and
            df.shape[0] == 1 and
            pd.to_datetime(df["date"].iloc[0]).time() == datetime.strptime("18:30", "%H:%M").time()
        )

        if is_fake_minute:
            logger.warning(f"⚠ {symbol} returned only 1 row with 18:30 candle — likely fallback daily bar. Saving as 'day'.")
            df["interval"] = "day"
            save_data(df, settings.tables.price_history)
            return fetch_stock_data(symbol, start=start, end=end, interval="day")  # 🔁 re-fetch as proper daily

    if (
        df is not None and not df.empty and
        normalized_interval == "minute" and
        df.shape[0] == 1 and
        pd.to_datetime(df["date"].iloc[0]).time() == datetime.strptime("18:30", "%H:%M").time()
    ):

        logger.warning(f"⚠️ {symbol} returned only 1 row with 18:30 candle — likely fallback daily bar. Saving as 'day'.")

        # ✅ Correct interval and timestamp
        df["interval"] = "day"
        df["date"] = pd.to_datetime(df["date"]).dt.floor("D")
        df.set_index("date", inplace=True)
        df = df[~df.index.duplicated(keep="last")]

        # ✅ Save cleanly as day-level data
        save_data(df, settings.tables.price_history)
        return df  # or return fetch_stock_data(..., interval="day") if you want to retry properly


    if df is not None and not df.empty:
        df["interval"] = "minute"
        save_data(df, settings.tables.price_history, if_exists="update")
        df = to_naive_utc(df, "date")
        df = df.set_index("date").sort_index()
        df.index = make_naive_index(df.index)
        df = df.loc[start.date():end.date()]
        df.attrs["start"] = start.date()
        df.attrs["end"] = end.date()

        for target_interval in ["15minute", "60minute", "day"]:
            try:
                down = downsample_ohlcv(df.copy(), target_interval)
                if not down.empty:
                    down["symbol"] = symbol
                    down["interval"] = target_interval
                    save_data(down, settings.tables.price_history)
                    log_once(f"{symbol}_{target_interval}_downsample", "debug", f"✅ Downsampled & saved {symbol} → {target_interval}")
            except Exception as e:
                log_once(f"{symbol}_{target_interval}_fail", "error", f"❌ Downsample failed for {symbol} → {target_interval}: {e}")

        if normalized_interval == "minute":
            return df


        try:
            return fetch_stock_data(symbol, start=start, end=end, interval=normalized_interval)
        except Exception as e:
            logger.warning(f"⚠ Could not fetch downsampled {normalized_interval} data for {symbol}: {e}")
            return pd.DataFrame()

    logger.warning(f"⛔ No data available for {symbol}. Adding to skiplist.")
    session = SessionLocal()
    try:
        skip = SkiplistStock(stock=symbol, reason="Missing price data", date_added=datetime.now())
        session.merge(skip)
        session.commit()
        logger.info(f"📛 {symbol} added to skiplist.")
    except Exception as e:
        logger.warning(f"⚠ Could not add {symbol} to skiplist: {e}")
    finally:
        session.close()

    return pd.DataFrame()

def save_data(df: pd.DataFrame, table_name: str, if_exists: str = "ignore") -> None:
    if df is None or df.empty:
        logger.warning(f"⚠️ Not saving '{table_name}': DataFrame is empty.")
        return

    # Convert datetime columns to naive UTC if required
    dt_cols = settings.date_columns.get(table_name, [])
    if dt_cols:
        df = ensure_df_naive_utc(df, dt_cols)

    # Drop rows with missing critical columns
    required_cols = ["date", "symbol", "interval", "open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if not missing_cols:
        df = df.dropna(subset=required_cols)
        try:
            df["date"] = pd.to_datetime(df["date"])
            df["symbol"] = df["symbol"].astype(str)
            df["interval"] = df["interval"].astype(str)
            df["open"] = df["open"].astype(float)
            df["high"] = df["high"].astype(float)
            df["low"] = df["low"].astype(float)
            df["close"] = df["close"].astype(float)
            df["volume"] = df["volume"].astype(int)
        except Exception as type_err:
            logger.error(f"❌ Data type casting failed in save_data('{table_name}'): {type_err}")
            return
    else:
        logger.debug(f"Skipping strict casting for '{table_name}' — missing expected columns: {missing_cols}")

    # Resolve physical table name
    phys = settings.table_map.get(table_name, table_name)
    try:
        insert_with_conflict_handling(df, phys, if_exists=if_exists)
        logger.success(f"✅ Saved data to '{phys}' successfully.")
    except Exception as e:
        logger.error(f"❌ save_data to '{phys}' failed: {e}")


def load_data(table_name: str, interval: str = None, stock: str = None, start: str = None, end: str = None) -> pd.DataFrame:
    session = SessionLocal()
    try:
        model = ORM_MODEL_MAP.get(table_name)
        if not model:
            logger.error(f"Unknown table requested: {table_name}")
            return pd.DataFrame()

        query = session.query(model)

        if hasattr(model, "stock") and stock:
            query = query.filter(model.stock == stock.upper().strip())
        if hasattr(model, "date"):
            if start:
                query = query.filter(model.date >= pd.to_datetime(start).date())
            if end:
                query = query.filter(model.date <= pd.to_datetime(end).date())

        records = query.all()
        df = pd.DataFrame([r.__dict__ for r in records])
        if df.empty:
            return df

        df = df.drop(columns=[c for c in df.columns if c.startswith("sa")], errors="ignore")

        dt_cols = settings.date_columns.get(table_name, [])
        if dt_cols:
            df = ensure_df_naive_utc(df, dt_cols)

        if "symbol" in df.columns and "stock" not in df.columns:
            df = df.rename(columns={"symbol": "stock"})

        return df
    except Exception as e:
        logger.error(f"load_data('{table_name}') failed: {e}")
        return pd.DataFrame()
    finally:
        session.close()



def get_last_close(symbol: str, sim_date: datetime = None) -> Optional[float]:
    try:
        sim_date = pd.to_datetime(sim_date or datetime.now()).normalize()
        logger.debug(f"🧪 get_last_close: {symbol} as of {sim_date} | tzinfo: {sim_date.tzinfo}")

        # Step 1: Try 1-minute bars with expanding lookback
        minute_lookback = 1
        max_minute_lookback = 5
        while minute_lookback <= max_minute_lookback:
            df_1m = fetch_stock_data(
                symbol,
                interval="1m",
                end=sim_date + timedelta(days=1),
                days=minute_lookback
            )
            if df_1m is not None and not df_1m.empty:
                df_1m = df_1m.reset_index() if df_1m.index.name else df_1m
                if "date" in df_1m.columns:
                    df_1m = df_1m.rename(columns={"date": "timestamp"})
                df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"]).dt.tz_localize(None)
                df_1m["date_only"] = df_1m["timestamp"].dt.normalize()

                df_filtered = df_1m[df_1m["date_only"] == sim_date]
                if not df_filtered.empty:
                    close = float(df_filtered.sort_values("timestamp")["close"].iloc[-1])
                    logger.debug(f"✅ {symbol} (1m, {minute_lookback}d): Close = {close}")
                    return close

            logger.debug(f"⚠ {symbol} (1m): no data in last {minute_lookback}d, expanding lookback")
            minute_lookback *= 2

        # Step 2: Fallback to daily with expanding lookback
        daily_lookback = 5
        max_daily_lookback = settings.price_fetch_days
        while daily_lookback <= max_daily_lookback:
            df_day = fetch_stock_data(
                symbol,
                interval="day",
                end=sim_date,
                days=daily_lookback
            )
            if df_day is not None and not df_day.empty:
                df_day = df_day.reset_index() if df_day.index.name else df_day
                # normalize to midnight timestamps
                if "date" in df_day.columns:
                    df_day["timestamp"] = (
                        pd.to_datetime(df_day["date"])
                          .dt.tz_localize(None)
                          .dt.normalize()
                    )
                else:
                    df_day["timestamp"] = (
                        pd.to_datetime(df_day["timestamp"])
                          .dt.tz_localize(None)
                          .dt.normalize()
                    )

                # 🩹 Ensure timestamp column is timezone-naive
                if isinstance(df_day["timestamp"].dtype, pd.DatetimeTZDtype):
                    df_day["timestamp"] = df_day["timestamp"].dt.tz_localize(None)

                # 🩹 Ensure sim_date is timezone-naive
                if sim_date.tzinfo is not None:
                    sim_date = sim_date.replace(tzinfo=None)

                # ✅ Now safe to compare
                df_day = df_day[df_day["timestamp"] <= sim_date]

                if not df_day.empty:
                    price = float(df_day.sort_values("timestamp")["close"].iloc[-1])
                    logger.success(
                        f"✅ Daily close for {symbol} over last {daily_lookback}d = {price}"
                    )
                    return price

            logger.debug(f"⚠ {symbol} (day): no data in last {daily_lookback}d, expanding lookback")
            daily_lookback *= 2

        logger.warning(
            f"⚠ No usable data for {symbol} after looking back up to {max_daily_lookback} days"
        )
        return None

    except Exception as e:
        logger.exception(f"❌ get_last_close('{symbol}') failed: {e}")
        return None

def list_partitions(base_table_prefix: str = None) -> List[str]:
    prefix = base_table_prefix or "stock_features"
    session = SessionLocal()
    try:
        result = session.execute(
            f"SELECT tablename FROM pg_tables WHERE tablename LIKE '{prefix}_%';"
        )
        return [row[0] for row in result]
    except Exception as e:
        logger.error(f"list_partitions failed: {e}")
        return []
    finally:
        session.close()

def ensure_price_table(interval: str):
    table_name = settings.tables.price_history
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        logger.warning(f" Table '{table_name}' not found. Creating it...")
        Base.metadata.tables[table_name].create(bind=engine, checkfirst=True)

def cache_price(df: pd.DataFrame):
    if df is None or df.empty:
        return
    if "interval" not in df.columns:
        df["interval"] = "day"
    df["interval"] = df["interval"].fillna("day")
    save_data(df, settings.tables.price_history)
