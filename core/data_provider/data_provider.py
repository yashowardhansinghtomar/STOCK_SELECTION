
from utils.time_utils import to_naive_utc
from core.data_provider.downsample import downsample_ohlcv
from typing import Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
from db.models import Base
from sqlalchemy import inspect
from db.db import engine

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
)

ORM_MODEL_MAP = {
    settings.instruments_table:       Instrument,
    settings.skiplist_table:          SkiplistStock,
    settings.fundamentals_table:      StockFundamental,
    settings.recommendations_table:   Recommendation,
    settings.open_positions_table:    OpenPosition,
    settings.trades_table:            PaperTrade,
    settings.encoding_table:          StockEncoding,
    settings.price_history_table:     StockPriceHistory,
    settings.feature_table_day:       StockFeatureDay,
    settings.feature_table_15m:       StockFeature15m,
    settings.feature_table_60m:       StockFeature60m,
    settings.feature_table_1m:        StockFeature1m,
    settings.ml_selected_stocks_table: MLSelectedStock,
}

def fetch_stock_data(symbol: str, start: str = None, end: str = None, interval: str = None, days: int = None) -> pd.DataFrame:
    from pytz import timezone
    ist = timezone("Asia/Kolkata")

    interval = interval or settings.price_fetch_interval

    INTERVAL_ALIASES = {
        "1m": "minute",
        "3m": "3minute",
        "5m": "5minute",
        "10m": "10minute",
        "15m": "15minute",
        "30m": "30minute",
        "60m": "60minute",
        "day": "day",
        "week": "week",
        "month": "month",
    }

    normalized_interval = INTERVAL_ALIASES.get(interval, interval)

    if isinstance(end, str): end = pd.to_datetime(end)
    if isinstance(start, str): start = pd.to_datetime(start)

    now_ist = datetime.now(ist)
    if not end:
        end = now_ist
    if not start:
        days = days or settings.price_fetch_days
        start = end - timedelta(days=days)

    if end.tzinfo is None: end = ist.localize(end)
    else: end = end.astimezone(ist)

    if start.tzinfo is None: start = ist.localize(start)
    else: start = start.astimezone(ist)

    session = SessionLocal()

    if session.query(SkiplistStock).filter(SkiplistStock.stock == symbol).first():
        logger.warning(f" Skipping {symbol} — already in skiplist.")
        session.close()
        return pd.DataFrame()

    try:
        Model = ORM_MODEL_MAP[settings.price_history_table]
        recs = (
            session.query(Model)
            .filter(Model.symbol == symbol)
            .filter(Model.interval == normalized_interval)
            .filter(Model.date <= end.date())
            .order_by(Model.date)
            .all()
        )
        if recs:
            df = pd.DataFrame([r.__dict__ for r in recs])
            df = df.drop(columns=["_sa_instance_state"], errors="ignore")
            df = to_naive_utc(df, "date")
            return df.set_index("date").sort_index()
    except Exception as e:
        logger.warning(f" Could not load cached {interval} data for {symbol}: {e}")
    finally:
        session.close()

    logger.info(f" Fetching 1-minute data for {symbol} for all intervals")
    df_1m = fetch_historical_data(symbol, start=start, end=end, interval="minute")

    if df_1m is not None and not df_1m.empty:
        df_1m["interval"] = "minute"
        save_data(df_1m, settings.price_history_table, if_exists="append")
        df_1m = to_naive_utc(df_1m, "date")
        df_1m = df_1m.set_index("date").sort_index()

        for target_interval in ["15minute", "60minute", "day"]:
            try:
                down = downsample_ohlcv(df_1m.copy(), target_interval)
                if not down.empty:
                    down["symbol"] = symbol
                    down["interval"] = target_interval
                    save_data(down, settings.price_history_table)
                    logger.success(f" Downsampled & saved {symbol} → {target_interval}")
            except Exception as e:
                logger.error(f" Downsample failed for {symbol} → {target_interval}: {e}")

        if normalized_interval == "minute":
            return df_1m
        else:
            return fetch_stock_data(symbol, start=start, end=end, interval=normalized_interval)

    logger.warning(f"No 1-minute data fetched for {symbol}")
    session = SessionLocal()
    try:
        skip = SkiplistStock(stock=symbol, reason=f"Missing 1minute data", date_added=datetime.now())
        session.merge(skip)
        session.commit()
        logger.info(f"{symbol} added to skiplist.")
    except Exception as e:
        logger.warning(f" Could not add {symbol} to skiplist: {e}")
    finally:
        session.close()

    return pd.DataFrame()

def save_data(df: pd.DataFrame, table_name: str, if_exists: str = "append") -> None:
    if df is None or df.empty:
        logger.warning(f" Not saving '{table_name}': DataFrame is empty.")
        return
    if "date" in df.columns:
        df = to_naive_utc(df, "date")

    phys = settings.table_map.get(table_name, table_name)
    try:
        insert_with_conflict_handling(df, phys, if_exists=if_exists)
        logger.success(f" Saved data to '{phys}' successfully.")
    except Exception as e:
        logger.error(f"save_data to '{phys}' failed: {e}")

def load_data(table_name: str, interval: str = None) -> pd.DataFrame:
    session = SessionLocal()
    try:
        model = ORM_MODEL_MAP.get(table_name)
        if not model:
            logger.error(f"Unknown table requested: {table_name}")
            return pd.DataFrame()

        records = session.query(model).all()
        df = pd.DataFrame([r.__dict__ for r in records])
        if df.empty:
            return df
        df = df.drop(columns=[c for c in df.columns if c.startswith("_sa_")], errors="ignore")

        for dt_col in settings.date_columns.get(table_name, []):
            if dt_col in df.columns:
                df = to_naive_utc(df, dt_col)

        if "symbol" in df.columns and "stock" not in df.columns:
            df = df.rename(columns={"symbol": "stock"})

        return df
    except Exception as e:
        logger.error(f"load_data('{table_name}') failed: {e}")
        return pd.DataFrame()
    finally:
        session.close()

def get_last_close(symbol: str) -> Optional[float]:
    session = SessionLocal()
    try:
        Model = ORM_MODEL_MAP[settings.price_history_table]
        rec = (
            session.query(Model)
            .filter(Model.symbol == symbol)
            .order_by(Model.date.desc())
            .first()
        )
        return float(rec.close) if rec else None
    except Exception as e:
        logger.error(f"get_last_close('{symbol}') failed: {e}")
        return None
    finally:
        session.close()

def delete_cached_features(stock: str, sim_date: datetime, interval: str = "day") -> None:
    session = SessionLocal()
    try:
        table_key = {
            "day": settings.feature_table_day,
            "15minute": settings.feature_table_15m,
            "60minute": settings.feature_table_60m,
            "minute": settings.feature_table_1m,
        }.get(interval)

        Model = ORM_MODEL_MAP.get(table_key)
        if not Model:
            logger.warning(f" No ORM model for interval: {interval}")
            return

        session.query(Model) \
            .filter(Model.stock == stock) \
            .filter(Model.date == sim_date) \
            .delete()
        session.commit()
        logger.success(f" Deleted cached features for {stock} on {sim_date} ({interval})")
    except Exception as e:
        logger.error(f"delete_cached_features failed: {e}")
    finally:
        session.close()

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
    table_name = settings.price_history_table
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
    save_data(df, settings.price_history_table)
