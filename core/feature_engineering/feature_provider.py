import pandas as pd
from core.data_provider.data_provider import fetch_stock_data, load_data
from core.feature_engineering.precompute_features import compute_features, insert_feature_row
from db.db import SessionLocal
from core.logger.logger import logger
from core.config.config import settings
from sqlalchemy.sql import text
from core.skiplist.skiplist import is_in_skiplist
from datetime import datetime, timedelta, date

FREQ_MAP = {
    "day": "B",
    "60minute": "60min",
    "15minute": "15min",
    "minute": "T"
}


def process_and_insert(df_price: pd.DataFrame, stock: str, interval: str) -> pd.DataFrame:
    stock = stock.strip().upper()
    df_price["stock"] = stock
    df_price["stock_encoded"] = hash(stock) % 10000
    df_feat = compute_features(df_price)

    if df_feat.empty:
        return pd.DataFrame()

    if "date" not in df_feat.columns:
        if isinstance(df_feat.index, pd.DatetimeIndex):
            df_feat = df_feat.reset_index()
        else:
            logger.error(f"Missing 'date' in computed features for {stock} @ {interval}")
            return pd.DataFrame()

    df_feat["date"] = pd.to_datetime(df_feat["date"]).dt.date

    # Trim to range if attrs are set
    start = df_price.attrs.get("start")
    end = df_price.attrs.get("end")
    if isinstance(start, datetime): start = start.date()
    if isinstance(end, datetime): end = end.date()

    if start and end:
        pre_trim = len(df_feat)
        df_feat = df_feat[(df_feat["date"] >= start) & (df_feat["date"] <= end)]
        logger.info(f"✂️ Trimmed features for {stock} @ {interval} to {start} → {end} ({pre_trim} → {len(df_feat)})")
    else:
        logger.warning(f"⚠️ No trimming enforced — attrs missing for {stock} @ {interval}")

    if df_feat.empty:
        logger.warning(f"No features to insert after trimming for {stock} @ {interval}")
        return pd.DataFrame()

    table = settings.interval_feature_table_map.get(interval)
    if not table:
        logger.error(f"Table not found for interval: {interval}")
        return pd.DataFrame()

    session = SessionLocal()
    try:
        for _, row in df_feat.iterrows():
            insert_feature_row(session, table, row.to_dict(), refresh=True)
        session.commit()
        logger.success(f"Inserted {len(df_feat)} features for {stock} @ {interval}")
    except Exception as e:
        session.rollback()
        logger.error(f"Insert failed for {stock} @ {interval}: {e}")
    finally:
        session.close()

    return df_feat


def fetch_features(stock: str, interval: str, refresh_if_missing: bool = True, start: str = None, end: str = None) -> pd.DataFrame:
    stock = stock.strip().upper()

    if is_in_skiplist(stock):
        logger.warning(f"Skipping {stock} — already in skiplist.")
        return pd.DataFrame()

    table = settings.interval_feature_table_map.get(interval)
    if not table:
        logger.error(f"Unknown interval: {interval}")
        return pd.DataFrame()

    # Normalize start/end
    if isinstance(start, str): start = pd.to_datetime(start)
    if isinstance(end, str): end = pd.to_datetime(end)
    if isinstance(start, date): start = datetime.combine(start, datetime.min.time())
    if isinstance(end, date): end = datetime.combine(end, datetime.min.time())

    logger.debug(f"📥 Looking up {stock} @ {interval} ({start.date() if start else 'None'} → {end.date() if end else 'None'})")

    # Query cached features
    session = SessionLocal()
    try:
        query = f"SELECT * FROM {table} WHERE stock = :stock"
        params = {"stock": stock}
        if start: query += " AND date >= :start"; params["start"] = start.date()
        if end:   query += " AND date <= :end";   params["end"] = end.date()

        df = pd.read_sql(text(query + " ORDER BY date"), session.bind, params=params)

        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            if start and end and interval in FREQ_MAP:
                expected = pd.date_range(start=start, end=end, freq=FREQ_MAP[interval])
                df.set_index("date", inplace=True)
                if expected.difference(df.index).empty:
                    return df.reset_index()
                df.reset_index(inplace=True)
            return df
    except Exception as e:
        logger.warning(f"Error checking cached features for {stock} @ {interval}: {e}")
    finally:
        session.close()

    # Fallback: fetch fresh 1m data and derive from it
    df_1m = fetch_stock_data(stock, interval="minute", start=start, end=end)
    if df_1m is None or df_1m.empty:
        logger.warning(f"No 1m price data for {stock} @ {interval} ({start} → {end})")
        return pd.DataFrame()

    df_1m.attrs["start"] = start.date() if start else None
    df_1m.attrs["end"] = end.date() if end else None

    df_main = process_and_insert(df_1m.copy(), stock, interval)

    if interval == "minute":
        for other_interval in ["15minute", "60minute", "day"]:
            try:
                df_down = df_1m.copy()
                df_down.attrs["start"] = df_1m.attrs["start"]
                df_down.attrs["end"] = df_1m.attrs["end"]
                df_down["interval"] = other_interval
                process_and_insert(df_down, stock, other_interval)
            except Exception as e:
                logger.warning(f"Failed to precompute for {other_interval}: {e}")

    return df_main


def fetch_features_with_backfill(stock: str, interval: str, sim_date=None, start=None, end=None) -> pd.DataFrame:
    stock = stock.strip().upper()

    LOOKBACK_PADDING = {
        "day": 30,
        "60minute": 5,
        "15minute": 2,
        "minute": 1,
    }.get(interval, 10)

    # If sim_date is given, calculate default start/end
    if sim_date:
        sim_date = pd.to_datetime(sim_date).normalize()
        if not end:
            end = sim_date
        if not start:
            start = end - timedelta(days=LOOKBACK_PADDING)
    else:
        end = pd.to_datetime(end or datetime.today())
        start = pd.to_datetime(start or (end - timedelta(days=settings.price_fetch_days)))
    
    # Load cached
    df = load_data(
        table_name=settings.interval_feature_table_map[interval],
        stock=stock,
        start=start,
        end=end
    )

    if not df.empty and sim_date:
        df = df[df["date"] == sim_date]

    if not df.empty:
        return df

    logger.warning(f"⚠️ Fallback to fetch: {stock} @ {interval} (start={start.date()}, end={end.date()})")
    return fetch_features(stock, interval, start=start, end=end)
