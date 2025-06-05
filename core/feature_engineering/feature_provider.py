# core/feature_engineering/feature_provider.py

import pandas as pd
from datetime import datetime, timedelta
from core.logger.logger import logger
from core.config.config import settings
from core.skiplist.skiplist import is_in_skiplist
from core.feature_store.feature_store import get_or_compute
from core.feature_engineering.regime_features import compute_regime_features
from core.data_provider.data_provider import fetch_stock_data, load_data
from core.feature_engineering.precompute_features import compute_features, insert_feature_row
from db.db import SessionLocal
from sqlalchemy.sql import text

FREQ_MAP = {
    "day": "B",
    "60minute": "60min",
    "15minute": "15min",
    "minute": "T"
}

def process_and_insert(df_price: pd.DataFrame, stock: str, interval: str) -> pd.DataFrame:
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
    if is_in_skiplist(stock):
        logger.warning(f"⏩ Skipping {stock} — already in skiplist.")
        return pd.DataFrame()

    try:
        date = pd.to_datetime(start).date() if start else pd.to_datetime(datetime.now()).date()
        features = get_or_compute(stock, interval, str(date)) if refresh_if_missing else pd.DataFrame()
        if not features.empty:
            raw_df = get_or_compute(stock, interval, str(date), window=20)
            regime_df = compute_regime_features(raw_df)
            features = pd.merge(features, regime_df, on="date", how="left")
            return features
    except Exception as e:
        logger.warning(f"⚠️ Failed to fetch/compute features for {stock} @ {interval}: {e}")

    # Fallback to legacy method if cache fails or feature store not populated
    table = settings.interval_feature_table_map.get(interval)
    if not table:
        logger.error(f"Unknown interval: {interval}")
        return pd.DataFrame()

    if isinstance(start, str): start = pd.to_datetime(start)
    if isinstance(end, str): end = pd.to_datetime(end)

    session = SessionLocal()
    try:
        query = f"SELECT * FROM {table} WHERE stock = :stock"
        params = {"stock": stock}
        if start is not None:
            query += " AND date >= :start"
            params["start"] = start.date()
        if end is not None:
            query += " AND date <= :end"
            params["end"] = end.date()

        df = pd.read_sql(text(query + " ORDER BY date"), session.bind, params=params)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            if start and end and interval in FREQ_MAP:
                expected = pd.date_range(start=start, end=end, freq=FREQ_MAP[interval])
                df.set_index("date", inplace=True)
                if expected.difference(df.index).empty:
                    return df.reset_index()
                df.reset_index(inplace=True)
            else:
                return df
    except Exception as e:
        logger.warning(f"Error checking cached features for {stock} @ {interval}: {e}")
    finally:
        session.close()

    df_1m = fetch_stock_data(stock, interval="minute", start=start, end=end)
    if df_1m is None or df_1m.empty:
        logger.warning(f"No 1m price data for {stock}")
        return pd.DataFrame()

    df_main = process_and_insert(df_1m.copy(), stock, interval)

    if interval == "minute":
        for other_interval in ["15minute", "60minute", "day"]:
            try:
                df_down = df_1m.copy()
                df_down["interval"] = other_interval
                process_and_insert(df_down, stock, other_interval)
            except Exception as e:
                logger.warning(f"Failed to precompute for {other_interval}: {e}")

    return df_main

def fetch_features_with_backfill(stock: str, interval: str, sim_date=None) -> pd.DataFrame:
    try:
        date = sim_date if sim_date else datetime.today().date()
        df = load_data(settings.interval_feature_table_map[interval])
        if "symbol" in df.columns and "stock" not in df.columns:
            df = df.rename(columns={"symbol": "stock"})

        df = df[df["stock"] == stock]
        if sim_date is not None:
            df = df[df["date"] == sim_date]

        if not df.empty:
            return df

        logger.warning(f"Fallback to direct fetch for {stock} @ {interval}")
        end = datetime.today()
        start = end - timedelta(days=settings.price_fetch_days)
        return fetch_features(stock, interval, start=start, end=end)

    except Exception as e:
        logger.warning(f"⚠️ Backfill fallback failed for {stock} @ {interval}: {e}")
        return pd.DataFrame()
