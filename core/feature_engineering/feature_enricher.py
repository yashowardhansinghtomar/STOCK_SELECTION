
# core/feature_enricher.py
import pandas as pd
from datetime import datetime
from core.logger.logger import logger
from db.db import SessionLocal
from sqlalchemy.sql import text
from core.config.config import settings
from utils.time_utils import to_naive_utc


def enrich_features(stock: str, sim_date: datetime, interval: str = "day") -> pd.DataFrame:
    table_name = settings.interval_feature_table_map.get(interval)
    if not table_name:
        logger.error(f"❌ Unknown interval '{interval}' for feature enrichment.")
        return pd.DataFrame()

    sim_date = pd.to_datetime(sim_date).date()  # naive UTC comparison

    session = SessionLocal()
    try:
        sql = text(f"""
            SELECT * FROM {table_name}
            WHERE stock = :stock AND date = :sim_date
            LIMIT 1
        """)
        result = session.execute(sql, {"stock": stock, "sim_date": sim_date}).fetchall()

        if not result:
            logger.warning(f"⚠️ No features for {stock} on {sim_date} @ {interval}.")
            return pd.DataFrame()

        row = dict(result[0]._mapping)
        df = pd.DataFrame([row])
        df = to_naive_utc(df, "date")

        if any(pd.isnull(df.get(k)).any() for k in ["sma_short", "sma_long", "rsi_thresh"]):
            logger.warning(f"⚠️ {stock} has NaN in key features. Skipping.")
            return pd.DataFrame()

        if df.at[0, "sma_short"] == df.at[0, "sma_long"] or df.at[0, "rsi_thresh"] > 100:
            logger.warning(f"⚠️ {stock} has suspicious features. Skipping.")
            return pd.DataFrame()

        logger.info(f"✅ Features ready for {stock} on {sim_date} @ {interval}")
        return df

    except Exception as e:
        logger.error(f"❌ enrich_features() failed for {stock} @ {interval}: {e}")
        return pd.DataFrame()
    finally:
        session.close()