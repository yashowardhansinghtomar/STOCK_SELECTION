# core/data_provider.py

from typing import Any, List, Optional
from datetime import datetime
import pandas as pd

from core.logger import logger
from core.config import settings
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
    StockFeature,
    MLSelectedStock,           # ← imported your new model
)

# Map logical table names to SQLAlchemy models
ORM_MODEL_MAP = {
    settings.instruments_table:       Instrument,
    settings.skiplist_table:          SkiplistStock,
    settings.fundamentals_table:      StockFundamental,
    settings.recommendations_table:   Recommendation,
    settings.open_positions_table:    OpenPosition,
    settings.paper_trades_table:      PaperTrade,
    settings.encoding_table:          StockEncoding,
    settings.price_history_table:     StockPriceHistory,
    settings.feature_table:           StockFeature,
    settings.ml_selected_stocks_table: MLSelectedStock,  # ← now supported
}


def load_data(table_name: str) -> pd.DataFrame:
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

        # drop SQLAlchemy internals
        df = df.drop(columns=[c for c in df.columns if c.startswith("_sa_")], errors="ignore")

        # parse any datetime columns
        for dt_col in settings.date_columns.get(table_name, []):
            if dt_col in df.columns:
                df[dt_col] = pd.to_datetime(df[dt_col])

        # ─── SPECIAL: ensure a 'stock' column where appropriate ───
        # features often come in as 'symbol'
        if table_name in {settings.feature_table, settings.ml_selected_stocks_table}:
            if "symbol" in df.columns and "stock" not in df.columns:
                df = df.rename(columns={"symbol": "stock"})
        # fundamentals come in as 'stock' already

        return df

    except Exception as e:
        logger.error(f"❌ load_data('{table_name}') failed: {e}")
        return pd.DataFrame()
    finally:
        session.close()


def save_data(df: pd.DataFrame, table_name: str, if_exists: str = "append") -> None:
    if df is None or df.empty:
        logger.warning(f"⚠️ Not saving '{table_name}': DataFrame is empty.")
        return

    phys = settings.table_map.get(table_name, table_name)
    try:
        insert_with_conflict_handling(df, phys, if_exists=if_exists)
        logger.success(f"📅 Saved data to '{phys}' successfully.")
    except Exception as e:
        logger.error(f"❌ save_data to '{phys}' failed: {e}")


def fetch_stock_data(
    symbol: str,
    end: str = None,
    interval: str = None,
    days: int = None
) -> pd.DataFrame:
    end = end or datetime.now().strftime("%Y-%m-%d")
    interval = interval or settings.price_fetch_interval
    days = days or settings.price_fetch_days

    session = SessionLocal()
    try:
        Model = ORM_MODEL_MAP[settings.price_history_table]
        recs: List[Any] = (
            session.query(Model)
                   .filter(Model.symbol == symbol)
                   .filter(Model.date <= pd.to_datetime(end))
                   .all()
        )
        if len(recs) >= settings.price_cache_min_rows:
            df = (
                pd.DataFrame([r.__dict__ for r in recs])
                  .drop(columns=["_sa_instance_state"], errors="ignore")
            )
            df["date"] = pd.to_datetime(df["date"])
            logger.debug(f"📆 Loaded {len(df)} rows for {symbol} from SQL")
            return df.set_index("date").sort_index()
    except Exception as e:
        logger.warning(f"⚠️ Could not load cached data for {symbol}: {e}")
    finally:
        session.close()

    logger.info(f"🌐 Fetching fresh price data for {symbol}")
    fresh = fetch_historical_data(symbol, end=end, interval=interval, days=days)
    if fresh is not None and not fresh.empty:
        save_data(fresh, settings.price_history_table, if_exists="append")
        return fresh.set_index("date").sort_index()

    logger.warning(f"⚠️ No data fetched for {symbol}")
    return pd.DataFrame()


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
        logger.error(f"❌ get_last_close('{symbol}') failed: {e}")
        return None
    finally:
        session.close()


def delete_cached_features(stock: str, sim_date: datetime) -> None:
    session = SessionLocal()
    try:
        Model = ORM_MODEL_MAP[settings.feature_table]
        session.query(Model) \
               .filter(Model.stock == stock) \
               .filter(Model.date == sim_date) \
               .delete()
        session.commit()
        logger.success(f"🗑️ Deleted cached features for {stock} on {sim_date}")
    except Exception as e:
        logger.error(f"❌ delete_cached_features failed: {e}")
    finally:
        session.close()


def list_partitions(base_table_prefix: str = None) -> List[str]:
    prefix = base_table_prefix or settings.feature_table
    session = SessionLocal()
    try:
        result = session.execute(
            f"SELECT tablename FROM pg_tables WHERE tablename LIKE '{prefix}_%';"
        )
        return [row[0] for row in result]
    except Exception as e:
        logger.error(f"❌ list_partitions failed: {e}")
        return []
    finally:
        session.close()
