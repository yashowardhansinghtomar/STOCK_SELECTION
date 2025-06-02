# db/postgres_manager.py

"""
PostgreSQL-compatible replacement for db_manager.py
Now using SQLAlchemy to avoid pandas warnings.
"""
from db.models import Instrument, SkiplistStock
from db.db import SessionLocal
import pandas as pd
from sqlalchemy import create_engine, text
from core.logger.logger import logger
from typing import List
from core.config.config import settings

# ✅ Create a global engine for reuse
engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)

def read_table(table_name):
    print(f"Connected to DB: {engine.url.database} (Host: {engine.url.host})")
    print(f"Reading table: {table_name}")

    query = f'SELECT * FROM "{table_name}"'
    df = pd.read_sql(query, con=engine)

    print(f"Rows fetched from {table_name}: {len(df)} rows")
    return df

def run_query(query: str, params=None, fetchall=True):
    with engine.begin() as conn:
        stmt = text(query)

        if isinstance(params, (list, tuple)):
            params = {f"param{i}": val for i, val in enumerate(params)}
            for i in range(len(params)):
                query = query.replace('%s', f":param{i}", 1)
            stmt = text(query)

        result = conn.execute(stmt, params or {})
        if fetchall:
            return result.fetchall()
        return None

def execute_raw_sql(sql: str):
    with engine.begin() as conn:
        conn.execute(text(sql))


def insert_dataframe(df: pd.DataFrame, table_name: str, if_exists: str = "append", index: bool = False):
    """
    Bulk-insert a DataFrame into a PostgreSQL table using SQLAlchemy.
    """
    df.to_sql(
        name=table_name,
        con=engine,
        if_exists=if_exists,
        index=index,
        method="multi"         # faster batch insert
    )


def get_all_symbols(only_usable: bool = True) -> List[str]:
    session = SessionLocal()
    try:
        result = session.query(Instrument.tradingsymbol).distinct().all()
        symbols = [r[0] for r in result]

        if only_usable:
            skiplist = session.query(SkiplistStock.stock).all()
            skip_set = set(s[0] for s in skiplist)
            usable = [s for s in symbols if s not in skip_set]

            if skip_set:
                logger.info(f"Skipped {len(skip_set)} skiplist stocks — using {len(usable)} usable symbols.")
            else:
                logger.info(f"No skiplist stocks. Using all {len(usable)} symbols.")
            return usable

        return symbols
    finally:
        session.close()
