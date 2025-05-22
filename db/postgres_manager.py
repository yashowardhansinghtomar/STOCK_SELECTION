# db/postgres_manager.py

"""
PostgreSQL-compatible replacement for db_manager.py
Now using SQLAlchemy to avoid pandas warnings.
"""

import pandas as pd
from sqlalchemy import create_engine, text
from config.postgres_config import get_pg_conn_params

# ✅ Create a global engine for reuse
def get_engine():
    params = get_pg_conn_params()
    return create_engine(f"postgresql+psycopg2://{params['user']}:{params['password']}@{params['host']}:{params['port']}/{params['dbname']}")

engine = get_engine()

def read_table(table_name):
    print(f"🛠 Connected to DB: {engine.url.database} (Host: {engine.url.host})")
    print(f"🛠 Reading table: {table_name}")

    query = f'SELECT * FROM "{table_name}"'
    df = pd.read_sql(query, con=engine)

    print(f"🛠 Rows fetched from {table_name}: {len(df)} rows")
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
from db.models import Instrument
from db.db import SessionLocal

def get_all_symbols():
    session = SessionLocal()
    try:
        results = session.query(Instrument.tradingsymbol).all()
        return [r[0] for r in results]
    finally:
        session.close()