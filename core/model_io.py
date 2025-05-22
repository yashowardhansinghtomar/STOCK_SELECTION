# core/model_io.py

import pickle
from datetime import datetime
import pandas as pd

from core.logger import logger
from core.config import settings
from db.conflict_utils import insert_with_conflict_handling
from db.postgres_manager import run_query

def save_model(name: str, model_obj, meta: dict = None) -> None:
    """
    Serialize and save a model to the configured SQL table with optional metadata.
    """
    table = settings.model_store_table
    blob = pickle.dumps(model_obj)
    meta = meta or {}

    # Ensure table exists with meta column
    run_query(f"""
    CREATE TABLE IF NOT EXISTS "{table}" (
        model_name TEXT PRIMARY KEY,
        model_blob BYTEA,
        updated_at TIMESTAMP,
        meta JSONB DEFAULT '{{}}'::jsonb
    );
    """, fetchall=False)

    # Upsert the new blob + meta
    df = pd.DataFrame([{
        "model_name": name,
        "model_blob": blob,
        "updated_at": datetime.now(),
        "meta": meta
    }])
    insert_with_conflict_handling(df, table)
    logger.success(f"📦 Model '{name}' saved to '{table}' with metadata.")

def load_model(name: str):
    """
    Load a model from the configured SQL table.
    """
    table = settings.model_store_table
    rows = run_query(
        f"SELECT model_blob FROM \"{table}\" WHERE model_name = %s LIMIT 1",
        (name,)
    )
    if not rows:
        raise FileNotFoundError(f"❌ No model found named '{name}' in '{table}'.")
    logger.info(f"📥 Loaded model '{name}' from '{table}'.")
    return pickle.loads(rows[0][0])

def get_model_metadata(name: str):
    rows = run_query(
        f"SELECT meta FROM \"{settings.model_store_table}\" WHERE model_name = %s LIMIT 1",
        (name,)
    )
    return rows[0][0] if rows else {}
