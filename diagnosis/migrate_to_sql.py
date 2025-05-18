"""
Migration script to import all file-based assets (CSV, PKL, JSON) into your SQLite DB for a fully SQL‑backed system.

- CSV → SQL tables via `insert_dataframe`
- PKL → `model_store` BLOB table
- JSON → `json_store` TEXT table
"""
import os
import pickle
import pandas as pd
import sqlite3
from config.paths import PATHS
from db.db_router import insert_dataframe, DB_PATH

# Table definitions
MODEL_TABLE = "model_store"
JSON_TABLE = "json_store"


def ensure_blob_json_tables():
    """Create the blob and JSON store tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cur = conn.cursor()
    # BLOB store for pickled models
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {MODEL_TABLE} (
            name TEXT PRIMARY KEY,
            model_blob BLOB
        )
    """
    )
    # TEXT store for JSON metadata
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {JSON_TABLE} (
            name TEXT PRIMARY KEY,
            content TEXT
        )
    """
    )
    conn.commit()
    conn.close()


def migrate_csv(key, loc_str):
    df = pd.read_csv(loc_str)
    insert_dataframe(df, key, if_exists='replace')
    print(f"Migrated CSV '{loc_str}' → table '{key}' ({len(df)} rows)")


def migrate_pkl(key, loc_str):
    with open(loc_str, 'rb') as f:
        blob = f.read()
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cur = conn.cursor()
    cur.execute(
        f"REPLACE INTO {MODEL_TABLE} (name, blob) VALUES (?, ?)",
        (key, sqlite3.Binary(blob))
    )
    conn.commit()
    conn.close()
    print(f"Migrated PKL '{loc_str}' → {MODEL_TABLE}['{key}']")


def migrate_json(key, loc_str):
    with open(loc_str, 'r', encoding='utf-8') as f:
        content = f.read()
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cur = conn.cursor()
    cur.execute(
        f"REPLACE INTO {JSON_TABLE} (name, content) VALUES (?, ?)",
        (key, content)
    )
    conn.commit()
    conn.close()
    print(f"Migrated JSON '{loc_str}' → {JSON_TABLE}['{key}']")


def migrate_to_sql():
    ensure_blob_json_tables()
    for key, loc in PATHS.items():
        loc_str = str(loc)

        if loc_str.lower().endswith('.csv'):
            if not os.path.exists(loc_str):
                print(f"Skipping {key}: CSV not found at {loc_str}")
                continue
            try:
                migrate_csv(key, loc_str)
            except Exception as e:
                print(f"❌ Failed to migrate CSV '{key}': {e}")

        elif loc_str.lower().endswith('.pkl'):
            if not os.path.exists(loc_str):
                print(f"Skipping {key}: PKL not found at {loc_str}")
                continue
            try:
                migrate_pkl(key, loc_str)
            except Exception as e:
                print(f"❌ Failed to migrate PKL '{key}': {e}")

        elif loc_str.lower().endswith('.json'):
            if not os.path.exists(loc_str):
                print(f"Skipping {key}: JSON not found at {loc_str}")
                continue
            try:
                migrate_json(key, loc_str)
            except Exception as e:
                print(f"❌ Failed to migrate JSON '{key}': {e}")

        else:
            print(f"Skipping {key}: not a file asset (path: {loc_str})")


if __name__ == '__main__':
    migrate_to_sql()
