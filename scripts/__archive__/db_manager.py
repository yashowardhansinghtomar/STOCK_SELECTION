import sqlite3
import pandas as pd
import time
from config.paths import PATHS
from core.logger.logger import logger

DB_PATH = PATHS["sqlite_db"]
WAL_ENABLED = False  # module-level flag

# --- Blocking WAL-safe connection ---
def get_connection():
    """Block until connection is possible and WAL is enabled."""
    global WAL_ENABLED
    while True:
        try:
            logger.debug("🔌 Attempting DB connection...")
            conn = sqlite3.connect(DB_PATH, timeout=10)

            if not WAL_ENABLED:
                try:
                    mode = conn.execute("PRAGMA journal_mode=WAL;").fetchone()[0]
                    logger.debug(f"🔍 Current journal_mode: {mode}")
                    if mode.lower() == "wal":
                        WAL_ENABLED = True
                        logger.info("✅ WAL mode enabled")
                    else:
                        logger.warnings(f"⚠️ WAL not enabled, current mode: {mode}")
                except Exception as e:
                    logger.warnings(f"⚠️ Could not enable WAL: {e}")

            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA temp_store = MEMORY;")
            logger.debug("✅ DB connection established")
            return conn

        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower():
                logger.warnings("🔒 DB locked during connect – retrying in 5s...")
                time.sleep(5)
            else:
                logger.error(f"❌ Unexpected DB error: {e}")
                raise

# --- Smart INSERT with retry ---
def insert_dataframe(df, table, if_exists="append", max_retries=5):
    logger.debug(f"📝 Preparing to insert dataframe into '{table}'...")
    if if_exists == "append":
        if table == "training_data":
            df = df.drop_duplicates(subset=["date", "stock"], keep="last")
        elif table == "recommendations":
            df = df.drop_duplicates(subset=["date", "stock"], keep="last")
        elif table in ["open_positions", "fundamentals", "stock_labels"]:
            df = df.drop_duplicates(subset=["stock"], keep="last")

    for attempt in range(max_retries):
        try:
            conn = get_connection()
            logger.debug(f"📥 Inserting {len(df)} rows into {table}...")
            df.to_sql(table, conn, if_exists=if_exists, index=False, method="multi")
            logger.info(f"💾 Inserted {len(df)} rows into {table} (if_exists={if_exists})")
            return
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower():
                wait = 2 ** attempt
                logger.warnings(f"🔁 DB locked – retrying in {wait}s (attempt {attempt + 1})")
                time.sleep(wait)
            else:
                logger.error(f"❌ SQLite error during insert: {e}")
                break
        finally:
            try:
                conn.close()
            except:
                logger.warnings("⚠️ Failed to close DB connection after insert attempt.")
    logger.error(f"❌ Failed to insert into {table} after {max_retries} retries.")

# --- Read entire table ---
def read_table(table):
    logger.debug(f"📖 Reading table: {table}")
    conn = get_connection()
    try:
        return pd.read_sql(f"SELECT * FROM {table}", conn)
    finally:
        conn.close()

# --- General-purpose query runner ---
def run_query(query, params=None):
    logger.debug(f"💬 Running query: {query}")
    conn = get_connection()
    try:
        cursor = conn.cursor()
        if params:
            logger.debug(f"📦 With params: {params}")
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        result = cursor.fetchall()
        conn.commit()
        logger.debug("✅ Query executed successfully")
        return result
    finally:
        conn.close()

# --- Utility: list all table names ---
def list_tables():
    logger.debug("📋 Listing all tables")
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()

# --- For manual fixes or schema changes ---
def execute_raw_sql(sql):
    logger.debug(f"🛠️ Executing raw SQL: {sql}")
    with get_connection() as conn:
        conn.execute(sql)
        conn.commit()


def enable_wal_mode(db_path=DB_PATH):
    logger.debug("🔧 Forcing WAL enablement")
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
        print("✅ WAL mode enabled.")
    except sqlite3.OperationalError as e:
        print(f"⚠️ Could not enable WAL: {e}")
