import json
import os

SKIPLIST_FILE = "skiplist_failed_stocks.json"

def load_skiplist():
    if os.path.exists(SKIPLIST_FILE):
        with open(SKIPLIST_FILE, "r") as f:
            return set(json.load(f))
    return set()

def add_to_skiplist(symbol, reason="fetch_failed"):
    from db.db_router import run_query
    query = """
    INSERT INTO skiplist_stocks (symbol, reason, imported_at)
    VALUES (%s, %s, NOW())
    ON CONFLICT (symbol)
    DO UPDATE SET reason = EXCLUDED.reason, imported_at = NOW();
    """
    run_query(query, params=(symbol, reason))

FAILED_PRECHECK_FILE = "failed_stocks.json"

def load_failed_precheck():
    if os.path.exists(FAILED_PRECHECK_FILE):
        with open(FAILED_PRECHECK_FILE, "r") as f:
            return json.load(f)
    return {}

def add_failed_precheck(stock: str, reason: str):
    failed = load_failed_precheck()
    failed[stock] = reason
    with open(FAILED_PRECHECK_FILE, "w") as f:
        json.dump(failed, f, indent=2)
