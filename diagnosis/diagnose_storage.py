# diagnose_storage.py

import os, sqlite3
from core.data_provider.data_provider import load_data
from pathlib import Path

DB_FILE = Path("project_data") / "trading_system.db"

def table_exists(table_name):
    if not DB_FILE.exists(): return False
    conn = sqlite3.connect(DB_FILE)
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    exists = cur.fetchone() is not None
    conn.close()
    return exists

print(f"{'key':25s} {'type':8s} {'path/or table':40s} exists?")
print("-" * 80)
for key, loc in load_data.items():
    loc_str = str(loc)
    if loc_str.endswith(".pkl") or loc_str.endswith(".csv"):
        kind = loc_str.split(".")[-1]
        exists = os.path.exists(loc_str)
    else:
        kind = "sql‑table"
        exists = table_exists(key)
    print(f"{key:25s} {kind:8s} {loc_str:40s} {exists}")
