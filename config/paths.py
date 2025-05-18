# config/paths.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent / "project_data"

PATHS = {
    # --- Logs & Archive ---
    "logs": BASE_DIR / "logs",
    "archive_dir": BASE_DIR / "results" / "history",

    # --- SQLite Database ---
    "sqlite_db": BASE_DIR / "trading_system.db",
}
