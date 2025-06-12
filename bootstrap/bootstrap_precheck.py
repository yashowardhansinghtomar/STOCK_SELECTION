# bootstrap/bootstrap_precheck.py

import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from core.config.config import settings
from core.data_provider.data_provider import fetch_stock_data
from db.postgres_manager import get_all_symbols


def check_model_path():
    path = Path("models/filter_model.lgb")
    return path.exists(), f"{path} exists" if path.exists() else f"{path} not found (expected if retraining)"


def check_database_connection():
    try:
        engine = create_engine(settings.database_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "Connected"
    except Exception as e:
        return False, str(e)


def check_price_history():
    try:
        df = fetch_stock_data("ICICIBANK", interval="day", start="2022-12-01", end="2022-12-31")
        if df is None or df.empty:
            return False, "No historical day data for ICICIBANK (Dec 2022)"
        return True, f"{len(df)} rows from Dec 2022"
    except Exception as e:
        return False, str(e)


def check_minute_data():
    try:
        # Jan 2, 2023 was a Monday (market open)
        df = fetch_stock_data("ICICIBANK", interval="minute", start="2023-01-02", end="2023-01-03")
        if df is None or df.empty:
            return False, "Minute data not available for ICICIBANK (Jan 2)"
        return True, f"{len(df)} rows of minute data"
    except Exception as e:
        return False, str(e)


def check_replay_buffer_table():
    try:
        engine = create_engine(settings.database_url)
        df = pd.read_sql("SELECT COUNT(*) as count FROM rl_replay_buffer", con=engine)
        return True, f"{df.iloc[0]['count']} trades already in buffer"
    except Exception as e:
        return False, str(e)


def check_filter_stock_universe():
    try:
        symbols = get_all_symbols()
        if not symbols:
            return False, "get_all_symbols() returned nothing"
        return True, f"{len(symbols)} symbols available"
    except Exception as e:
        return False, str(e)


def main():
    checks = [
        ("✅ Filter model already exists?", check_model_path),
        ("🔌 PostgreSQL database connection", check_database_connection),
        ("📈 Day-level history present", check_price_history),
        ("🕐 Minute-level history present", check_minute_data),
        ("📦 Replay buffer table accessible", check_replay_buffer_table),
        ("🔍 Filter model stock universe available", check_filter_stock_universe),
    ]

    print("\n🧪 Pre-Bootstrap System Health Check\n" + "-" * 40)
    for name, fn in checks:
        ok, msg = fn()
        status = "✅" if ok else "❌"
        print(f"{status} {name}: {msg}")
        if not ok:
            print("   ⚠️  Fix this before running bootstrap!\n")

    print("-" * 40)


if __name__ == "__main__":
    main()
