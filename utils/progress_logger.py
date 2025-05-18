# utils/progress_logger.py

import sqlite3
from datetime import datetime

def log_model_progress(sharpe_ratio, accuracy, win_rate, num_trades, db_path="project_data/trading_system.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_progress (
            date TEXT,
            sharpe_ratio REAL,
            accuracy REAL,
            num_trades INTEGER,
            win_rate REAL
        )
    """)

    cursor.execute("""
        INSERT INTO model_progress (date, sharpe_ratio, accuracy, num_trades, win_rate)
        VALUES (?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        sharpe_ratio,
        accuracy,
        num_trades,
        win_rate
    ))

    conn.commit()
    conn.close()
    print("✅ Model performance logged to SQL")
