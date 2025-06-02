# utils/progress_logger.py

import sqlite3
from datetime import datetime

DB_PATH = "data/progress_log.db"

def log_model_progress(model_name: str, loss: float, buffer_size: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_training_progress (
            timestamp TEXT,
            model TEXT,
            loss REAL,
            buffer_size INT
        )
    """)
    cursor.execute(
        "INSERT INTO model_training_progress (timestamp, model, loss, buffer_size) VALUES (?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), model_name, loss, buffer_size)
    )
    conn.commit()
    conn.close()
