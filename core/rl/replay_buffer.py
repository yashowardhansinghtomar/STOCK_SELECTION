# core/rl/replay_buffer.py

import json
import pandas as pd
from datetime import datetime
from collections import deque
from db.postgres_manager import insert_rows
from core.logger.logger import logger
from core.time_context.time_context import get_simulation_date

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, stock, interval, state, action, reward, done, info):
        self.buffer.append({
            "stock": stock,
            "interval": interval,
            "state": state,
            "action": action,
            "reward": reward,
            "done": done,
            "info": info,
            "date": get_simulation_date(),
            "timestamp": datetime.now()
        })

    def flush_to_sql(self, table="replay_buffer"):
        if not self.buffer:
            logger.info("[REPLAY BUFFER] Nothing to flush.")
            return

        rows = []
        for row in self.buffer:
            rows.append({
                "stock": row["stock"],
                "interval": row["interval"],
                "date": row["date"],
                "reward": row["reward"],
                "done": row["done"],
                "timestamp": row["timestamp"],
                "state": json.dumps(row["state"]),
                "action": json.dumps(row["action"]),
                "reason": row["info"].get("reason", None),
                "reward_breakdown": json.dumps(row["info"].get("reward_breakdown", {}))
            })

        df = pd.DataFrame(rows)
        insert_rows(df, table)
        logger.success(f"[REPLAY BUFFER] Flushed {len(rows)} rows to SQL.")
        self.buffer.clear()
