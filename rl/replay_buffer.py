# rl/replay_buffer.py
import pandas as pd
import os
from datetime import datetime, timedelta

class ReplayBuffer:
    def __init__(self, path="data/replay_buffer.parquet"):
        self.path = path
        self.buffer = pd.DataFrame(columns=["stock", "date", "state", "action", "reward", "next_state", "done"])
        if os.path.exists(self.path):
            self.buffer = pd.read_parquet(self.path)

    def add_episode(self, stock, date, state, action, reward, next_state, done):
        row = {
            "stock": stock,
            "date": pd.to_datetime(date),
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        }
        self.buffer = pd.concat([self.buffer, pd.DataFrame([row])], ignore_index=True)
        self.buffer.to_parquet(self.path, index=False)

    def sample(self, n=1000):
        return self.buffer.sample(n=min(n, len(self.buffer)))

    def clear_old(self, days=30):
        cutoff = datetime.now() - timedelta(days=days)
        self.buffer = self.buffer[self.buffer["date"] > cutoff]
        self.buffer.to_parquet(self.path, index=False)

    def __len__(self):
        return len(self.buffer)


