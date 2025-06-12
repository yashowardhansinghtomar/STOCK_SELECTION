# db/replay_buffer_sql.py

import pandas as pd
import json
from datetime import datetime
from db.postgres_manager import run_query


class SQLReplayBuffer:
    def __init__(self):
        self.buffer = []  # Not used for memory, but kept for compatibility

    def add(self, trade_result: dict, tags: dict = None):
        tags = tags or {}
        episode = {
            "stock": trade_result["symbol"],
            "date": trade_result["entry_time"].date(),
            "interval": trade_result["meta"].get("interval", "day"),
            "action": trade_result["direction"],
            "reward": trade_result["reward"],
            "features": json.dumps(trade_result["meta"]),
            "strategy_config": json.dumps(trade_result["meta"].get("strategy_config", {})),
            "inserted_at": datetime.now()
        }
        sql = """
        INSERT INTO rl_replay_buffer
            (stock, date, interval, action, reward, features, strategy_config, inserted_at)
        VALUES
            (%(stock)s, %(date)s, %(interval)s, %(action)s, %(reward)s, %(features)s::jsonb, %(strategy_config)s::jsonb, %(inserted_at)s)
        """
        run_query(sql, params=episode, fetchall=False)

    def count_real_trades(self):
        sql = "SELECT COUNT(*) as count FROM rl_replay_buffer WHERE reward IS NOT NULL"
        rows = run_query(sql, fetchall=True)
        result = pd.DataFrame(rows)
        return result.iloc[0]["count"] if not result.empty else 0

    def load_all(self):
        sql = "SELECT * FROM rl_replay_buffer ORDER BY date"
        rows = run_query(sql, fetchall=True)
        return pd.DataFrame(rows)

    def clear(self):
        run_query("DELETE FROM rl_replay_buffer", fetchall=False)

    def size(self):
        sql = "SELECT COUNT(*) as count FROM rl_replay_buffer"
        rows = run_query(sql, fetchall=True)
        result = pd.DataFrame(rows)
        return result.iloc[0]["count"] if not result.empty else 0


# standalone helpers
def insert_replay_episode(episode: dict):
    sql = """
    INSERT INTO rl_replay_buffer
        (stock, date, interval, action, reward, features, strategy_config, inserted_at)
    VALUES
        (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)
    """
    params = (
        episode["stock"],
        episode["date"],
        episode["interval"],
        episode["action"],
        episode["reward"],
        json.dumps(episode["features"]),
        json.dumps(episode.get("strategy_config", {})),
        datetime.now()
    )
    run_query(sql, params=params, fetchall=False)


def load_replay_episodes(stock: str = None, interval: str = None):
    sql = "SELECT * FROM rl_replay_buffer"
    conditions = []
    params = []

    if stock:
        conditions.append("stock = %s")
        params.append(stock)
    if interval:
        conditions.append("interval = %s")
        params.append(interval)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    sql += " ORDER BY date"
    rows = run_query(sql, params=params, fetchall=True)
    return pd.DataFrame(rows)


def clear_old_episodes(days: int = 90):
    sql = """
    DELETE FROM rl_replay_buffer
    WHERE inserted_at < NOW() - INTERVAL '%s days'
    """
    run_query(sql, params=[days], fetchall=False)


def count_by_stock():
    sql = """
    SELECT stock, COUNT(*) as count
    FROM rl_replay_buffer
    GROUP BY stock
    ORDER BY count DESC
    """
    rows = run_query(sql, fetchall=True)
    return pd.DataFrame(rows)


def policy_converged(min_trades: int = 500, min_avg_reward: float = 0.5, max_reward_std: float = 1.0) -> bool:
    sql = """
    SELECT reward
    FROM rl_replay_buffer
    WHERE reward IS NOT NULL
    ORDER BY date DESC
    LIMIT 500
    """
    rows = run_query(sql, fetchall=True)
    df = pd.DataFrame(rows)
    if df.empty or len(df) < min_trades:
        return False

    rewards = df["reward"].astype(float)
    avg_reward = rewards.mean()
    std_reward = rewards.std()

    print(f"[POLICY CHECK] Trades={len(rewards)} | Avg={avg_reward:.3f} | Std={std_reward:.3f}")
    return avg_reward >= min_avg_reward and std_reward <= max_reward_std
