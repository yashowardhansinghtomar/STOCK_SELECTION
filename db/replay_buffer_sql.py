from db.postgres_manager import run_query
import pandas as pd
import json
from datetime import datetime

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

    return run_query(sql, params=params, fetchall=True, as_dataframe=True)

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
    return run_query(sql, fetchall=True, as_dataframe=True)