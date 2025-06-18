# db/replay_buffer_sql.py

import pandas as pd
import json
from datetime import datetime
from db.postgres_manager import run_query

class SQLReplayBuffer:
    def __init__(self):
        self.buffer = []  # kept for compatibility

    def _insert_episode(self, episode: dict):
        """
        Internal helper to insert a replay episode.
        """
        if "inserted_at" not in episode:
            episode["inserted_at"] = datetime.now()

        # Ensure JSON serialization
        if isinstance(episode.get("features"), dict):
            episode["features"] = json.dumps(episode["features"])
        if isinstance(episode.get("strategy_config"), dict):
            episode["strategy_config"] = json.dumps(episode["strategy_config"])

        # Use colon‐style binds for all parameters and cast JSON strings to JSONB
        sql = """
        INSERT INTO rl_replay_buffer
            (stock, date, interval, action, reward, features, strategy_config, inserted_at)
        VALUES
            (:stock, :date, :interval, :action, :reward,
             CAST(:features AS JSONB), CAST(:strategy_config AS JSONB), :inserted_at)
        """
        run_query(sql, params=episode, fetchall=False)

    def add(self, trade_result: dict, tags: dict = None):
        """
        Historical bootstrap: merge trade_result.meta + tags into features JSON.
        """
        tags = tags or {}
        features = {**trade_result["meta"], **tags}
        episode = {
            "stock": trade_result["symbol"],
            "date": trade_result["entry_time"].date(),
            "interval": trade_result["meta"].get("interval", "day"),
            "action": trade_result["direction"],
            "reward": trade_result["reward"],
            "features": json.dumps(features),
            "strategy_config": json.dumps(trade_result["meta"].get("strategy_config", {})),
            "inserted_at": datetime.now()
        }
        self._insert_episode(episode)

    def add_episode(
        self,
        stock: str,
        date,
        state,
        action,
        reward: float,
        next_state,
        done: bool,
        interval: str = "day",
        strategy_config: dict = None,
        tags: dict = None
    ):
        """
        Live bootstrap: record state, next_state, done + optional tags.
        """
        tags = tags or {}
        sim_date = date.date() if hasattr(date, "date") else pd.to_datetime(date).date()
        features = {"state": state, "next_state": next_state, "done": done, **tags}
        episode = {
            "stock": stock,
            "date": sim_date,
            "interval": interval,
            "action": action,
            "reward": reward,
            "features": json.dumps(features),
            "strategy_config": json.dumps(strategy_config or {}),
            "inserted_at": datetime.now()
        }
        self._insert_episode(episode)

    def count_real_trades(self) -> int:
        """
        Return count of episodes where reward IS NOT NULL.
        """
        sql = "SELECT COUNT(*) FROM rl_replay_buffer WHERE reward IS NOT NULL"
        rows = run_query(sql, fetchall=True)
        return int(rows[0][0]) if rows else 0

    def size(self) -> int:
        """
        Return total number of episodes.
        """
        sql = "SELECT COUNT(*) FROM rl_replay_buffer"
        rows = run_query(sql, fetchall=True)
        return int(rows[0][0]) if rows else 0

    def load_all(self) -> pd.DataFrame:
        """
        Load all episodes, ordered by date and insertion time.
        """
        sql = "SELECT * FROM rl_replay_buffer ORDER BY date, inserted_at"
        rows = run_query(sql, fetchall=True)
        return pd.DataFrame(rows)

    def clear(self):
        """Remove all episodes."""
        run_query("DELETE FROM rl_replay_buffer", fetchall=False)

    def clear_old_episodes(self, days: int = 90):
        """
        Delete episodes older than `days` days.
        """
        interval_str = f"{days} days"
        sql = """
        DELETE FROM rl_replay_buffer
        WHERE inserted_at < NOW() - INTERVAL %s
        """
        run_query(sql, params=[interval_str], fetchall=False)


def load_replay_episodes(stock: str = None, interval: str = None) -> pd.DataFrame:
    """
    Load replay episodes, optionally filtering by stock and/or interval.
    """
    sql = "SELECT * FROM rl_replay_buffer"
    conditions, params = [], []
    if stock:
        conditions.append("stock = %s"); params.append(stock)
    if interval:
        conditions.append("interval = %s"); params.append(interval)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY date, inserted_at"
    rows = run_query(sql, params=params, fetchall=True)
    return pd.DataFrame(rows)

def count_by_stock() -> pd.DataFrame:
    """
    Return DataFrame of episode counts grouped by stock.
    """
    sql = """
    SELECT stock, COUNT(*) AS count
    FROM rl_replay_buffer
    GROUP BY stock
    ORDER BY count DESC
    """
    rows = run_query(sql, fetchall=True)
    return pd.DataFrame(rows)

def policy_converged(
    min_trades: int = 500,
    min_avg_reward: float = 0.5,
    max_reward_std: float = 1.0
) -> bool:
    """
    Determine if policy has converged based on recent rewards.
    """
    sql = """
    SELECT reward
    FROM rl_replay_buffer
    WHERE reward IS NOT NULL
    ORDER BY date DESC
    LIMIT %s
    """
    rows = run_query(sql, params=[min_trades], fetchall=True)
    df = pd.DataFrame(rows)
    if df.empty or len(df) < min_trades:
        return False
    rewards = df["reward"].astype(float)
    return (rewards.mean() >= min_avg_reward) and (rewards.std() <= max_reward_std)

from db.postgres_manager import run_query
from datetime import datetime

def update_phase(self, replay_buffer):
    real_count = replay_buffer.count_real_trades()
    converged = policy_converged()

    if self.phase == 0 and real_count > 200:
        self.phase = 1
    elif self.phase == 1 and converged:
        self.phase = 2

    # Decay epsilon
    self.epsilon = max(0.1, self.epsilon - 0.01)

    # Log phase update (use positional params → converted internally)
    run_query("""
        INSERT INTO system_phase_history (date, phase, epsilon, real_trade_count, converged, updated_at)
        VALUES (:param0, :param1, :param2, :param3, :param4, :param5)
        ON CONFLICT (date) DO UPDATE
        SET phase = EXCLUDED.phase,
            epsilon = EXCLUDED.epsilon,
            real_trade_count = EXCLUDED.real_trade_count,
            converged = EXCLUDED.converged,
            updated_at = EXCLUDED.updated_at
    """, [
        datetime.now().date(), self.phase, self.epsilon,
        real_count, converged, datetime.now()
    ], fetchall=False)
