# agents/replay_logger.py

from core.event_bus import subscribe_to_events
from db.db import SessionLocal
from db.replay_buffer_sql import insert_replay_episode
from core.logger.logger import logger
import pandas as pd

def handle_event(event):
    if event["event_type"] != "TRADE_CLOSE":
        return

    try:
        episode = {
            "stock": event["symbol"],
            "date": event["timestamp"][:10],
            "features": {},  # Optional: preload from cache
            "action": 1,
            "reward": float(event.get("reward", 0)),
            "interval": event.get("interval", "day"),
            "strategy_config": event.get("strategy_config", {}),
            "done": True,

            # 🎯 Enriched reward signals for PPO training
            "missed_pnl": float(event.get("missed_pnl", 0)),
            "holding_cost": float(event.get("holding_cost", 0)),
            "slippage_penalty": float(event.get("slippage_penalty", 0)),
            "capital_efficiency": float(event.get("capital_efficiency", 0)),

            # 📈 Optional: regime awareness (VIX tag, etc.)
            "regime_tag": event.get("regime_tag"),
        }

        logger.info(
            f"[REPLAY LOGGER] {episode['stock']} | reward: {episode['reward']:.2f} | "
            f"missed: {episode['missed_pnl']:.2f}, hold: {episode['holding_cost']:.2f}, "
            f"slip: {episode['slippage_penalty']:.2f}, eff: {episode['capital_efficiency']:.2f}"
        )
        insert_replay_episode(episode)

    except Exception as e:
        logger.error(f"[REPLAY LOGGER] ⚠️ Failed to insert replay episode: {e}")

def log_replay_row(stock, action, reason, model=None, prediction=None, confidence=None, signal=None, date=None):
    """
    For logging rejected or skipped signals, e.g., during arbitration.
    """
    from db.postgres_manager import run_query
    query = """
    INSERT INTO replay_buffer
    (date, stock, action, reason, model, prediction, confidence, signal)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    run_query(query, params=(date, stock, action, reason, model, prediction, confidence, signal), fetchall=False)

if __name__ == "__main__":
    subscribe_to_events(handle_event)
