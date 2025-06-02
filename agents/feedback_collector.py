from core.event_bus import subscribe_to_events
from db.replay_buffer_sql import insert_replay_episode
from core.logger.logger import logger

def handle_trade_close(event):
    if event.get("event_type") != "TRADE_CLOSE":
        return

    try:
        episode = {
            "stock": event["symbol"],
            "date": event["timestamp"][:10],
            "features": {},  # Optional: can later cache features here
            "action": 1,  # Since it's a close, assume "hold to exit"
            "reward": float(event.get("reward", 0)),
            "interval": event.get("interval", "day"),
            "strategy_config": event.get("strategy_config", {}),
            "done": True,
            "missed_pnl": float(event.get("missed_pnl", 0)),
            "holding_cost": float(event.get("holding_cost", 0)),
            "slippage_penalty": float(event.get("slippage_penalty", 0)),
            "capital_efficiency": float(event.get("capital_efficiency", 0)),
        }

        insert_replay_episode(episode)
        logger.info(f"[FEEDBACK COLLECTOR] Logged feedback for {episode['stock']}")

    except Exception as e:
        logger.error(f"[FEEDBACK COLLECTOR] ⚠️ Error logging feedback: {e}")

if __name__ == "__main__":
    subscribe_to_events(handle_trade_close)
