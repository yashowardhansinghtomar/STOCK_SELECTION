# core/redis_utils.py

import redis
from core.config.config import settings
import time
from core.logger.logger import logger
import os
r = redis.Redis.from_url(os.getenv("REDIS_URL", settings.REDIS_URL), decode_responses=True)
TTL = 6 * 60 * 60  # 6 hours

def enqueue_feature_backfill(symbol: str, interval: str) -> bool:
    """
    Enqueue a feature backfill job for a specific symbol and interval.
    Returns True if enqueued, False if already in progress or done.
    """
    key = f"feature_status:{symbol}:{interval}"
    if r.exists(key):
        return False  # already queued, working, or done

    r.rpush("feature_queue", f"{symbol}|{interval}")
    r.setex(key, TTL, "queued")
    return True


def wait_for_feature_ready(symbol: str, interval: str, timeout: int = 300, poll_interval: int = 5) -> str:
    """
    Block until the feature job finishes or timeout hits.
    Returns: 'done', 'error: <msg>', or 'timeout'
    """
    key = f"feature_status:{symbol}:{interval}"
    waited = 0

    while waited < timeout:
        status = r.get(key)
        if status:
            decoded = status.decode()
            if decoded.startswith("done") or decoded.startswith("error"):
                return decoded
        time.sleep(poll_interval)
        waited += poll_interval

    return "timeout"

def enqueue_feature_backfill(symbol: str, interval: str, ttl: int = TTL):
    """
    Queue a symbol for feature backfill unless already enqueued.
    """
    key = f"feature_status:{symbol}:{interval}"
    if not r.exists(key):
        r.rpush("feature_queue", f"{symbol}|{interval}")
        r.setex(key, ttl, "queued")
        logger.info(f"📥 Enqueued {symbol} @ {interval}")
    else:
        logger.debug(f"⏭️ Already enqueued or processed: {symbol} @ {interval}")
