# enqueue_jobs.py
import redis
from db.postgres_manager import get_all_symbols
from core.config.config import settings
import os

r = redis.Redis.from_url(os.getenv("REDIS_URL", settings.REDIS_URL), decode_responses=True)
INTERVALS = ["day", "60minute", "15minute", "minute"]
TTL = 6 * 60 * 60  # 6 hours

for symbol in get_all_symbols():
    for interval in INTERVALS:
        key = f"feature_status:{symbol}:{interval}"
        if not r.exists(key):
            r.rpush("feature_queue", f"{symbol}|{interval}")
            r.setex(key, TTL, "queued")

print("✅ Jobs enqueued.")
