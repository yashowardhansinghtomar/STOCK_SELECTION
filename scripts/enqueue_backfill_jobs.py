# scripts/enqueue_backfill_jobs.py

import redis
import json
from db.postgres_manager import get_all_symbols
import os

r = redis.Redis(host=os.getenv("REDIS_HOST", "redis"), port=int(os.getenv("REDIS_PORT", "6379")), decode_responses=True)
QUEUE_NAME = "feature_backfill_queue"

INTERVALS = ["day", "60minute", "15minute", "minute"]
stocks = get_all_symbols()

for stock in stocks:
    for interval in INTERVALS:
        job = {"stock": stock, "interval": interval}
        r.rpush(QUEUE_NAME, json.dumps(job))
        print(f"📤 Enqueued: {stock} @ {interval}")
