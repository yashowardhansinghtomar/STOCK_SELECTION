# workers/backfill_feature_worker.py

import redis
import time
import json
from core.feature_engineering.feature_provider import fetch_features
from core.logger.logger import logger
import os
r = redis.Redis(host=os.getenv("REDIS_HOST", "redis"), port=int(os.getenv("REDIS_PORT", "6379")), decode_responses=True)

QUEUE_NAME = "feature_backfill_queue"

while True:
    job = r.lpop(QUEUE_NAME)
    if not job:
        time.sleep(1)
        continue

    try:
        task = json.loads(job)
        stock = task["stock"]
        interval = task["interval"]
        status_key = f"feature_status:{stock}:{interval}"

        r.set(status_key, "in_progress")
        logger.info(f"🚀 Starting feature fetch: {stock} @ {interval}")

        df = fetch_features(stock, interval, refresh_if_missing=True)

        if df.empty:
            r.set(status_key, "fail")
            r.set(f"feature_error:{stock}:{interval}", "Empty features returned")
        else:
            r.set(status_key, "done")
            logger.info(f"✅ Done: {stock} @ {interval}")

    except Exception as e:
        logger.error(f"❌ Worker failed: {e}")
        r.set(status_key, "fail")
        r.set(f"feature_error:{stock}:{interval}", str(e))
