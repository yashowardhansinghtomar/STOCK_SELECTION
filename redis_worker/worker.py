# worker.py
import redis
import time
import traceback
from core.feature_engineering.feature_provider import fetch_features
from core.logger.logger import logger
import os
print("🐛 RAW ENV DB URL:", os.getenv("database_url"))
from core.config.config import settings
print("🧪 Settings DB URL:", settings.database_url)

r = redis.Redis.from_url(os.getenv("REDIS_URL", settings.REDIS_URL), decode_responses=True)
RETRY_DELAY = 10  # seconds
TTL = 6 * 60 * 60  # 6 hours

while True:
    task = r.lpop("feature_queue")
    if task is None:
        time.sleep(5)
        continue

    symbol, interval = task.decode().split("|")
    status_key = f"feature_status:{symbol}:{interval}"

    try:
        logger.info(f"🔧 Processing {symbol} @ {interval}")
        fetch_features(symbol, interval, refresh_if_missing=True)
        r.setex(status_key, TTL, "done")
        logger.success(f"✅ Done {symbol} @ {interval}")
    except Exception as e:
        error_msg = f"❌ Error for {symbol} @ {interval}: {e}"
        logger.error(error_msg)
        traceback.print_exc()
        r.setex(status_key, TTL, f"error: {e}")
        time.sleep(RETRY_DELAY)
