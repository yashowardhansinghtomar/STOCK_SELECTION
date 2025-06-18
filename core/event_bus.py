# core/event_bus.py

import json
from datetime import datetime
from core.logger.logger import logger

try:
    import redis
    r = redis.Redis(decode_responses=True)
    r.ping()
    REDIS_ENABLED = True
except Exception as e:
    REDIS_ENABLED = False
    logger.warning(f"⚠️ Redis not available — EventBus disabled: {e}")

def publish_event(event_type: str, payload: dict):
    payload = payload.copy()  # avoid modifying original
    payload['event_type'] = event_type
    payload['timestamp'] = datetime.utcnow().isoformat()

    if REDIS_ENABLED:
        try:
            r.xadd("event_stream", payload)
            logger.debug(f"📡 Event published to Redis: {event_type}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to publish event: {e}")
    else:
        logger.debug(f"[EventBus DISABLED] {event_type}: {payload}")

def subscribe_to_events(callback, last_id='0'):
    if not REDIS_ENABLED:
        logger.warning("⚠️ Redis subscription skipped — EventBus disabled.")
        return

    while True:
        try:
            response = r.xread({"event_stream": last_id}, block=0)
            for stream, messages in response:
                for msg_id, data in messages:
                    callback(data)
                    last_id = msg_id
        except Exception as e:
            logger.warning(f"⚠️ Redis subscription error: {e}")
            break
