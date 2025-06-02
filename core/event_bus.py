# core/event_bus.py
import redis
import json
from datetime import datetime

r = redis.Redis(decode_responses=True)

def publish_event(event_type: str, payload: dict):
    payload['event_type'] = event_type
    payload['timestamp'] = datetime.utcnow().isoformat()
    r.xadd("event_stream", payload)

def subscribe_to_events(callback, last_id='0'):
    while True:
        response = r.xread({"event_stream": last_id}, block=0)
        for stream, messages in response:
            for msg_id, data in messages:
                callback(data)
                last_id = msg_id
