# monitor/feature_status_dashboard.py
import os
import redis
import time

r = redis.Redis(host=os.getenv("REDIS_HOST", "redis"), port=int(os.getenv("REDIS_PORT", "6379")), decode_responses=True)

def check_status(stock, interval):
    status = r.get(f"feature_status:{stock}:{interval}")
    error = r.get(f"feature_error:{stock}:{interval}")
    return status or "pending", error

stocks = ["TCS", "INFY", "RELIANCE"]  # Example
intervals = ["day", "60minute", "15minute", "minute"]

for stock in stocks:
    for interval in intervals:
        status, error = check_status(stock, interval)
        print(f"{stock} @ {interval} → {status.upper()}")
        if error:
            print(f"   ⚠️ {error}")
