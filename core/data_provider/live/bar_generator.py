# core/data_provider/live/bar_generator.py

import redis
import json
import pandas as pd
from datetime import datetime
import time
from pytz import timezone
from core.logger.logger import logger
import os


IST = timezone("Asia/Kolkata")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

SYMBOLS = ["RELIANCE", "TCS", "INFY"]
INTERVAL_SEC = 60  # 1-minute bars

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

def get_ticks(symbol: str):
    key = f"ticks:{symbol}"
    raw = r.lrange(key, 0, -1)
    r.delete(key)  # clear after reading
    return [json.loads(x) for x in raw]

def build_ohlcv(ticks: list):
    if not ticks:
        return None

    df = pd.DataFrame(ticks)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    o = df["price"].iloc[0]
    h = df["price"].max()
    l = df["price"].min()
    c = df["price"].iloc[-1]
    v = len(df)  # simple volume = tick count

    return {
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
        "start": df.index[0].strftime("%Y-%m-%d %H:%M:%S"),
        "end": df.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
    }

def enqueue_feature_task(symbol: str, interval: str):
    key = f"feature_status:{symbol}:{interval}"
    if not r.exists(key):
        r.rpush("feature_queue", f"{symbol}|{interval}")
        r.setex(key, 6 * 60 * 60, "queued")
        logger.info(f"📬 Enqueued feature task for {symbol} @ {interval}")

def main():
    print("📈 Starting bar generator...")
    while True:
        for symbol in SYMBOLS:
            ticks = get_ticks(symbol)
            bar = build_ohlcv(ticks)
            if bar:
                ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{ts}] ⏱️ 1-min bar → {symbol}:", bar)

                # Push feature generation task for "minute" interval
                enqueue_feature_task(symbol, "minute")

        # Align with next minute
        sleep_duration = INTERVAL_SEC - datetime.now().second
        time.sleep(max(sleep_duration, 1))

if __name__ == "__main__":
    main()
