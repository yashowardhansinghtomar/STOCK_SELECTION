# tick_collector_redis.py

from kiteconnect import KiteTicker
from datetime import datetime
import redis
import json
import os
from pytz import timezone

API_KEY = os.getenv("ZERODHA_API_KEY")
ACCESS_TOKEN = os.getenv("ZERODHA_ACCESS_TOKEN")

INSTRUMENT_TOKENS = {
    "RELIANCE": 738561,
    "TCS": 2953217,
    "INFY": 408065,
}

IST = timezone("Asia/Kolkata")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

def on_ticks(ws, ticks):
    for tick in ticks:
        token = tick['instrument_token']
        symbol = next((s for s, t in INSTRUMENT_TOKENS.items() if t == token), str(token))
        ltp = tick.get("last_price")
        ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

        tick_data = {"price": ltp, "timestamp": ts}
        redis_key = f"ticks:{symbol}"

        r.rpush(redis_key, json.dumps(tick_data))
        print(f"[{ts}] {symbol} → ₹{ltp}")

def on_connect(ws, response):
    print("✅ Connected to Zerodha WebSocket")
    ws.subscribe(list(INSTRUMENT_TOKENS.values()))
    ws.set_mode(ws.MODE_FULL, list(INSTRUMENT_TOKENS.values()))

def on_close(ws, code, reason):
    print("🔌 Disconnected:", code, reason)

def on_error(ws, code, reason):
    print("❌ Error:", code, reason)

def main():
    kws = KiteTicker(API_KEY, ACCESS_TOKEN)
    kws.on_ticks = on_ticks
    kws.on_connect = on_connect
    kws.on_close = on_close
    kws.on_error = on_error

    try:
        kws.connect(threaded=False)
    except KeyboardInterrupt:
        print("🛑 Exiting gracefully")

if __name__ == "__main__":
    main()
