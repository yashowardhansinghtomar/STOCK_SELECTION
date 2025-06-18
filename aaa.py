# from core.data_provider.data_provider import get_last_close
# from datetime import datetime

# # Choose a symbol and date within your known data range
# symbol = "RELIANCE"
# sim_date = datetime(2023, 6, 28)  # change this to a date where you know data exists

# price = get_last_close(symbol, sim_date)
# print(f"Close price for {symbol} on {sim_date.date()} = {price}")

# debug_inspect_trades.py




from core.data_provider.data_provider import fetch_stock_data, get_last_close
from datetime import datetime, timedelta
import pandas as pd
from pytz import timezone
from utils.time_utils import to_naive_utc

symbol = "PRAXIS"
ist = timezone("Asia/Kolkata")

# Step 1: Set exit time
exit_ist = ist.localize(datetime(2025, 6, 18, 15, 15))  # Trade exit_time in IST
exit_naive_utc = exit_ist.astimezone(timezone("UTC")).replace(tzinfo=None)

# Step 2: Fetch 1m candles around exit time
start = exit_naive_utc - timedelta(minutes=30)
end = exit_naive_utc + timedelta(minutes=5)
bars = fetch_stock_data(symbol, interval="minute", start=start, end=end)

if bars is not None and not bars.empty:
    bars["timestamp"] = pd.to_datetime(bars.index if bars.index.name else bars["timestamp"])
    bars = to_naive_utc(bars, "timestamp")

    found = bars[bars["timestamp"] == exit_naive_utc]
    if not found.empty:
        print(f"✅ Found exact exit candle at {exit_naive_utc}:\n{found}")
    else:
        print(f"❌ No exact match for {exit_naive_utc}. Available rows:")
        print(bars.tail(10))
else:
    print(f"⚠️ No minute bars returned — attempting daily fallback...")

# Step 3: Try daily close fallback
daily_close = get_last_close(symbol, exit_ist)
print(f"📊 Fallback daily close for {symbol} on {exit_ist.date()}: {daily_close}")







# from core.data_provider.data_provider import fetch_stock_data
# from datetime import datetime
# from pytz import timezone

# ist = timezone("Asia/Kolkata")
# start = ist.localize(datetime(2023, 1, 2, 9, 15))
# end   = ist.localize(datetime(2023, 1, 2, 15, 30))

# df = fetch_stock_data("TCS", interval="minute", start=start, end=end)
# print(df)















# from core.data_provider.data_provider import fetch_stock_data

# symbol = "SMCGLOBAL"
# start = "2023-01-08"
# end = "2023-01-09"

# df = fetch_stock_data(symbol, interval="minute", start=start, end=end)
# print(f"✅ Retrieved {len(df)} rows for {symbol} [{start} → {end}]")
# print(df.head())












# from core.data_provider.data_provider import fetch_stock_data

# symbols = ["RELIANCE"]
# for sym in symbols:
#     print(f"📡 Refetching {sym}...")
#     df = fetch_stock_data(sym, interval="minute", start="2023-01-03", end="2023-01-04")
#     print(f"✅ {sym} → {len(df)} rows")
