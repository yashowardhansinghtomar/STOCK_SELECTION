# from core.data_provider.data_provider import get_last_close
# from datetime import datetime

# # Choose a symbol and date within your known data range
# symbol = "RELIANCE"
# sim_date = datetime(2023, 6, 28)  # change this to a date where you know data exists

# price = get_last_close(symbol, sim_date)
# print(f"Close price for {symbol} on {sim_date.date()} = {price}")

# debug_inspect_trades.py








from core.data_provider.data_provider import fetch_stock_data

symbol = "SMCGLOBAL"
start = "2023-01-08"
end = "2023-01-09"

df = fetch_stock_data(symbol, interval="minute", start=start, end=end)
print(f"✅ Retrieved {len(df)} rows for {symbol} [{start} → {end}]")
print(df.head())












from core.data_provider.data_provider import fetch_stock_data

symbols = ["RELIANCE"]
for sym in symbols:
    print(f"📡 Refetching {sym}...")
    df = fetch_stock_data(sym, interval="minute", start="2023-01-03", end="2023-01-04")
    print(f"✅ {sym} → {len(df)} rows")
