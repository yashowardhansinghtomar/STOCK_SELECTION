from core.data_provider.data_provider import fetch_stock_data

if __name__ == "__main__":
    df = fetch_stock_data(symbol="TCS", interval="day", days=30)
    if df is not None and not df.empty:
        print(f"✅ Got {len(df)} rows from {df.index.min().date()} to {df.index.max().date()}")
        print(df.tail())
    else:
        print("❌ No data fetched.")

from core.data_provider.data_provider import fetch_stock_data

if __name__ == "__main__":
    df = fetch_stock_data(symbol="TCS", interval="day", end="2025-05-21", days=60)
    print(df[df.index > "2025-05-15"])
