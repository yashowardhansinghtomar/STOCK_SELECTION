from core.data_provider import fetch_stock_data

def get_prices(symbol: str, start: str = None, end: str = None, interval: str = "day"):
    df = fetch_stock_data(symbol, start=start, end=end, interval=interval)
    if df is None or df.empty:
        return None
    return df[["open", "high", "low", "close", "volume"]].reset_index()
