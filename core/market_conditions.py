# core/market_conditions.py

from core.data_provider.data_provider import fetch_stock_data
import numpy as np

TOP_NIFTY_STOCKS = ["RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "LT", "SBIN"]

def get_volatility_regime(date, days=30):
    """
    Classifies volatility regime using avg std dev of daily returns
    across top index-weighted stocks.
    """
    returns = []
    for symbol in TOP_NIFTY_STOCKS:
        df = fetch_stock_data(symbol, interval="day", end=date, days=days)
        if df is not None and not df.empty and "close" in df:
            df["returns"] = df["close"].pct_change()
            returns += df["returns"].dropna().tolist()

    if not returns:
        return "med"

    std_dev = np.std(returns)

    if std_dev < 0.01:
        return "low"
    elif std_dev < 0.02:
        return "med"
    else:
        return "high"
