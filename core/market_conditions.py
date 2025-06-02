from core.data_provider.data_provider import fetch_stock_data
import numpy as np

def get_volatility_regime(date, symbol="NIFTYBEES", days=30):
    """
    Classifies volatility regime as 'low', 'med', or 'high' based on std dev of daily returns.
    Defaults to NIFTYBEES as market proxy.
    """
    df = fetch_stock_data(symbol, interval="day", end=date, days=days)
    if df is None or df.empty or "close" not in df:
        return "med"  # fallback

    df["returns"] = df["close"].pct_change()
    std_dev = df["returns"].std()

    if std_dev < 0.01:
        return "low"
    elif std_dev < 0.02:
        return "med"
    else:
        return "high"
