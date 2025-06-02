# utils/stock_health_precheck.py

from core.data_provider.data_provider import load_data
from core.logger.logger import logger

def is_stock_tradeable(stock: str, verbose=False) -> bool:
    try:
        df = load_data(stock)
        if df is None or df.empty:
            if verbose: logger.debug(f"⛔ {stock}: No price data.")
            return False

        df.columns = [col.lower() for col in df.columns]

        if "close" not in df.columns or "volume" not in df.columns:
            if verbose: logger.debug(f"⛔ {stock}: Missing close/volume columns.")
            return False

        if df["close"].dropna().shape[0] < 100:
            if verbose: logger.debug(f"⛔ {stock}: Not enough history (<100).")
            return False

        if (df["volume"] <= 0).sum() > 10:
            if verbose: logger.debug(f"⛔ {stock}: Too many zero-volume days.")
            return False

        if df["close"].tail(30).isna().any():
            if verbose: logger.debug(f"⛔ {stock}: Missing recent prices.")
            return False
        if (df["volume"].tail(30) <= 0).any():
            if verbose: logger.debug(f"⛔ {stock}: Recent zero-volume trading.")
            return False

        return True

    except Exception as e:
        if verbose: logger.debug(f"⛔ {stock}: Precheck error {e}")
        return False
